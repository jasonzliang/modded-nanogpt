"""
M1 Mac MPS-compatible version of train_gpt.py

Key changes from CUDA version:
1. Uses MPS (Metal Performance Shaders) instead of CUDA
2. Single GPU only (no distributed training)
3. PyTorch native operations instead of Triton kernels
4. No FP8 operations (uses bfloat16/float16)
5. PyTorch SDPA instead of flash_attn
6. Simplified optimizer without NCCL communication
"""

import os
import sys

# Read the current file for logging
with open(sys.argv[0], 'r') as f:
    code = f.read()

import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
import gc

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import MPS-compatible kernel replacements
from mps_kernels import XXT, ba_plus_cAA, FusedLinearReLUSquareFunction, FusedSoftcappedCrossEntropy

# -----------------------------------------------------------------------------
# Device setup for M1 Mac

def get_device():
    """Get the best available device for M1 Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        print("WARNING: MPS not available, falling back to CPU")
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Disable distributed training for M1 (single GPU)
world_size = 1
rank = 0
grad_accum_steps = 8  # Accumulate gradients to simulate larger batch

# -----------------------------------------------------------------------------
# Linear layer (no FP8 on MPS)

class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features).
    MPS version: no FP8 support, uses standard matmul.
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # FP8 not supported on MPS - ignore these params
        self.use_fp8 = False

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight)

    def forward(self, x: Tensor):
        return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# Polar Express (Newton-Schulz orthogonalization)

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor, split_baddbmm: bool = False):
    """
    Polar Express Sign Method for orthogonalization.
    MPS-compatible version using PyTorch native operations.
    """
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    # Select batched vs unbatched
    if split_baddbmm:
        BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
    else:
        aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the iterations
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)  # A = X @ X.mT
        ba_plus_cAA(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A

        if split_baddbmm:
            BX_matmul(B, X, out=C)
            C.add_(X, alpha=a)
        else:
            aX_plus_BX(X, B, X, beta=a, out=C)

        X, C = C, X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Simplified Optimizer for M1 (no distributed communication)

@dataclass
class ParamConfig:
    """Per-parameter configuration for optimizer."""
    label: str
    optim: str  # "adam" or "normuon"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    eps: float | None = None
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


class NorMuonAndAdam:
    """
    Combined optimizer for M1 Mac (single GPU, no distributed).
    Handles both NorMuon (for projection matrices) and Adam (for embeddings/scalars).
    """
    def __init__(self, named_params, param_table: dict, work_order: list,
                 adam_defaults: dict, normuon_defaults: dict):
        self.adam_defaults = adam_defaults
        self.normuon_defaults = normuon_defaults
        self.param_table = param_table
        self.work_order = work_order

        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}

        for name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table
            assert label not in self._param_by_label
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        present = set(self._param_by_label.keys())
        assert set(work_order) == present

        self._init_state()

        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self.split_embed = False
        self._lm_head_param = self._param_by_label.get("lm_head")
        self._embed_param = self._param_by_label.get("embed")

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        table_entry = self.param_table[label]
        optim = table_entry["optim"]
        adam_betas = table_entry.get("adam_betas")
        lr_mul = table_entry.get("lr_mul", 1.0)
        wd_mul = table_entry.get("wd_mul", 1.0)

        if optim == "adam":
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"NorMuon param {label} must have .reshape attribute")

            chunk_shape = reshape
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            per_matrix_lr_mul = None
            if label == "mlp":
                per_matrix_lr_mul = []
                for i in range(reshape[0]):
                    is_c_proj = (i % 2 == 1)
                    per_matrix_lr_mul.append(2.0 if is_c_proj else 1.0)

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"],
                initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=reshape[0],
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                exp_avg = torch.zeros_like(param, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "normuon":
                chunk_shape = p_cfg.reshape

                momentum_buffer = torch.zeros(chunk_shape, dtype=torch.float32, device=param.device)

                if chunk_shape[-2] >= chunk_shape[-1]:
                    second_mom_shape = (*chunk_shape[:-1], 1)
                else:
                    second_mom_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                second_momentum_buffer = torch.zeros(second_mom_shape, dtype=torch.float32, device=param.device)

                mantissa = torch.zeros(chunk_shape, dtype=torch.uint16, device=param.device)

                self.param_states[param] = dict(
                    momentum_buffer=momentum_buffer,
                    second_momentum_buffer=second_momentum_buffer,
                    mantissa=mantissa,
                )

    def reset(self):
        self.split_embed = False
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "normuon":
                p_state = self.param_states[param]
                p_state["momentum_buffer"].zero_()
                p_state["mantissa"].zero_()
                p_state["second_momentum_buffer"].zero_()

    def copy_lm_state_to_embed(self):
        lm_head = self._lm_head_param
        embed = self._embed_param
        lm_state = self.param_states[lm_head]
        embed_state = self.param_states[embed]

        embed_state['step'] = lm_state['step']

        for key in ["exp_avg", "exp_avg_sq"]:
            embed_state[key].copy_(lm_state[key].T)

        self.split_embed = True

    def state_dict(self):
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        id_to_param = {id(p): p for p in self.param_cfgs.keys()}

        for param_id, saved_p_state in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                p_state = self.param_states[param]
                for k, v in saved_p_state.items():
                    if isinstance(v, torch.Tensor) and k in p_state:
                        target_dtype = p_state[k].dtype
                        p_state[k] = v.to(dtype=target_dtype, device=p_state[k].device)
                    else:
                        p_state[k] = v

    @torch.no_grad()
    def step(self, do_adam: bool = True):
        lm_param, embed_param = self._lm_head_param, self._embed_param

        for label in self.work_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]

            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None:
                continue

            if label == "lm_head" and do_adam and not self.split_embed:
                if embed_param is not None and embed_param.grad is not None:
                    param.grad.add_(embed_param.grad.T)

            if label == "embed" and not self.split_embed:
                continue

            grad = param.grad
            if p_cfg.optim == "normuon":
                grad = grad.view(p_cfg.reshape)

            if p_cfg.optim == "adam":
                self._adam_update(param, grad, p_cfg)
            else:
                self._normuon_update(param, grad, p_cfg)

        if do_adam and not self.split_embed and embed_param is not None and lm_param is not None:
            embed_param.data.copy_(lm_param.data.T)

        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue
            param.grad = None

    def _adam_update(self, param: nn.Parameter, grad: Tensor, p_cfg: ParamConfig):
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        p_state = self.param_states[param]
        p_state["step"] += 1
        t = p_state["step"]

        bias1, bias2 = 1 - beta1 ** t, 1 - beta2 ** t
        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        NorMuonAndAdam._adam_update_step(
            param, grad, p_state["exp_avg"], p_state["exp_avg_sq"],
            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    def _normuon_update(self, param: nn.Parameter, grad: Tensor, p_cfg: ParamConfig):
        chunk_shape = grad.shape

        p_state = self.param_states[param]
        grad = grad.float()

        momentum_buffer = p_state["momentum_buffer"]
        momentum_buffer.lerp_(grad, 1 - p_cfg.momentum)
        updated_grads = grad.lerp_(momentum_buffer, p_cfg.momentum)

        self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.lr)
        self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)

        is_large_matrix = chunk_shape[-2] > 1024
        v_chunk = polar_express(updated_grads, split_baddbmm=is_large_matrix)

        red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2
        v_chunk = NorMuonAndAdam._apply_normuon_variance_reduction(
            v_chunk, p_state["second_momentum_buffer"], p_cfg.beta2, red_dim
        )

        param_view = param.data.view(p_cfg.reshape)

        if p_cfg.per_matrix_lr_mul is not None:
            for mat_idx in range(p_cfg.chunk_size):
                self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.per_matrix_lr_mul[mat_idx] * p_cfg.lr)
                self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)
                NorMuonAndAdam._cautious_wd_and_update_inplace(
                    param_view[mat_idx].view(torch.uint16), p_state["mantissa"][mat_idx], v_chunk[mat_idx],
                    self._eff_wd_t, self._eff_lr_t
                )
        else:
            NorMuonAndAdam._cautious_wd_and_update_inplace(
                param_view.view(torch.uint16), p_state["mantissa"], v_chunk,
                self._eff_wd_t, self._eff_lr_t
            )

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        p_precise = p_precise_raw.view(torch.float32)
        mask = (grad * p_precise) >= 0
        p_precise.copy_(p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((p_precise_raw >> 16).to(torch.uint16))
        mantissa.copy_(p_precise_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
        v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = v_chunk.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        return v_chunk.mul_(final_scale.type_as(v_chunk))

# -----------------------------------------------------------------------------
# Model components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        theta = torch.outer(t, angular_freq)
        self.factor1 = nn.Buffer(theta.cos().to(torch.bfloat16), persistent=False)
        self.factor2 = nn.Buffer(theta.sin().to(torch.bfloat16), persistent=False)
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        theta = torch.outer(t, self.angular_freq)
        self.factor1.copy_(theta.cos())
        self.factor2.copy_(theta.sin())
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


class YarnPairedHead(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, angular_freq)
        theta2 = torch.outer(t_odd, angular_freq)
        self.factor1 = nn.Buffer(
            torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
            persistent=False
        )
        self.factor2 = nn.Buffer(
            torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
            persistent=False
        )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = args.block_size * old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        t_even = 2 * t
        t_odd = 2 * t + 1
        theta1 = torch.outer(t_even, self.angular_freq)
        theta2 = torch.outer(t_odd, self.angular_freq)
        self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
        self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


@dataclass
class AttnArgs:
    ve: torch.Tensor
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    ve_gate_w: torch.Tensor


class CausalSelfAttention(nn.Module):
    """
    MPS-compatible attention using PyTorch's scaled_dot_product_attention.
    """
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1

        yarn = attn_args.yarn
        ve, sa_lambdas, key_offset = attn_args.ve, attn_args.sa_lambdas, attn_args.key_offset
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = yarn.rotary(q), yarn.rotary(k)

        if key_offset:
            k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

        if ve is not None:
            ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T, self.num_heads, 1)
            v = v + ve_gate_out * ve.view_as(v)

        # Use PyTorch SDPA instead of flash_attn
        # Transpose for SDPA: (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=yarn.attn_scale)
        y = y.transpose(1, 2)  # Back to (B, T, num_heads, head_dim)

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))
        return y


class PairedHeadCausalSelfAttention(nn.Module):
    """
    Paired head attention - MPS compatible version.
    """
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        assert self.hdim == self.dim

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)
        assert B == 1

        yarn = attn_args.yarn
        ve, sa_lambdas = attn_args.ve, attn_args.sa_lambdas
        attn_gate_w, ve_gate_w = attn_args.attn_gate_w, attn_args.ve_gate_w

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)

        q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
        k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
        v = v.reshape(B, T*2, self.num_heads//2, self.head_dim)

        q, k = yarn.rotary(q), yarn.rotary(k)

        q = q.view(B, T*2, self.num_heads//2, self.head_dim)
        k = k.view(B, T*2, self.num_heads//2, self.head_dim)

        if ve is not None:
            ve_gate_out = 2 * torch.sigmoid(F.linear(x[..., :12], ve_gate_w)).view(B, T*2, self.num_heads//2, 1)
            v = v + ve_gate_out * ve.view_as(v)

        # Use PyTorch SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=yarn.attn_scale)
        y = y.transpose(1, 2)

        y = y.view(B, T, self.num_heads, self.head_dim)
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))
        return y


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, c_fc: Tensor, c_proj: Tensor):
        return FusedLinearReLUSquareFunction.apply(x, c_fc, c_proj)


class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, has_attn: bool, has_mlp: bool, use_paired_head: bool):
        super().__init__()
        if has_attn:
            if use_paired_head:
                self.attn = PairedHeadCausalSelfAttention(dim, head_dim, num_heads)
            else:
                self.attn = CausalSelfAttention(dim, head_dim, num_heads)
        else:
            self.attn = None
        self.mlp = MLP() if has_mlp else None

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor = None, c_fc: Tensor = None, c_proj: Tensor = None):
        if self.attn is not None:
            x = x + self.attn(norm(x), attn_args, qkvo_w)
        if self.mlp is not None:
            x = x + self.mlp(norm(x), c_fc, c_proj)
        return x

# -----------------------------------------------------------------------------
# Main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


@dataclass
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        vocab_size = next_multiple_of_n(vocab_size, n=128)

        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)
        self.smear_gate.weight.label = 'smear_gate'

        self.skip_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.skip_gate.weight)
        self.skip_gate.weight.label = 'skip_gate'

        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        for embed in self.value_embeds:
            nn.init.zeros_(embed.weight)
        for i, ve in enumerate(self.value_embeds):
            ve.weight.label = f've{i}'

        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12))
        self.attn_gate_bank.label = 'attn_gate_bank'
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12))
        self.ve_gate_bank.label = 've_gate_bank'

        self.attn_layer_indices = [i for i in range(num_layers) if i != 6]
        self.mlp_layer_indices = list(range(num_layers))

        hdim = num_heads * head_dim
        mlp_hdim = 4 * model_dim

        self.layer_to_attn_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.attn_layer_indices)}
        self.layer_to_mlp_idx = {layer_idx: bank_idx for bank_idx, layer_idx in enumerate(self.mlp_layer_indices)}

        self.attn_bank = nn.Parameter(torch.empty(len(self.attn_layer_indices), 4 * model_dim, hdim))
        self.attn_bank.label = 'attn'
        self.attn_bank.reshape = (len(self.attn_layer_indices) * 4, hdim, hdim)

        num_mlp_with_padding = len(self.mlp_layer_indices) + 1
        self.mlp_bank = nn.Parameter(torch.empty(num_mlp_with_padding, 2, mlp_hdim, model_dim))
        self.mlp_bank.label = 'mlp'
        self.mlp_bank.reshape = (num_mlp_with_padding * 2, mlp_hdim, model_dim)

        attn_std = model_dim ** -0.5
        attn_bound = (3 ** 0.5) * attn_std
        mlp_std = 0.5 * (model_dim ** -0.5)
        mlp_bound = (3 ** 0.5) * mlp_std
        with torch.no_grad():
            self.attn_bank[:, :model_dim * 3, :].uniform_(-attn_bound, attn_bound)
            self.attn_bank[:, model_dim * 3:, :].zero_()
            self.mlp_bank[:, 0, :, :].uniform_(-mlp_bound, mlp_bound)
            self.mlp_bank[:, 1, :, :].zero_()

        self.paired_head_layers = [0, 2, 5, 9]
        self.blocks = nn.ModuleList([
            Block(model_dim, head_dim, num_heads,
                  has_attn=(i in self.layer_to_attn_idx),
                  has_mlp=(i in self.layer_to_mlp_idx),
                  use_paired_head=(i in self.paired_head_layers))
            for i in range(num_layers)
        ])
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = YarnPairedHead(head_dim, max_seq_len)

        # No FP8 on MPS
        self.lm_head = CastedLinearT(model_dim, vocab_size, use_fp8=False)

        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)
        self.lm_head.weight.label = 'lm_head'

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.embed.weight.label = 'embed'
        with torch.no_grad():
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, model_dim)
        self.bigram_embed.weight.label = 'bigram_embed'
        nn.init.zeros_(self.bigram_embed.weight)

        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.x0_lambdas.label = 'x0_lambdas'

        # No distributed, so no padding needed
        self.scalars = nn.Parameter(
            torch.cat([
                1.1 * torch.ones(num_layers),
                *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],
                0.1 * torch.ones(num_layers),
                torch.zeros(1),
                0.5*torch.ones(1),
                -1.5 * torch.ones(1),
            ])
        )
        self.scalars.label = 'scalars'

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        mtp_weights, ws_short, ws_long = schedule_cfg.mtp_weights, schedule_cfg.ws_short, schedule_cfg.ws_long

        skip_connections = []
        skip_in = [3]
        skip_out = [6]
        x_backout = None
        backout_layer = 7

        resid_lambdas = self.scalars[: 1 * self.num_layers]
        x0_lambdas = self.x0_lambdas
        sa_lambdas = self.scalars[1 * self.num_layers: 3 * self.num_layers].view(-1, 2)
        bigram_lambdas = self.scalars[3 * self.num_layers: 4 * self.num_layers]
        smear_lambda = self.scalars[4 * self.num_layers]
        backout_lambda = self.scalars[4 * self.num_layers+1]
        skip_lambda = self.scalars[4 * self.num_layers+2]

        short_bm = ws_short * args.block_size
        long_bm = ws_long * args.block_size
        bm_sizes = [short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, None, short_bm, short_bm, short_bm, long_bm]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==long_bm for b in bm_sizes]

        x = self.embed(input_seq)
        x0_bigram = self.bigram_embed(bigram_input_seq)[None]

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[1], ve[2]] + [None] * (self.num_layers - 5) + [ve[0], ve[1], ve[2]]
        assert len(ve) == self.num_layers

        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        ag = [w.bfloat16() for w in self.attn_gate_bank.unbind(0)]
        veg = [w.bfloat16() for w in self.ve_gate_bank.unbind(0)]
        attn_gates = ag[:6] + [None] + ag[6:]
        ve_gates = [veg[0], veg[1]] + [None] * (self.num_layers - 5) + [veg[2], veg[3], veg[4]]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers

        attn_weights = self.attn_bank.unbind(0)
        mlp_fcs = self.mlp_bank[:, 0, :, :].unbind(0)
        mlp_projs = self.mlp_bank[:, 1, :, :].unbind(0)

        for i in range(self.num_layers):
            yarn = self.yarn_paired_head if i in self.paired_head_layers else self.yarn
            attn_args = AttnArgs(
                ve=ve[i],
                sa_lambdas=sa_lambdas[i],
                seqlens=seqlens,
                bm_size=bm_sizes[i],
                yarn=yarn,
                key_offset=key_offset[i],
                attn_gate_w=attn_gates[i],
                ve_gate_w=ve_gates[i]
            )
            if i in skip_out:
                skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))
                x = x + skip_gate_out * skip_connections.pop()
            if i == 0:
                x = (resid_lambdas[0] + x0_lambdas[0]) * x + bigram_lambdas[0] * x0_bigram
            else:
                x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + bigram_lambdas[i] * x0_bigram

            qkvo_w = attn_weights[self.layer_to_attn_idx[i]] if i in self.layer_to_attn_idx else None
            c_fc = mlp_fcs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None
            c_proj = mlp_projs[self.layer_to_mlp_idx[i]] if i in self.layer_to_mlp_idx else None

            x = self.blocks[i](x, attn_args, qkvo_w, c_fc, c_proj)
            if i in skip_in:
                skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        x -= backout_lambda * x_backout
        x = norm(x)
        logits = self.lm_head(x)

        if self.training:
            losses = FusedSoftcappedCrossEntropy.apply(logits.view(-1, logits.size(-1)), target_seq, mtp_weights)
            loss = losses.sum()
        else:
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            logits_for_loss = logits.float()
            loss = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="mean")
        return loss

# -----------------------------------------------------------------------------
# Data loading

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=False)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens
    return tokens

BOS_ID = 50256

class BOSFinder:
    def __init__(self, tokens: Tensor, world_size: int = 1, quickload: bool = False):
        self.tokens = tokens
        self.size = tokens.numel()
        self.quickload = quickload
        if quickload:
            self.bos_idx = (tokens[:4_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
            self.thread = None
            self.ready = threading.Event()
            self.start()
        else:
            self.bos_idx = (tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.i = 0
        self.world_size = world_size
        self.batch_iter = 0

    def _load(self):
        self.bos_idx_async = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        self.bos_idx = self.bos_idx_async

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        if self.quickload and self.batch_iter == 5:
            self.get()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                end = min(self.bos_idx[idx + 1] if idx + 1 < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur
                idx += 1

            assert cur_len == num_tokens_local + 1
        self.i = idx
        self.batch_iter += 1
        return starts, ends


class DataPreloader:
    def __init__(self, file_iter, world_size: int = 1):
        self.file_iter = file_iter
        self.world_size = world_size
        self.thread = None
        self.data = None
        self.ready = threading.Event()

    def _load(self):
        tokens = _load_data_shard(next(self.file_iter))
        self.data = (tokens, BOSFinder(tokens, self.world_size))
        self.ready.set()

    def start(self):
        self.ready.clear()
        self.thread = threading.Thread(target=self._load)
        self.thread.start()

    def get(self):
        if self.thread:
            self.ready.wait()
            self.thread.join()
        return self.data


def get_bigram_hash(x):
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size - 1
    x = x.to(torch.int32).clone()
    x[0] = mod
    x[1:] = torch.bitwise_xor(rand_int_1 * x[1:], rand_int_2 * x[:-1]) % mod
    return x


def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    world_size = 1
    rank = 0
    assert num_tokens % (world_size * grad_accum_steps) == 0
    num_tokens = num_tokens // grad_accum_steps

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

    file_iter = iter(files)
    tokens = _load_data_shard(next(file_iter))
    if align_to_bos:
        finder = BOSFinder(tokens, world_size=world_size, quickload=True)
        preloader = DataPreloader(file_iter, world_size)
        preloader.start()
    else:
        pos = 0

    while True:
        num_tokens_local = num_tokens // world_size
        max_num_docs = next_multiple_of_n(num_tokens_local // 300, n=128)

        if align_to_bos:
            try:
                seq_starts, seq_ends = finder.next_batch(num_tokens_local, max_seq_len)
                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])
            except StopIteration:
                tokens, finder = preloader.get()
                preloader.start()
                continue

            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if pos + num_tokens + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0

            pos_local = pos + rank * num_tokens_local
            buf = tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local,)
            _targets = buf[1:].view(num_tokens_local,)

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            pos += num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        new_params = yield (
            _inputs.to(device=device, non_blocking=True),
            _targets.to(device=device, non_blocking=True),
            _cum_lengths.to(device=device, non_blocking=True),
            _bigram_inputs.to(device=device, non_blocking=True)
        )

        if new_params is not None:
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (world_size * new_grad_accum_steps) == 0
            num_tokens = new_num_tokens // new_grad_accum_steps
            max_seq_len = new_max_seq_len

# -----------------------------------------------------------------------------
# Training Management

def get_bs(step: int):
    if step >= args.num_scheduled_iterations:
        return args.train_bs_extension
    x = step / args.num_scheduled_iterations
    bs_idx = int(len(args.train_bs_schedule) * x)
    return args.train_bs_schedule[bs_idx]

def get_ws(step: int):
    if step >= args.num_scheduled_iterations:
        return args.ws_final // 2, args.ws_final
    x = step / args.num_scheduled_iterations
    assert 0 <= x < 1
    ws_idx = int(len(args.ws_schedule) * x)
    return args.ws_schedule[ws_idx] // 2, args.ws_schedule[ws_idx]

def get_lr(step: int):
    if step > args.num_scheduled_iterations:
        return 0.1
    lr_max = 1.0
    x = step / args.num_scheduled_iterations
    if x > 1/3:
        lr_max = 1.52
    if x > 2/3:
        lr_max = 1.73
    if x >= 1 - args.cooldown_frac:
        w = (1 - x) / args.cooldown_frac
        lr = lr_max * w + (1 - w) * 0.1
        return lr
    return lr_max

def get_muon_momentum(step: int, muon_warmup_steps=300, muon_cooldown_steps=50, momentum_min=0.85, momentum_max=0.95):
    momentum_cd_start = args.num_iterations - muon_cooldown_steps
    if step < muon_warmup_steps:
        frac = step / muon_warmup_steps
        momentum = momentum_min + frac * (momentum_max - momentum_min)
    elif step > momentum_cd_start:
        frac = (step - momentum_cd_start) / muon_cooldown_steps
        momentum = momentum_max - frac * (momentum_max - momentum_min)
    else:
        momentum = momentum_max
    return momentum


class TrainingManager:
    """Training manager for M1 Mac (single GPU)."""
    def __init__(self, model):
        self.mtp_weights_schedule = self._build_mtp_schedule()
        self.model = model

        self.param_table = {
            "attn":           {"optim": "normuon", "adam_betas": None},
            "mlp":            {"optim": "normuon", "adam_betas": None},
            "scalars":        {"optim": "adam",    "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "ve0":            {"optim": "adam",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "ve1":            {"optim": "adam",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "ve2":            {"optim": "adam",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "bigram_embed":   {"optim": "adam",    "adam_betas": [0.75, 0.95], "lr_mul": 75.,  "wd_mul": 5.0},
            "smear_gate":     {"optim": "adam",    "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "skip_gate":      {"optim": "adam",    "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
            "attn_gate_bank": {"optim": "adam",    "adam_betas": [0.9,  0.99]},
            "ve_gate_bank":   {"optim": "adam",    "adam_betas": [0.9,  0.99]},
            "x0_lambdas":     {"optim": "adam",    "adam_betas": [0.65, 0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "lm_head":        {"optim": "adam",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
            "embed":          {"optim": "adam",    "adam_betas": [0.5,  0.95], "wd_mul": 150.},
        }

        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "x0_lambdas",
            "ve0", "ve1", "ve2", "bigram_embed",
            "lm_head", "embed",
            "attn", "mlp",
        ]

        adam_defaults = dict(lr=0.008, eps=1e-10, weight_decay=0.005)
        normuon_defaults = dict(lr=0.023, momentum=0.95, beta2=0.95, weight_decay=1.2)

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
        )

        self.split_step = math.ceil(args.split_embed_frac * args.num_scheduled_iterations) | 1
        self.reset()

    def _build_mtp_schedule(self):
        mtp_weights_schedule = []
        for s in range(args.num_iterations + 1):
            x = s / args.num_scheduled_iterations
            if x < 1/3:
                w = [1.0, 0.5, 0.25 * (1 - 3*x)]
            elif x < 2/3:
                w = [1.0, 0.5 * (1 - (3*x - 1))]
            else:
                w = [1.0]
            mtp_weights_schedule.append(torch.tensor(w, device=device))
        return mtp_weights_schedule

    def apply_final_ws_ext(self):
        self.ws_long = args.ws_validate_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights=self.mtp_weights,
            ws_short=self.ws_short,
            ws_long=self.ws_long
        )

    def _is_adam_step(self, step: int):
        return step % 2 == 1

    def get_transition_steps(self):
        transition_steps = []
        ws_short, ws_long = get_ws(0)
        for step in range(1, args.num_iterations):
            ws_short, new_ws_long = get_ws(step)
            if new_ws_long != ws_long:
                transition_steps.append(step)
                ws_long = new_ws_long
        return transition_steps

    def advance_schedule(self, step: int):
        self.ws_short, new_ws_long = get_ws(step)
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long, new_ws_long)
            self.model.yarn_paired_head.apply(self.ws_long, new_ws_long)

        new_batch_size = get_bs(step)
        if new_batch_size != self.batch_size:
            self.train_loader_send_args = (new_batch_size, args.train_max_seq_len, grad_accum_steps)
            self.batch_size = new_batch_size
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = self.mtp_weights_schedule[step]

    def step_optimizers(self, step: int):
        step_lr = get_lr(step)
        muon_momentum = get_muon_momentum(step)
        do_adam = self._is_adam_step(step)

        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_momentum

        self.optimizer.step(do_adam=do_adam)

        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)

        self.optimizer.reset()

        self.ws_short, self.ws_long = get_ws(0)
        self.batch_size = get_bs(0)
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()

    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

# -----------------------------------------------------------------------------
# Hyperparameters (adjusted for M1 Mac)

@dataclass
class Hyperparameters:
    # data
    train_files: str = "../data/fineweb10B/fineweb_train_*.bin"
    val_files: str = "../data/fineweb10B/fineweb_val_*.bin"
    val_tokens: int = 10485760
    # batch sizes - REDUCED for M1 Mac memory constraints
    train_bs_schedule: tuple = (2 * 2048 * 8, 4 * 2048 * 8, 6 * 2048 * 8)  # Reduced from 8/16/24
    train_bs_extension: int = 6 * 2048 * 8
    train_max_seq_len: int = 128 * 16
    val_batch_size: int = 1 * 64 * 1024 * 8  # Reduced from 4x
    # optimization
    num_scheduled_iterations: int = 1560
    num_extension_iterations: int = 40
    num_iterations: int = num_scheduled_iterations + num_extension_iterations
    cooldown_frac: float = 0.55
    split_embed_frac: float = 2/3
    # evaluation and logging
    run_id: str = f"{uuid.uuid4()}"
    val_loss_every: int = 250
    save_checkpoint: bool = False
    # attention masking
    block_size: int = 128
    ws_schedule: tuple = (3, 7, 11)
    ws_final: int = 13
    ws_validate_post_yarn_ext: int = 20
    # bigram hash embedding
    bigram_vocab_size = 50304 * 5

args = Hyperparameters()

data_path = os.environ.get("DATA_PATH", ".")
args.train_files = os.path.join(data_path, args.train_files)
args.val_files = os.path.join(data_path, args.val_files)

# -----------------------------------------------------------------------------
# Main

if __name__ == "__main__":
    master_process = True

    # Logging
    logfile = None
    if master_process:
        run_id = args.run_id
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(f"Log file: {logfile}")

    def print0(s, console=False):
        if master_process:
            if logfile:
                with open(logfile, "a") as f:
                    if console:
                        print(s)
                    print(s, file=f)
            elif console:
                print(s)

    print0(code)
    print0("=" * 100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__}")
    print0(f"Using device: {device}")
    if device.type == "mps":
        print0("MPS (Metal Performance Shaders) backend active")
    print0("=" * 100)

    # Create model
    model: nn.Module = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=args.val_batch_size // grad_accum_steps
    ).to(device)

    # Convert to bfloat16 where supported
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.attn_bank.data = model.attn_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()

    # Note: torch.compile may have limited support on MPS
    try:
        model = torch.compile(model, dynamic=False, fullgraph=True)
        print0("Model compiled successfully", console=True)
    except Exception as e:
        print0(f"torch.compile not fully supported on MPS, using eager mode: {e}", console=True)

    training_manager = TrainingManager(model)

    # Warmup
    print0("Warming up model...", console=True)
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizer=training_manager.get_state()
    )

    train_loader = distributed_data_generator(
        args.train_files, args.train_bs_schedule[0], args.train_max_seq_len,
        grad_accum_steps=grad_accum_steps
    )
    val_loader = distributed_data_generator(
        args.val_files, args.val_batch_size, -1,
        grad_accum_steps=grad_accum_steps, align_to_bos=False
    )

    # Quick warmup
    for step in [0, 1, 2]:
        training_manager.advance_schedule(step)
        model.eval()
        with torch.no_grad():
            inputs, targets, cum_seqlens, bigram_inputs = next(val_loader)
            model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args())
        model.train()
        for idx in range(grad_accum_steps):
            send_args = training_manager.train_loader_send_args
            inputs, targets, cum_seqlens, bigram_inputs = train_loader.send(send_args)
            (model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) / grad_accum_steps).backward()
        training_manager.step_optimizers(step)

    print0("Resetting model...", console=True)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    del val_loader, train_loader, initial_state
    model.train()

    # Training
    train_loader = distributed_data_generator(
        args.train_files, args.train_bs_schedule[0], args.train_max_seq_len,
        grad_accum_steps=grad_accum_steps
    )

    gc.collect()

    training_time_ms = 0

    # MPS sync
    if device.type == "mps":
        torch.mps.synchronize()

    t0 = time.perf_counter()
    train_steps = args.num_iterations

    for step in range(train_steps + 1):
        last_step = (step == train_steps)
        training_manager.advance_schedule(step)

        # Validation
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step:
                training_manager.apply_final_ws_ext()

            if device.type == "mps":
                torch.mps.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)

            model.eval()
            assert args.val_tokens % args.val_batch_size == 0
            val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
            val_loader = distributed_data_generator(
                args.val_files, args.val_batch_size, -1,
                grad_accum_steps=grad_accum_steps, align_to_bos=False
            )
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets, cum_seqlens, bigram_inputs = next(val_loader)
                    val_loss += model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args())
            val_loss /= val_steps
            del val_loader

            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            model.train()

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            break

        # Training step
        for idx in range(grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs = train_loader.send(training_manager.train_loader_send_args)
            (model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()) / grad_accum_steps).backward()
        training_manager.step_optimizers(step)

        # Logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    # Memory stats
    if device.type == "mps":
        print0(f"peak memory allocated: {torch.mps.current_allocated_memory() // 1024 // 1024} MiB", console=True)
