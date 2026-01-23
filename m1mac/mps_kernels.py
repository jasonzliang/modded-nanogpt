"""
MPS-compatible kernel replacements for M1 Mac.
These replace the Triton CUDA kernels with PyTorch native operations.

Performance Note: These are significantly slower than the optimized Triton kernels
but provide functional equivalents that work on Apple Silicon MPS backend.
"""

import torch
import torch.nn.functional as F


def XXT(A: torch.Tensor, out: torch.Tensor):
    """
    Compute C = A @ A.T (symmetric matrix multiplication)
    PyTorch native replacement for Triton XXT kernel.
    """
    assert A.ndim == 2 or A.ndim == 3
    if A.ndim == 2:
        result = torch.mm(A, A.T)
    else:
        result = torch.bmm(A, A.transpose(-2, -1))
    out.copy_(result)
    return out


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Compute C = alpha * A @ A.T + beta * A
    PyTorch native replacement for Triton ba_plus_cAA kernel.
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"

    if A.ndim == 2:
        result = alpha * torch.mm(A, A.T) + beta * A
    else:
        result = alpha * torch.bmm(A, A.transpose(-2, -1)) + beta * A
    out.copy_(result)
    return out


def linear_relu_square(a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor = None):
    """
    Compute relu(a @ b.T)^2 with optional backward pass support.
    PyTorch native replacement for Triton linear_relu_square kernel.

    Forward: pre = a @ b.T, post = relu(pre)^2
    Backward: uses aux (which stores pre from forward) to compute gradients
    """
    # a: (M, K), b: (N, K) -> output: (M, N)
    pre = F.linear(a, b)  # a @ b.T

    if aux is None:
        # Forward pass
        post = F.relu(pre).square()
        return pre, post
    else:
        # Backward pass - aux contains the pre-activation from forward
        # d(relu(x)^2)/dx = 2 * relu(x) * (x > 0) = 2 * relu(x)
        # pre here is the upstream gradient, aux is the saved pre-activation
        grad = pre * 2 * F.relu(aux)
        return grad


class FusedLinearReLUSquareFunction(torch.autograd.Function):
    """
    Fused operation: relu(x @ W1.T)^2 @ W2
    MPS-compatible replacement using PyTorch native operations.

    Note: W1 projects up (D -> H), W2 projects down (H -> D).
    Both weights stored as (H, D), but W1 is used transposed via F.linear.
    """
    @staticmethod
    def forward(ctx, x, W1, W2):
        # x: (B, T, D), W1: (H, D), W2: (H, D) -> output: (B, T, D)
        # W1 used via F.linear (x @ W1.T), W2 used directly (post @ W2)
        x_flat = x.view(-1, x.shape[-1])

        # First linear + relu^2
        pre = F.linear(x_flat, W1)  # x @ W1.T
        post = F.relu(pre).square()

        # Second linear (direct matmul since W2 is stored as (H, D))
        out = post @ W2  # post @ W2

        ctx.save_for_backward(x, W1, W2, pre, post)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, W1, W2, pre, post = ctx.saved_tensors
        x_flat = x.view(-1, x.shape[-1])
        grad_flat = grad_output.view(-1, grad_output.shape[-1])

        # Gradient for W2: post.T @ grad_output
        dW2 = post.T @ grad_flat

        # Gradient through second linear: grad_output @ W2.T (since forward is post @ W2)
        d_post = F.linear(grad_flat, W2)

        # Gradient through relu^2: d(relu(x)^2)/dx = 2*relu(x)*(x>0) = 2*relu(x)
        d_pre = d_post * 2 * F.relu(pre)

        # Gradient for W1: d_pre.T @ x
        dW1 = d_pre.T @ x_flat

        # Gradient for x: d_pre @ W1
        dx = F.linear(d_pre, W1.T)

        return dx.view(x.shape), dW1, dW2


class FusedSoftcappedCrossEntropy(torch.autograd.Function):
    """
    Fused softcapped cross entropy loss.
    MPS-compatible replacement using PyTorch native operations.

    Softcap: z = A * sigmoid((logits + B) / C)
    Loss: cross_entropy(z, targets) weighted by mtp_weights
    """
    @staticmethod
    def forward(ctx, logits, targets, mtp_weights, A=23.0, B=5.0, C=7.5):
        n_rows, n_cols = logits.shape
        if mtp_weights is None:
            mtp_weights = torch.tensor([1.0], device=logits.device, dtype=torch.float32)
        n_predict = mtp_weights.shape[0]

        # Apply softcap: z = A * sigmoid((logits + B) / C)
        z = A * torch.sigmoid((logits.float() + B) / C)

        # Compute log-sum-exp for numerical stability
        lse = torch.logsumexp(z, dim=-1)

        # Compute weighted losses for multi-token prediction
        losses = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)
        row_indices = torch.arange(n_rows, device=logits.device)
        for k in range(n_predict):
            weight = mtp_weights[k]  # Keep as tensor, don't use .item()
            # Get target indices, handling boundary
            valid_mask = (row_indices + k < n_rows).float()
            target_indices = torch.clamp(row_indices + k, max=n_rows - 1)
            shifted_targets = targets[target_indices]

            # Gather the logits for targets
            z_target = z.gather(1, shifted_targets.unsqueeze(1)).squeeze(1)

            # Cross entropy: lse - z_target (weight=0 naturally contributes nothing)
            loss_k = (lse - z_target) * weight * valid_mask
            losses += loss_k

        ctx.save_for_backward(logits, targets, mtp_weights, lse)
        ctx.params = (A, B, C)
        return losses

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, mtp_weights, lse = ctx.saved_tensors
        A, B, C = ctx.params
        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]

        # Recompute softcap
        u = (logits.float() + B) / C
        sigmoid_u = torch.sigmoid(u)
        z = A * sigmoid_u

        # Softmax probabilities
        p = torch.exp(z - lse.unsqueeze(1))

        # Compute sum of weights for each position
        S_w = torch.zeros(n_rows, dtype=torch.float32, device=logits.device)
        for k in range(n_predict):
            valid_mask = torch.arange(n_rows, device=logits.device) + k < n_rows
            S_w += torch.where(valid_mask, mtp_weights[k], torch.zeros_like(mtp_weights[k]))

        # Gradient of softcapped cross entropy
        # d_loss/d_logits = S_w * p - sum_k(weight_k * indicator(target_k))
        grad_z = grad_output.unsqueeze(1) * S_w.unsqueeze(1) * p

        for k in range(n_predict):
            weight = mtp_weights[k]
            valid_mask = torch.arange(n_rows, device=logits.device) + k < n_rows
            target_indices = torch.clamp(
                torch.arange(n_rows, device=logits.device) + k,
                max=n_rows - 1
            )
            shifted_targets = targets[target_indices]

            # Subtract weight at target positions
            grad_z.scatter_add_(
                1,
                shifted_targets.unsqueeze(1),
                -grad_output.unsqueeze(1) * weight * valid_mask.float().unsqueeze(1)
            )

        # Chain rule through softcap: dz/dx = (1/C) * z * (1 - sigmoid_u)
        dz_dx = (1.0 / C) * z * (1.0 - sigmoid_u)
        grad_input = (grad_z * dz_dx).to(logits.dtype)

        return grad_input, None, None, None, None, None
