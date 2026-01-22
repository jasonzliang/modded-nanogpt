# M1 Mac (Apple Silicon) Version of modded-nanogpt

This directory contains a modified version of the training code that runs on Apple Silicon Macs using MPS (Metal Performance Shaders) for GPU acceleration.

## Key Differences from CUDA Version

| Feature | CUDA Version | M1 Mac Version |
|---------|-------------|----------------|
| GPU Backend | CUDA + Triton | MPS (Metal) |
| Distributed Training | 8 GPUs with NCCL | Single GPU only |
| FP8 Operations | Yes (torch._scaled_mm) | No (uses bfloat16) |
| Custom Kernels | Triton JIT compiled | PyTorch native ops |
| Flash Attention | flash_attn library | PyTorch SDPA |
| Batch Sizes | 8/16/24 × 2048 × 8 | 2/4/6 × 2048 × 8 (reduced) |

## Performance Expectations

**Important:** This M1 Mac version will be **significantly slower** than the CUDA version due to:

1. **No Triton kernels** - The highly optimized Triton kernels are replaced with PyTorch native operations
2. **No FP8** - Uses bfloat16/float16 instead of FP8 quantized operations
3. **Single GPU** - No multi-GPU parallelism (M1 has unified memory but single GPU)
4. **No flash attention** - Uses standard PyTorch scaled_dot_product_attention
5. **Reduced batch sizes** - Memory constraints require smaller batches

Expect training to take **10-50x longer** than on 8× H100 GPUs.

## Requirements

```bash
# Python 3.10+ recommended
pip install torch>=2.0  # MPS support
pip install numpy tqdm huggingface-hub
```

**Note:** The `kernels` package and `triton` are NOT needed for the M1 Mac version.

## Setup

1. Ensure you have the data files in the expected location:
```
data/fineweb10B/fineweb_train_*.bin
data/fineweb10B/fineweb_val_*.bin
```

Or set the DATA_PATH environment variable:
```bash
export DATA_PATH="/path/to/your/data"
```

2. Make the run script executable:
```bash
chmod +x run.sh
```

## Running

```bash
cd m1mac
./run.sh
```

Or directly:
```bash
python train_gpt.py
```

## Memory Considerations

The M1 Mac version uses reduced batch sizes to fit in typical M1/M2 memory:

- **M1 (8GB)**: May need to further reduce batch sizes
- **M1 Pro/Max (16-64GB)**: Default settings should work
- **M1 Ultra (64-128GB)**: Can increase batch sizes closer to original

To adjust batch sizes, modify `args.train_bs_schedule` in `train_gpt.py`:

```python
# Current (reduced for M1):
train_bs_schedule: tuple = (2 * 2048 * 8, 4 * 2048 * 8, 6 * 2048 * 8)

# For more memory:
train_bs_schedule: tuple = (4 * 2048 * 8, 8 * 2048 * 8, 12 * 2048 * 8)

# For less memory:
train_bs_schedule: tuple = (1 * 2048 * 8, 2 * 2048 * 8, 3 * 2048 * 8)
```

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | Main training script (MPS-compatible) |
| `mps_kernels.py` | PyTorch native replacements for Triton kernels |
| `run.sh` | Simple bash script to run training |
| `README.md` | This file |

## Troubleshooting

### "MPS not available"

Ensure you have:
- macOS 12.3+ (Monterey or later)
- PyTorch 2.0+ with MPS support
- Apple Silicon Mac (M1/M2/M3)

Check MPS availability:
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
print(torch.backends.mps.is_built())      # Should be True
```

### Out of Memory

Reduce batch sizes in `train_gpt.py`:
```python
train_bs_schedule: tuple = (1 * 2048 * 8, 2 * 2048 * 8, 3 * 2048 * 8)
val_batch_size: int = 1 * 64 * 1024 * 4  # Reduced
```

### torch.compile errors

MPS has limited torch.compile support. The code falls back to eager mode if compilation fails. This is normal and the model will still train (slightly slower).

### Slow performance

This is expected. MPS is significantly slower than CUDA for this workload. For production training, use NVIDIA GPUs with the original CUDA version.

## Limitations

1. **No distributed training** - Single M1 GPU only
2. **No FP8** - Apple Silicon doesn't support FP8 operations
3. **No Triton** - Triton only supports NVIDIA GPUs
4. **No flash_attn** - Uses PyTorch's built-in SDPA instead
5. **Slower training** - Expect 10-50x slower than multi-GPU CUDA setup

## For Development/Testing Only

This M1 Mac version is primarily useful for:
- Code development and debugging
- Small-scale experiments
- Understanding the model architecture
- Testing changes before deploying to CUDA cluster

For actual model training at scale, use the original CUDA version with NVIDIA GPUs.
