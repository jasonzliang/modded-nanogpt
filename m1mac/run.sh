#!/bin/bash
# M1 Mac training script
# Single GPU execution (no distributed training)

# Optional: Set data path if different from default
# export DATA_PATH="/path/to/data"

# Set environment variables for MPS
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Get the directory of the script
DIR="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

# Change the current working directory to the script's directory
cd "$DIR"

# Run training directly (no torchrun needed for single GPU)
python train_gpt.py
