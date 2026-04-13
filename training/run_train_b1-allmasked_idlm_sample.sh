#!/bin/bash
set -eo pipefail
echo "Job started at $(date)"

# --- Credentials (set via environment or fill in) ---
export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
export WANDB_ENTITY="${WANDB_ENTITY}"

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="${SCRIPT_DIR}/llama_factory_sdar"
MODEL_DIR="${SCRIPT_DIR}/model/Qwen3-8B-b1-allmasked"
export PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH:-}"
TRAINING_CONFIG="./examples/train_idlm/qwen3_8b_b1-allmasked_sample.yaml"

# --- Environment ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate idlm
# Use system CUDA (nvcc required by DeepSpeed); fall back to conda if not found
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=$CONDA_PREFIX
fi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export DS_ACCELERATOR=cuda
export CC=gcc  # use system gcc for Triton JIT (conda's cc can't find libcuda)

# --- Cache directories (use local fs to avoid NFS stale handles) ---
CACHE_BASE="/dev/shm/${USER}/idlm_cache_$$"
mkdir -p "$CACHE_BASE"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
rm -rf "${LLAMAFACTORY_DIR}/torchinductor_${USER}" 2>/dev/null || true

# --- Logging dirs ---
export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$WANDB_DIR" "$HF_HOME"

# --- Launch training ---
MASTER_PORT=$(python3 -c 'from socket import socket; s=socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
cd "$LLAMAFACTORY_DIR"
torchrun --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=$MASTER_PORT \
    src/llamafactory/launcher.py "$TRAINING_CONFIG" \
    model_name_or_path="$MODEL_DIR"

echo "Training completed at $(date)"
