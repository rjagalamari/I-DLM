#!/bin/bash
# Launch one of the matched b7 LoRA ablation arms (A=vanilla, B=rollout, C=relay).
# Usage: bash run_train_b7_lora.sh {vanilla|rollout|relay}
# Replace MODEL_DIR with the Unity path to the I-DLM-8B weights before launching.
set -eo pipefail

ARM="${1:-relay}"
case "$ARM" in
  vanilla) CONFIG="./examples/train_idlm/qwen3_8b_b7-vanilla-lora.yaml" ;;
  rollout) CONFIG="./examples/train_idlm/qwen3_8b_b7-rollout-lora.yaml" ;;
  relay)   CONFIG="./examples/train_idlm/qwen3_8b_b7-relay-lora.yaml" ;;
  *) echo "Usage: $0 {vanilla|rollout|relay}" >&2; exit 1 ;;
esac

echo "Job started at $(date) arm=${ARM}"

export WANDB_API_KEY="${WANDB_API_KEY}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
export WANDB_ENTITY="${WANDB_ENTITY}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMAFACTORY_DIR="${SCRIPT_DIR}/llama_factory_sdar"
# PLACEHOLDER: path to Qwen3-8B-b7-allmasked WITH weights.
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/model/Qwen3-8B-b7-allmasked}"
export PYTHONPATH="${LLAMAFACTORY_DIR}/src:${PYTHONPATH:-}"

# Prefer the relay env (torch 2.12 + flex_attention); fall back to idlm.
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx relay; then
    conda activate relay
elif conda env list | awk '{print $1}' | grep -qx idlm; then
    conda activate idlm
fi

if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
else
    export CUDA_HOME=$CONDA_PREFIX
fi
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export DS_ACCELERATOR=cuda
export CC=gcc

CACHE_BASE="/dev/shm/${USER}/idlm_cache_$$"
mkdir -p "$CACHE_BASE"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/torchinductor"
export CUDA_CACHE_PATH="${CACHE_BASE}/cuda"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"

export WANDB_DIR="${WANDB_DIR:-$HOME/wandb}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$WANDB_DIR" "$HF_HOME"

MASTER_PORT=$(python3 -c 'from socket import socket; s=socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
cd "$LLAMAFACTORY_DIR"
# Ram baseline was 1x A100. Bump --nproc_per_node if you have more GPUs;
# keep per_device_train_batch_size=1 (SDAR [noisy|clean] concat constraint).
NPROC="${NPROC:-1}"
torchrun --nnodes=1 --nproc_per_node="$NPROC" --master_addr=localhost --master_port=$MASTER_PORT \
    src/llamafactory/launcher.py "$CONFIG" \
    model_name_or_path="$MODEL_DIR"

echo "Training completed at $(date)"
