#!/bin/bash
# Set up a complete I-DLM inference environment from scratch.
#
# Usage:
#   bash setup_env.sh
#
# This creates a conda environment "idlm", installs all dependencies,
# and installs SGLang with I-DLM support from the bundled source.
#
# Prerequisites:
#   - conda (miniconda or anaconda)
#   - CUDA 12.9 driver installed
#   - GPU with compute capability >= 8.0 (A100/H100/etc.)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="idlm"

echo "=== Setting up I-DLM inference environment ==="
echo ""

# Step 1: Create conda environment
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Activating..."
else
    echo "Creating conda env '${ENV_NAME}' with Python 3.12..."
    conda create -n ${ENV_NAME} python=3.12 -y
fi

# Activate
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}
echo "Using Python: $(which python3) ($(python3 --version))"
echo ""

# Step 2: Install PyTorch 2.9.1 + CUDA 12.9
echo "Installing PyTorch 2.9.1 (CUDA 12.9)..."
pip install torch==2.9.1+cu129 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu129
echo ""

# Step 3: Install SGLang with I-DLM support
echo "Installing SGLang with I-DLM support..."
bash "${SCRIPT_DIR}/install.sh"
echo ""

# Step 4: Verify
echo "=== Verification ==="
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import sglang; print(f'SGLang: {sglang.__version__}')
from sglang.srt.dllm.algorithm.idlm_blockN import IDLMBlockN
print('IDLMBlockN: OK')
import flashinfer; print(f'FlashInfer: {flashinfer.__version__}')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Launch server (example with N=8 model):"
echo "  python -m sglang.launch_server \\"
echo "    --model-path yifanyu/I-DLM-8B \\"
echo "    --dllm-algorithm IDLMBlockN \\"
echo "    --dllm-algorithm-config ${SCRIPT_DIR}/configs/idlm_blockN4_config.yaml \\"
echo "    --trust-remote-code --tp-size 1 \\"
echo "    --mem-fraction-static 0.85 --max-running-requests 32 \\"
echo "    --attention-backend flashinfer --dtype bfloat16"
