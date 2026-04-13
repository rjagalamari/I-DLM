#!/bin/bash
# Install I-DLM inference support (SGLang with ISD algorithm).
#
# Usage:
#   bash install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing SGLang with I-DLM support (from bundled source)..."
pip install -e "${SCRIPT_DIR}/sglang"

echo ""
echo "Done! Verify with:"
echo "  python -c \"from sglang.srt.dllm.algorithm.idlm_blockN import IDLMBlockN; print('OK')\""
