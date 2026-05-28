#!/bin/bash
# One-click environment setup for EdgeRazor-LLM
# Usage: bash setup_env.sh

set -e

ENV_NAME="edgerazor"

echo "Creating conda environment: ${ENV_NAME}"
conda create -n ${ENV_NAME} python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "Installing base dependencies..."
pip install -r "$(dirname "$0")/requirements.txt"

echo "Installing flash-attn (requires compilation, may take a few minutes)..."
pip install flash-attn==2.8.3 --no-build-isolation

echo ""
echo "========================================"
echo "Setup complete!"
echo "Activate with: conda activate ${ENV_NAME}"
echo "========================================"
