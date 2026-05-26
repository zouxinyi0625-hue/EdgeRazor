#!/bin/bash

################################################################################
# Run W4 + W1.58 sequentially for MaiProfile Qwen3-1.7B
################################################################################
# Usage: bash run_all.sh <layer_name>
# Example: bash run_all.sh layer1_delta
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYER="${1:-layer0_signal}"

echo "Running W4A8KV8 then W1.58A8KV8 for layer: ${LAYER}"
echo ""

bash "${SCRIPT_DIR}/run.sh" w4 "${LAYER}" && bash "${SCRIPT_DIR}/run.sh" w1.58 "${LAYER}"
