#!/bin/bash

# ============================================================================
# Environment Configuration
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_RETRY_TIMES=10
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_LEVEL=NVL
export DEEPSPEED_TIMEOUT=5400
export NCCL_TIMEOUT=5400
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

################################################################################
# EdgeRazor-QLLM MaiProfile Training Pipeline
################################################################################
# Usage: bash run.sh [w4|w2.79|w1.58]
################################################################################

# ============================================================================
# User Configuration
# ============================================================================

PATH_PREFIX="/path/to/your/environment"
MODEL_NAME="Qwen/Qwen3-1.7B"

# Select quantization config based on argument
QUANT_ARG="${1:-w4}"
case "${QUANT_ARG}" in
    "w4")
        QUANT_CONFIG="W4A8KV8"
        TRAIN_YAML_DIR="w4a8kv8"
        RUN_NAME="maiprofile_qwen3-1.7b_w4a8kv8"
        ;;
    "w2.79")
        QUANT_CONFIG="W2.79A8KV8"
        TRAIN_YAML_DIR="w2.79a8kv8"
        RUN_NAME="maiprofile_qwen3-1.7b_w2.79a8kv8"
        ;;
    "w1.58")
        QUANT_CONFIG="W1.58A8KV8"
        TRAIN_YAML_DIR="w1.58a8kv8"
        RUN_NAME="maiprofile_qwen3-1.7b_w1.58a8kv8"
        ;;
    *)
        echo "Usage: bash run.sh [w4|w2.79|w1.58]"
        exit 1
        ;;
esac

TRAIN_YAML="train"
TRAIN_VERSION="train_maiprofile"

# ============================================================================
# Derived Paths
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_YAML_PATH="${SCRIPT_DIR}/${TRAIN_YAML_DIR}/${TRAIN_YAML}.yaml"

CODE_ROOT="${PATH_PREFIX}/code/EdgeRazor-QLLM"
TEMPLATE_NAME="${MODEL_NAME}-${QUANT_CONFIG}-Template"
TEMPLATE_PATH="${CODE_ROOT}/template/${MODEL_NAME}/${TEMPLATE_NAME}"

TRAIN_ROOT="${CODE_ROOT}/${TRAIN_VERSION}"
FINAL_MODEL="${TRAIN_ROOT}/final_model"
EVAL_MODEL="${TRAIN_ROOT}/${MODEL_NAME}"

TRAIN_LOG_DIR="${TRAIN_ROOT}/logs"
TENSORBOARD_DIR="${TRAIN_ROOT}/tensorboard"
RESULT_DIR="${TRAIN_ROOT}/results"

QUANT_CONFIG_PATH="${CODE_ROOT}/src/train.yaml"
CONVERT_SCRIPT="${CODE_ROOT}/src/convert/convert_qweight.py"

export PATH="${HOME}/.conda/envs/edgerazor/bin:${PATH}"

# ============================================================================
# Pre-flight
# ============================================================================

cp "${TRAIN_YAML_PATH}" "${QUANT_CONFIG_PATH}"

# Use MaiProfile config class
sed -i.bak "s/= \"exp_for_sed\"/= \"${RUN_NAME}\"/" "${CODE_ROOT}/src/config_maiprofile.py"
sed -i.bak "s/config = EdgeRazorTrainConfigFor[^(]*()/config = EdgeRazorTrainConfigForMaiProfile()/" "${CODE_ROOT}/src/main.py"

# Ensure main.py imports config_maiprofile
if ! grep -q "from config_maiprofile import" "${CODE_ROOT}/src/main.py"; then
    sed -i.bak "1i from config_maiprofile import EdgeRazorTrainConfigForMaiProfile" "${CODE_ROOT}/src/main.py"
fi

# ============================================================================
# Pipeline
# ============================================================================

echo "================================================================================"
echo "EdgeRazor-QLLM MaiProfile Training Pipeline"
echo "================================================================================"
echo "Model:           ${MODEL_NAME}"
echo "Quantization:    ${QUANT_CONFIG}"
echo "Experiment:      ${RUN_NAME}"
echo "Output:          ${TRAIN_ROOT}"
echo "================================================================================"
echo ""

# ---- Step 1: Prepare ----
echo "[Step 1/3] Preparing environment..."
mkdir -p "${TRAIN_LOG_DIR}" "${TENSORBOARD_DIR}" "${RESULT_DIR}"
if [ -d "${TEMPLATE_PATH}" ]; then
    cp -rf "${TEMPLATE_PATH}" "${EVAL_MODEL}"
    echo "  ✓ Template copied"
fi
echo ""

# ---- Step 2: Train ----
echo "[Step 2/3] Starting distributed training..."
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
TRAINING_LOG="${TRAIN_LOG_DIR}/train_${timestamp}.log"
time {
    deepspeed --num_gpus=8 "${CODE_ROOT}/src/main.py"
} 2>&1 | tee "${TRAINING_LOG}"
echo "  ✓ Training completed"
echo ""

# ---- Step 3: Convert weights ----
echo "[Step 3/3] Converting weights..."
python "${CONVERT_SCRIPT}" \
    --quant_config "${QUANT_CONFIG_PATH}" \
    --unquantized_model "${FINAL_MODEL}" \
    --quantized_model "${EVAL_MODEL}" \
    --dtype bfloat16
echo "  ✓ Weight conversion completed"
echo ""

echo "================================================================================"
echo "MaiProfile Training Pipeline Completed!"
echo "  Final model:   ${FINAL_MODEL}"
echo "  Eval model:    ${EVAL_MODEL}"
echo "================================================================================"
