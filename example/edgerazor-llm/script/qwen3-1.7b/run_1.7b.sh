#!/bin/bash

# ============================================================================
# Environment Configuration (shared across all experiments)
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
# EdgeRazor Training Pipeline: Qwen3-1.7B W4 + W1.58 Sequential Run
################################################################################

# ============================================================================
# Shared Configuration
# ============================================================================
PATH_PREFIX="/scratch/azureml/cr/j/ca4c1a241a09470a950b55e21b068fbf/exe/wd"
MODEL_NAME="Qwen/Qwen3-1.7B"
CODE_ROOT="${PATH_PREFIX}/EdgeRazor"
CONVERT_SCRIPT="${CODE_ROOT}/src/convert/convert_qweight.py"
CONFIG_CLASS="EdgeRazorTrainConfigForQwen3_1_7B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="${HOME}/.conda/envs/edgerazor/bin:${PATH}"

# Inject config class into main.py
sed -i.bak "s/config = EdgeRazorTrainConfigFor[^(]*()/config = ${CONFIG_CLASS}()/" "${CODE_ROOT}/src/main.py"

run_experiment() {
    local QUANT_CONFIG="$1"
    local RUN_NAME="$2"
    local TRAIN_VERSION="$3"
    local TRAIN_YAML_DIR="$4"

    echo "================================================================================"
    echo "EdgeRazor Training & Evaluation Pipeline"
    echo "================================================================================"
    echo "Model:           ${MODEL_NAME}"
    echo "Quantization:    ${QUANT_CONFIG}"
    echo "Experiment:      ${RUN_NAME}"
    echo "Output:          ${CODE_ROOT}/${TRAIN_VERSION}"
    echo "================================================================================"
    echo ""

    local TEMPLATE_NAME="${MODEL_NAME}-${QUANT_CONFIG}-Template"
    local TEMPLATE_PATH="${CODE_ROOT}/template/${MODEL_NAME}/${TEMPLATE_NAME}"
    local TRAIN_ROOT="${CODE_ROOT}/${TRAIN_VERSION}"
    local FINAL_MODEL="${TRAIN_ROOT}/final_model"
    local EVAL_MODEL="${TRAIN_ROOT}/${MODEL_NAME}"
    local TRAIN_LOG_DIR="${TRAIN_ROOT}/logs"
    local TENSORBOARD_DIR="${TRAIN_ROOT}/tensorboard"
    local RESULT_DIR="${TRAIN_ROOT}/results"
    local QUANT_CONFIG_PATH="${CODE_ROOT}/src/train.yaml"

    # Copy train config and set experiment tag
    cp "${TRAIN_YAML_DIR}/train.yaml" "${QUANT_CONFIG_PATH}"
    sed -i.bak "s/= \"exp_for_sed\"/= \"${RUN_NAME}\"/" "${CODE_ROOT}/src/config.py"

    # Step 1: Prepare environment
    echo "[Step 1/4] Preparing environment..."
    mkdir -p "${TRAIN_LOG_DIR}" "${TENSORBOARD_DIR}" "${RESULT_DIR}"
    cp -rf "${TEMPLATE_PATH}" "${EVAL_MODEL}"
    echo "  ✓ Output directories created"
    echo "  ✓ Template copied to: ${EVAL_MODEL}"
    echo ""

    # Step 2: Train
    echo "[Step 2/4] Starting distributed training with DeepSpeed..."
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    local TRAINING_LOG="${TRAIN_LOG_DIR}/train_${timestamp}.log"
    time {
        deepspeed --num_gpus=8 "${CODE_ROOT}/src/main.py"
    } 2>&1 | tee "${TRAINING_LOG}"
    echo "  ✓ Training completed → ${TRAINING_LOG}"
    echo ""

    # Step 3: Convert weights
    echo "[Step 3/4] Converting weights to quantized format..."
    python "${CONVERT_SCRIPT}" \
        --quant_config "${QUANT_CONFIG_PATH}" \
        --unquantized_model "${FINAL_MODEL}" \
        --quantized_model "${EVAL_MODEL}" \
        --dtype bfloat16
    echo "  ✓ Weight quantization completed"
    echo ""

    # Step 4: Evaluate
    echo "[Step 4/4] Running benchmark evaluation..."
    local TASK_NAME="EdgeRazor_Eval_QLLM"
    local TASK_NAME_CHAT="EdgeRazor_Eval_QLLM_Chat"

    # (a) A16KV16
    local QUANT_EVAL="a16kv16"
    local RESULT_DIR_SUB="${RESULT_DIR}/${QUANT_EVAL}"
    local EVAL_LOG="${RESULT_DIR_SUB}/eval_${QUANT_EVAL}_${timestamp}.log"
    mkdir -p "${RESULT_DIR_SUB}"

    time {
        accelerate launch -m lm_eval --model hf \
            --model_args pretrained="${EVAL_MODEL}" \
            --tasks "${TASK_NAME}" --log_samples --batch_size 32 \
            --output_path "${RESULT_DIR_SUB}/${TASK_NAME}" \
            --confirm_run_unsafe_code
    } 2>&1 | tee "${EVAL_LOG}"

    time {
        accelerate launch -m lm_eval --model hf \
            --model_args pretrained="${EVAL_MODEL}" --apply_chat_template \
            --tasks "${TASK_NAME_CHAT}" --log_samples --batch_size auto \
            --output_path "${RESULT_DIR_SUB}/${TASK_NAME_CHAT}" \
            --confirm_run_unsafe_code
    } 2>&1 | tee -a "${EVAL_LOG}"

    # (b) A8KV8
    QUANT_EVAL="a8kv8"
    RESULT_DIR_SUB="${RESULT_DIR}/${QUANT_EVAL}"
    EVAL_LOG="${RESULT_DIR_SUB}/eval_${QUANT_EVAL}_${timestamp}.log"
    mkdir -p "${RESULT_DIR_SUB}"

    time {
        accelerate launch -m lm_eval --model hf \
            --model_args pretrained="${EVAL_MODEL}",trust_remote_code=True \
            --tasks "${TASK_NAME}" --log_samples --batch_size 32 \
            --output_path "${RESULT_DIR_SUB}/${TASK_NAME}" \
            --confirm_run_unsafe_code --trust_remote_code
    } 2>&1 | tee "${EVAL_LOG}"

    time {
        accelerate launch -m lm_eval --model hf \
            --model_args pretrained="${EVAL_MODEL}",trust_remote_code=True --apply_chat_template \
            --tasks "${TASK_NAME_CHAT}" --log_samples --batch_size auto \
            --output_path "${RESULT_DIR_SUB}/${TASK_NAME_CHAT}" \
            --confirm_run_unsafe_code --trust_remote_code
    } 2>&1 | tee -a "${EVAL_LOG}"

    echo "  ✓ Evaluation completed → ${RESULT_DIR}"
    echo ""

    # Reset config.py for next run
    sed -i.bak "s/= \"${RUN_NAME}\"/= \"exp_for_sed\"/" "${CODE_ROOT}/src/config.py"

    echo "================================================================================"
    echo "Experiment [${RUN_NAME}] Completed!"
    echo "================================================================================"
    echo ""
}

################################################################################
# Run 1: Qwen3-1.7B W4A8KV8
################################################################################
run_experiment "W4A8KV8" "qwen3-1.7b_w4a8kv8" "train_w4" "${SCRIPT_DIR}/4-bit"

################################################################################
# Run 2: Qwen3-1.7B W1.58A8KV8
################################################################################
run_experiment "W1.58A8KV8" "qwen3-1.7b_w1.58a8kv8" "train_w1.58" "${SCRIPT_DIR}/1.58-bit"

echo "================================================================================"
echo "All experiments completed!"
echo "  W4A8KV8 output:    ${CODE_ROOT}/train_w4"
echo "  W1.58A8KV8 output: ${CODE_ROOT}/train_w1.58"
echo "================================================================================"
