#!/usr/bin/env python3
import os

# Environment paths
PATH_PREFIX = "/scratch/azureml/cr/j/ca4c1a241a09470a950b55e21b068fbf/exe/wd"
CODE_ROOT = f"{PATH_PREFIX}/EdgeRazor"
SRC_ROOT = f"{CODE_ROOT}/example/edgerazor-llm/src"
DATA_ROOT = f"{CODE_ROOT}/data/maiprofile"

os.makedirs(DATA_ROOT, exist_ok=True)


class EdgeRazorTrainConfigForMaiProfile8B:
    config_path   = f"{SRC_ROOT}/train.yaml"
    ds_path       = f"{SRC_ROOT}/ds_z3_config_qwen3.json"
    teacher_path  = "Qwen/Qwen3-8B"
    student_path  = "Qwen/Qwen3-8B"
    # All available layer datasets
    _all_datasets = {
        "layer0_signal":       f"{DATA_ROOT}/curation_data_layer0_signal_train.jsonl",
        "layer1_delta":        f"{DATA_ROOT}/curation_data_layer1_delta_train.jsonl",
        "layer1_actual":       f"{DATA_ROOT}/curation_data_layer1_actual_train.jsonl",
        "layer1_intent":       f"{DATA_ROOT}/curation_data_layer1_intent_train.jsonl",
        "layer2_temporal":     f"{DATA_ROOT}/curation_data_layer2_temporal_train.jsonl",
        "layer3_persona":      f"{DATA_ROOT}/curation_data_layer3_persona_train.jsonl",
        "layer3_seasonality":  f"{DATA_ROOT}/curation_data_layer3_seasonality_train.jsonl",
    }

    _layer = os.environ.get("LAYER", "layer0_signal")
    _quant_config = os.environ.get("QUANT_CONFIG", "")
    _run_tag = os.environ.get("RUN_TAG", "")
    dataset_path  = [_all_datasets[_layer]]
    _suffix = f"{_layer}_{_quant_config}" if _quant_config else _layer
    _suffix = f"{_suffix}_{_run_tag}" if _run_tag else _suffix
    output_dir    = f"{CODE_ROOT}/train_maiprofile_8b/{_suffix}"
    final_model   = f"{output_dir}/final_model"

    # Data already contains system prompts
    add_system_prompt = False

    # Training — 8B needs even smaller lr
    max_seq_len   = 8192
    epoch         = 3
    steps         = -1
    optim         = "adamw_8bit"
    lr            = 5e-6
    lr_scheduler  = "constant_with_warmup"
    min_lr        = 0
    warmup_ratio  = 0.05
    weight_decay  = 0.01
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00

    do_eval       = False

    per_device_bs  = 1
    grad_acc_steps = 64        # effective bs = 1*64*8 = 512
    grad_chkpt     = True
    save_strategy  = "steps"
    save_steps     = 200
    eval_steps     = 999999

    attn_implementation = "flash_attention_2"

    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001

    tag_name             = "exp_for_sed"
