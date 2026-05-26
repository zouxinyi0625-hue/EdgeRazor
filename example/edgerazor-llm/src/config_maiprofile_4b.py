#!/usr/bin/env python3
import os

# Environment paths
PATH_PREFIX = "/path/to/your/environment"  # Change this to your actual path prefix
CODE_ROOT = f"{PATH_PREFIX}/EdgeRazor"
SRC_ROOT = f"{CODE_ROOT}/example/edgerazor-llm/src"
DATA_ROOT = f"{CODE_ROOT}/data/maiprofile"

os.makedirs(DATA_ROOT, exist_ok=True)


class EdgeRazorTrainConfigForMaiProfile4B:
    config_path   = f"{SRC_ROOT}/train.yaml"
    ds_path       = f"{SRC_ROOT}/ds_z3_config_qwen3.json"
    teacher_path  = "Qwen/Qwen3-4B-Instruct-2507"
    student_path  = "Qwen/Qwen3-4B-Instruct-2507"
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
    dataset_path  = [_all_datasets[_layer]]
    output_dir    = f"{CODE_ROOT}/train_maiprofile_4b/{_layer}"
    final_model   = f"{output_dir}/final_model"

    # Data already contains system prompts
    add_system_prompt = False

    # Training — 4B needs smaller batch size than 1.7B
    max_seq_len   = 4096
    epoch         = 3
    steps         = -1
    optim         = "adamw_8bit"
    lr            = 1e-5       # slightly lower lr for larger model
    lr_scheduler  = "constant_with_warmup"
    min_lr        = 0
    warmup_ratio  = 0.05
    weight_decay  = 0.01
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00

    do_eval       = False

    per_device_bs  = 2         # 4B model needs smaller batch (vs 4 for 1.7B)
    grad_acc_steps = 32        # compensate smaller bs: effective bs = 2*32*8 = 512
    grad_chkpt     = True
    save_strategy  = "steps"
    save_steps     = 500
    eval_steps     = 500

    attn_implementation = "flash_attention_2"

    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001

    tag_name             = "exp_for_sed"
