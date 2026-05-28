#!/usr/bin/env python3
import os

# Environment paths
PATH_PREFIX = "/scratch/azureml/cr/j/ec2ababe2e7140d4b2703f49a4839996/exe/wd"
CODE_ROOT = f"{PATH_PREFIX}/EdgeRazor"
DATA_ROOT = f"{CODE_ROOT}/data"

# Create necessary directories if they don't exist
os.makedirs(DATA_ROOT, exist_ok=True)


# Model: Qwen3-0.6B
class EdgeRazorTrainConfigForQwen3_0_6B:
    config_path   = f"{CODE_ROOT}/src/train.yaml"
    ds_path       = f"{CODE_ROOT}/src/ds_z3_config_qwen3.json"
    teacher_path  = "Qwen/Qwen3-0.6B"
    student_path  = "Qwen/Qwen3-0.6B"
    dataset_path  = [
        f"{DATA_ROOT}/ii_7M_instruct.jsonl",              # 7.5M
        f"{DATA_ROOT}/ii_gen_1.4M_instruct.jsonl",        # 1.4M
        f"{DATA_ROOT}/tulu_0.6M_instruct.jsonl",          # 0.6M
        f"{DATA_ROOT}/am_1.4M_instruct.jsonl",            # 1.4M
        f"{DATA_ROOT}/task_0.2M_instruct.jsonl",          # 0.2M
    ]
    output_dir    = f"{CODE_ROOT}/train"
    final_model   = f"{output_dir}/final_model"
    
    # Training
    max_seq_len   = 1024
    epoch         = 1
    steps         = -1 # -1 means using epoch to control training length, 2k for 4bit Qwen3-0.6B
    optim         = "adamw_8bit"
    lr            = 2e-5
    lr_scheduler  = "constant_with_warmup"
    min_lr        = 0
    warmup_ratio  = 0.05
    weight_decay  = 0.01
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00
    
    # Evaluation/Validation
    do_eval       = False
    
    # Training environment
    per_device_bs  = 8        # Qwen3-0.6B=8
    grad_acc_steps = 16       # 
    grad_chkpt     = True     # gradient_checkpointing
    save_strategy  = "steps"  # options: "no", "epoch", "steps"
    save_steps     = 1000     # total_steps=2106*epoch=2106*4=8424 steps
    eval_steps     = 1000     # 
    
    # Attn
    attn_implementation  = "flash_attention_2"  # ["eager", "flash_attention_2", "sdpa"]
    
    # MoE Loss factors (deprecated for Dense LLMs)
    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001
    
    # Custom config
    tag_name             = "exp_for_sed"


# Model: Qwen3-1.7B
class EdgeRazorTrainConfigForQwen3_1_7B:
    config_path   = f"{CODE_ROOT}/src/train.yaml"
    ds_path       = f"{CODE_ROOT}/src/ds_z3_config_qwen3.json"
    teacher_path  = "Qwen/Qwen3-1.7B"
    student_path  = "Qwen/Qwen3-1.7B"
    dataset_path  = [
        f"{DATA_ROOT}/ii_7M_instruct.jsonl",              # 7.5M
        f"{DATA_ROOT}/ii_gen_1.4M_instruct.jsonl",        # 1.4M
        f"{DATA_ROOT}/tulu_0.6M_instruct.jsonl",          # 0.6M
        f"{DATA_ROOT}/am_1.4M_instruct.jsonl",            # 1.4M
        f"{DATA_ROOT}/task_0.2M_instruct.jsonl",          # 0.2M
    ]
    output_dir    = f"{CODE_ROOT}/train"
    final_model   = f"{output_dir}/final_model"
    
    # Training
    max_seq_len   = 1024
    epoch         = 2
    steps         = -1 # -1 means using epoch to control training length, 2k for 4bit Qwen3-1.7B
    optim         = "adamw_8bit"
    lr            = 2e-5
    lr_scheduler  = "constant_with_warmup"
    min_lr        = 0
    warmup_ratio  = 0.05
    weight_decay  = 0.01
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00
    
    # Evaluation/Validation
    do_eval       = False
    
    # Training environment
    per_device_bs  = 12       # Qwen3-1.7B=12
    grad_acc_steps = 16       # 
    grad_chkpt     = True     # gradient_checkpointing
    save_strategy  = "steps"  # options: "no", "epoch", "steps"
    save_steps     = 1000     # total_steps=2106*epoch=2106*4=8424 steps
    eval_steps     = 1000     # 
    
    # Attn
    attn_implementation  = "flash_attention_2"  # ["eager", "flash_attention_2", "sdpa"]
    
    # MoE Loss factors (deprecated for Dense LLMs)
    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001
    
    # Custom config
    tag_name             = "exp_for_sed"


# Model: MobileLLM-350M
class EdgeRazorTrainConfigForMobileLLM_350M:
    config_path   = f"{CODE_ROOT}/src/train.yaml"
    ds_path       = f"{CODE_ROOT}/src/ds_z3_config_qwen3.json"
    teacher_path  = "facebook/MobileLLM-ParetoQ-350M-BF16"
    student_path  = "facebook/MobileLLM-ParetoQ-350M-BF16"
    dataset_path  = [
        f"{DATA_ROOT}/ii_1.5M_base.jsonl",        # 1.48M
        f"{DATA_ROOT}/task_0.2M_instruct.jsonl",  # 0.2M
    ]
    output_dir    = f"{CODE_ROOT}/train"
    final_model   = f"{output_dir}/final_model"
    
    # Training
    max_seq_len   = 1024
    epoch         = -1  # 2: 4-bit; 4: 2.79-bit; 5: 1.88-bit, 1.58-bit [changed by main.py]
    steps         = -1
    optim         = "adamw_8bit"
    lr            = 2e-5
    lr_scheduler  = "cosine_with_min_lr"
    min_lr        = 0
    warmup_ratio  = 0.01
    weight_decay  = 0.01
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00
    
    # Evaluation/Validation
    do_eval       = False
    
    # Training environment
    per_device_bs  = 60       # 60*4*8=1920 total_batch_size
    grad_acc_steps = 4        # xxxxMiB / 81920MiB = xx.xx%
    save_strategy  = "steps"  # options: "no", "epoch", "steps"
    save_steps     = 500      # total_steps=2106*epoch=2106*4=8424 steps
    eval_steps     = 500      # =save_steps
    
    # Attn
    attn_implementation  = "flash_attention_2"  # ["eager", "flash_attention_2", "sdpa"]
    grad_chkpt           = True                 # gradient_checkpointing
    
    # MoE Loss factors
    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001
    
    # Custom config
    tag_name             = "exp_for_sed"
