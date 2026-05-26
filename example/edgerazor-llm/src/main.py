# ruff: noqa: F541, F401

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime

import numpy._core.multiarray
import torch
from config import EdgeRazorTrainConfigForQwen3_0_6B, EdgeRazorTrainConfigForQwen3_1_7B, EdgeRazorTrainConfigForMobileLLM_350M
from dataset import ReasoningDataset
from modeling import (
    create_student_model,
    create_teacher_model,
)
from torch.serialization import add_safe_globals
from trainer import EdgeRazorTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    TrainingArguments,
)

from edgerazor import EdgeRazor

# Change to EdgeRazorTrainConfigForQwen3_1_7B() or EdgeRazorTrainConfigForMobileLLM_350M() for different model configurations [Auto by run.sh]
config = EdgeRazorTrainConfigForQwen3_0_6B()

if "w4a8kv8" in config.tag_name:
    config.steps = 2_000
if "MobileLLM" in config.teacher_path:
    if "w4a8kv8" in config.tag_name:
        config.epoch = 2
    elif "w2.79a8kv8" in config.tag_name:
        config.epoch = 4
    elif "w1.88a8kv8" in config.tag_name:
        config.epoch = 5
    elif "w1.58a8kv8" in config.tag_name:
        config.epoch = 5
    else:
        raise ValueError(f"Unknown tag_name for MobileLLM: {config.tag_name}")



if __name__ == "__main__":
    add_safe_globals([numpy._core.multiarray._reconstruct])
    
    print("="*80)
    print("Final Training Configuration:")
    print(f"  Teacher Path: {config.teacher_path}")
    print(f"  Student Path: {config.student_path}")
    print(f"  Config Path: {config.config_path}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  Epochs: {config.epoch}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Per Device Batch Size: {config.per_device_bs}")
    print(f"  Gradient Accumulation Steps: {config.grad_acc_steps}")
    print(f"  Max Sequence Length: {config.max_seq_len}")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_name = os.path.basename(config.teacher_path)
    student_name = os.path.basename(config.student_path)
    tensorboard_name = f"{timestamp}"
    if config.tag_name:
        tensorboard_name = f"{config.tag_name}" + "_" + tensorboard_name
    
    device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
    print(f"Device map: {device_map}")
    print(f"TensorBoard experiment name: {tensorboard_name}")
    
    teacher_model = create_teacher_model(
        teacher_path=config.teacher_path,
        device_map=device_map,
        attn_implementation=config.attn_implementation,
        debug=True,
    )
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    edgerazor = EdgeRazor(config=config.config_path)
    student_model = create_student_model(
        student_path=config.student_path,
        device_map=device_map,
        attn_implementation=config.attn_implementation,
        edgerazor=edgerazor,
        debug=True,
    )
    student_model.train()
    
    print(f"Teacher model device: {next(teacher_model.parameters()).device}")
    print(f"Student model device: {next(student_model.parameters()).device}")
    tokenizer = AutoTokenizer.from_pretrained(config.student_path, use_cache=False, use_fast=True)
    
    args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=tensorboard_name,
        num_train_epochs=config.epoch,
        max_steps=config.steps,  # -1 means use epoch to control training length
        do_train=True,
        seed=3407,
        data_seed=3407,

        # total_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_devices
        per_device_train_batch_size=config.per_device_bs,
        gradient_accumulation_steps=config.grad_acc_steps,

        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler,
        lr_scheduler_kwargs={
            'min_lr': config.min_lr,
        },
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        adam_beta1=config.adam_beta1,
        adam_beta2=config.adam_beta2,
        adam_epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,

        bf16=True,
        fp16=False,
        gradient_checkpointing=config.grad_chkpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=20,
        save_safetensors=True,

        logging_steps=1,
        logging_dir=f"{config.output_dir}/tensorboard/{tensorboard_name}",
        report_to='tensorboard',
        logging_first_step=True,
        log_level='info',
        do_eval=config.do_eval,
        eval_strategy='no',

        dataloader_num_workers=16,
        dataloader_pin_memory=True,

        deepspeed=config.ds_path,
        ddp_find_unused_parameters=False,
        dataloader_drop_last=True,
    )

    # Data preparation
    dataset = ReasoningDataset(
        dataset_path=config.dataset_path,
        tokenizer=tokenizer,
        max_seq_len=config.max_seq_len,
        add_system_prompt=getattr(config, 'add_system_prompt', True),
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=config.max_seq_len,
        label_pad_token_id=-100,
        return_tensors='pt',
    )
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    batch_size = num_gpus * args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = len(dataset) // batch_size * args.num_train_epochs
    dataset_len = len(dataset)
    print(f"PyTorch batch_size: {batch_size}, HF-Trainer batch_size: {args.per_device_train_batch_size}*{args.gradient_accumulation_steps}*{args.world_size}={args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}")
    print(f"Total steps: {total_steps}, Dataset samples: {dataset_len}")

    trainer = EdgeRazorTrainer(
                                model=student_model,
                                teacher_model=teacher_model,
                                args=args,
                                train_dataset=dataset,
                                tokenizer=tokenizer,
                                data_collator=data_collator,
                                router_aux_loss_coef=config.router_aux_loss_coef,
                                router_z_loss_coef=config.router_z_loss_coef,
                                edgerazor=edgerazor,
                                )
    
    print(f"Starting training with the following configuration:")
    print(f"  - Teacher model: {teacher_name}")
    print(f"  - Student model: {student_name}")
    print(f"  - Dataset size: {dataset_len}")
    print(f"  - Configured max_seq_len: {config.max_seq_len}")
    print(f"  - Batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - TensorBoard logs: {args.logging_dir}")

    trainer.train(resume_from_checkpoint=False)

    trainer.model.to(torch.bfloat16)
    trainer.save_model(config.final_model)
    trainer.save_state()
