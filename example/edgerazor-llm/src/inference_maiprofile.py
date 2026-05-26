#!/usr/bin/env python3
"""
inference_maiprofile.py — Run quantized Qwen3-1.7B inference on MaiProfile
curation data and produce output JSONL files compatible with run_evaluation.py.

Usage (single GPU):
    python inference_maiprofile.py \
        --model-path Qwen/Qwen3-1.7B \
        --data-dir /path/to/curation_data/ \
        --output-root /path/to/output/ \
        --layer layer1_delta

Usage (multi-GPU data parallel):
    python inference_maiprofile.py \
        --model-path Qwen/Qwen3-1.7B \
        --data-dir /path/to/curation_data/ \
        --output-root /path/to/output/ \
        --layer layer1_delta \
        --num-gpus 8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from maiprofile_parser import CURATION_FILE_TO_LAYER, LAYER_PARSERS


def load_model(model_path: str, trust_remote_code: bool = False):
    """Load quantized model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model, tokenizer


def read_curation_data(filepath: str) -> list[dict]:
    """Read JSONL curation data file. Each line has a 'messages' field."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_prompt_messages(record: dict) -> list[dict]:
    """Extract prompt messages (all but last assistant response) from a record."""
    messages = record.get("messages", [])
    if not messages:
        return []
    # Remove the last assistant response (ground truth)
    if messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def generate_response(
    model, tokenizer, messages: list[dict], max_new_tokens: int = 2048
) -> str:
    """Generate model response for a given conversation."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    # Decode only newly generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_responses_batch(
    model, tokenizer, messages_batch: list[list[dict]], max_new_tokens: int = 2048
) -> list[str]:
    """Generate model responses for a batch of conversations."""
    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        for msgs in messages_batch
    ]
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    results = []
    for i in range(len(texts)):
        generated = outputs[i][input_len:]
        results.append(tokenizer.decode(generated, skip_special_tokens=True))
    return results


def detect_date_str(data_dir: Path) -> str:
    """Auto-detect date_str from the first record's metadata in any curation file."""
    for filename in CURATION_FILE_TO_LAYER:
        filepath = data_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if line:
                    record = json.loads(line)
                    date = record.get("metadata", {}).get("date")
                    if date:
                        return date
    return "20260101"


def launch_multi_gpu(args):
    """Spawn one process per GPU, each handles a shard of data."""
    processes = []
    for gpu_id in range(args.num_gpus):
        cmd = [
            sys.executable, __file__,
            "--model-path", args.model_path,
            "--data-dir", args.data_dir,
            "--output-root", args.output_root,
            "--layer", args.layer,
            "--max-new-tokens", str(args.max_new_tokens),
            "--batch-size", str(args.batch_size),
            "--gpu-id", str(gpu_id),
            "--num-workers", str(args.num_gpus),
        ]
        if args.date_str:
            cmd.extend(["--date-str", args.date_str])
        if args.max_samples > 0:
            cmd.extend(["--max-samples", str(args.max_samples)])
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        p = subprocess.Popen(cmd)
        processes.append(p)
        print(f"[Launcher] Started worker {gpu_id} (pid={p.pid})")

    # Wait for all workers
    for p in processes:
        p.wait()

    # Merge shard outputs
    print("[Launcher] All workers finished. Merging outputs...")
    merge_shards(args)
    print("[Launcher] Done.")


def merge_shards(args):
    """Merge per-worker shard files into a single output file."""
    data_dir = Path(args.data_dir)
    date_str = args.date_str or detect_date_str(data_dir)
    output_dir = Path(args.output_root) / date_str
    layer = args.layer

    final_path = output_dir / f"{layer}.jsonl"
    with open(final_path, "w", encoding="utf-8") as out_f:
        for gpu_id in range(args.num_gpus):
            shard_path = output_dir / f"{layer}_shard{gpu_id}.jsonl"
            if shard_path.exists():
                with open(shard_path, "r", encoding="utf-8") as sf:
                    for line in sf:
                        out_f.write(line)
                shard_path.unlink()  # Clean up shard file
    print(f"  ✓ Merged → {final_path}")


def run_inference(
    model,
    tokenizer,
    data_dir: Path,
    output_root: Path,
    layer: str,
    date_str: str | None = None,
    max_new_tokens: int = 2048,
    max_samples: int = 0,
    worker_id: int = 0,
    num_workers: int = 1,
    batch_size: int = 8,
):
    """Run inference on a single layer's test data and write output."""
    if date_str is None:
        date_str = detect_date_str(data_dir)
        print(f"[Worker {worker_id}] Auto-detected date_str: {date_str}")

    output_dir = output_root / date_str
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the test file for this layer
    filename = f"curation_data_{layer}_test.jsonl"
    filepath = data_dir / filename
    if not filepath.exists():
        # Fallback to _50k
        filename = f"curation_data_{layer}_50k.jsonl"
        filepath = data_dir / filename
    if not filepath.exists():
        print(f"[ERROR] No data file found for {layer} in {data_dir}")
        return

    print(f"[Worker {worker_id}] Processing {filename} → {layer}")
    records = read_curation_data(str(filepath))
    parser = LAYER_PARSERS[layer]

    # Limit total samples first, then shard across workers
    if max_samples > 0:
        records = records[:max_samples]
    records = [r for i, r in enumerate(records) if i % num_workers == worker_id]
    print(f"[Worker {worker_id}] Assigned {len(records)} records (batch_size={batch_size})")

    # Write to shard file if multi-GPU, else directly to final
    if num_workers > 1:
        output_path = output_dir / f"{layer}_shard{worker_id}.jsonl"
    else:
        output_path = output_dir / f"{layer}.jsonl"

    parse_failures = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for batch_start in tqdm(range(0, len(records), batch_size),
                                desc=f"[Worker {worker_id}]", position=worker_id):
            batch_records = records[batch_start:batch_start + batch_size]

            # Prepare batch
            batch_messages = []
            batch_meta = []
            for i_abs, record in enumerate(batch_records, start=batch_start):
                messages = extract_prompt_messages(record)
                if not messages:
                    continue
                metadata = record.get("metadata", {})
                user_id = metadata.get("user_id", f"user_{i_abs}")
                rec_date = metadata.get("date", date_str)
                batch_messages.append(messages)
                batch_meta.append((user_id, rec_date))

            if not batch_messages:
                continue

            # Batch generate
            raw_outputs = generate_responses_batch(
                model, tokenizer, batch_messages, max_new_tokens
            )

            # Parse and write
            for (user_id, rec_date), raw_output in zip(batch_meta, raw_outputs):
                parsed = parser(user_id, rec_date, raw_output)
                if parsed is None:
                    parse_failures += 1
                    parsed = {
                        "user_id": user_id,
                        "date": rec_date,
                        "layer": layer,
                        "_raw": raw_output,
                        "_parse_error": True,
                    }
                out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")

    print(f"[Worker {worker_id}] ✓ Written {output_path} (parse failures: {parse_failures}/{len(records)})")


def run_sample(args):
    """Run a few samples and print full input/output for debugging."""
    data_dir = Path(args.data_dir)
    layer = args.layer

    # Find data file
    filename = f"curation_data_{layer}_test.jsonl"
    filepath = data_dir / filename
    if not filepath.exists():
        filename = f"curation_data_{layer}_50k.jsonl"
        filepath = data_dir / filename
    if not filepath.exists():
        print(f"[ERROR] No data file found for {layer} in {data_dir}")
        return

    records = read_curation_data(str(filepath))[:args.sample]
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.trust_remote_code)
    print(f"Model loaded. Running {len(records)} samples.\n")

    parser_fn = LAYER_PARSERS[layer]

    for i, record in enumerate(records):
        messages = extract_prompt_messages(record)
        if not messages:
            continue

        metadata = record.get("metadata", {})
        user_id = metadata.get("user_id", f"user_{i}")
        date = metadata.get("date", "unknown")

        # Ground truth
        gt = record.get("messages", [])[-1].get("content", "") if record.get("messages") else ""

        print(f"{'='*80}")
        print(f"[Sample {i+1}/{len(records)}] user_id={user_id}, date={date}")
        print(f"{'='*80}")

        # Print input messages
        print("\n--- INPUT MESSAGES ---")
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if len(content) > 500:
                content = content[:500] + f"... ({len(content)} chars total)"
            print(f"[{role}]: {content}\n")

        # Generate
        raw_output = generate_response(model, tokenizer, messages, args.max_new_tokens)

        print("--- MODEL OUTPUT (raw) ---")
        print(raw_output)
        print()

        # Parse
        parsed = parser_fn(user_id, date, raw_output)
        print("--- PARSED ---")
        if parsed:
            print(json.dumps(parsed, ensure_ascii=False, indent=2)[:2000])
        else:
            print("[PARSE FAILED]")

        print("\n--- GROUND TRUTH ---")
        if len(gt) > 1000:
            gt = gt[:1000] + f"... ({len(gt)} chars total)"
        print(gt)
        print()


def main():
    parser = argparse.ArgumentParser(description="MaiProfile inference with quantized model")
    parser.add_argument("--model-path", required=True, help="Path to quantized model checkpoint")
    parser.add_argument("--data-dir", required=True, help="Directory containing curation_data_*.jsonl files")
    parser.add_argument("--output-root", required=True, help="Output root directory")
    parser.add_argument("--layer", required=True,
                        choices=["layer0_signal", "layer1_delta", "layer1_actual", "layer1_intent",
                                 "layer2_temporal", "layer3_persona", "layer3_seasonality"],
                        help="Which layer to run inference on")
    parser.add_argument("--date-str", default=None, help="Date string for output folder (YYYYMMDD). Auto-detected from data if omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens to generate")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples per worker (0=all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8)")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for model loading")
    parser.add_argument("--sample", type=int, default=0, help="Run N samples in debug mode (print full input/output)")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs for data parallel inference")
    parser.add_argument("--gpu-id", type=int, default=None, help="(Internal) GPU worker ID, used by multi-GPU launcher")
    parser.add_argument("--num-workers", type=int, default=None, help="(Internal) Total workers, used by multi-GPU launcher")
    args = parser.parse_args()

    if args.sample > 0:
        run_sample(args)
    elif args.num_gpus > 1 and args.gpu_id is None:
        # Launcher mode: spawn one process per GPU
        launch_multi_gpu(args)
    else:
        # Single GPU or worker mode
        gpu_id = args.gpu_id or 0
        num_workers = args.num_workers or 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"[Worker {gpu_id}/{num_workers}] Loading model from {args.model_path}...")
        model, tokenizer = load_model(args.model_path, args.trust_remote_code)
        print(f"[Worker {gpu_id}/{num_workers}] Model loaded.")

        run_inference(
            model=model,
            tokenizer=tokenizer,
            data_dir=Path(args.data_dir),
            output_root=Path(args.output_root),
            layer=args.layer,
            date_str=args.date_str,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            worker_id=gpu_id,
            num_workers=num_workers,
            batch_size=args.batch_size,
        )
        print(f"[Worker {gpu_id}/{num_workers}] Done.")


if __name__ == "__main__":
    main()
