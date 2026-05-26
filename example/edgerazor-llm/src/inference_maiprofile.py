#!/usr/bin/env python3
"""
inference_maiprofile.py — Run quantized Qwen3-1.7B inference on MaiProfile
curation data and produce output JSONL files compatible with run_evaluation.py.

Usage:
    python inference_maiprofile.py \
        --model-path /path/to/checkpoint_or_exported_model \
        --data-dir /path/to/curation_data/ \
        --output-root /path/to/output/ \
        --date-str 20260101 \
        [--max-new-tokens 2048] \
        [--batch-size 4] \
        [--trust-remote-code]
"""

import argparse
import json
import os
import sys
from pathlib import Path

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


def run_inference(
    model,
    tokenizer,
    data_dir: Path,
    output_root: Path,
    layer: str,
    date_str: str | None = None,
    max_new_tokens: int = 2048,
):
    """Run inference on a single layer's test data and write output."""
    if date_str is None:
        date_str = detect_date_str(data_dir)
        print(f"[INFO] Auto-detected date_str: {date_str}")

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

    print(f"[INFO] Processing {filename} → {layer}")
    records = read_curation_data(str(filepath))
    parser = LAYER_PARSERS[layer]
    output_path = output_dir / f"{layer}.jsonl"

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records):
            messages = extract_prompt_messages(record)
            if not messages:
                continue

            # Extract metadata
            metadata = record.get("metadata", {})
            user_id = metadata.get("user_id", f"user_{i}")
            rec_date = metadata.get("date", date_str)

            # Generate
            raw_output = generate_response(
                model, tokenizer, messages, max_new_tokens
            )

            # Parse into structured format
            parsed = parser(user_id, rec_date, raw_output)
            if parsed is None:
                print(f"  [WARN] Failed to parse output for user {user_id}")
                parsed = {
                    "user_id": user_id,
                    "date": rec_date,
                    "layer": layer,
                    "_raw": raw_output,
                    "_parse_error": True,
                }

            out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(records)}] done")

    print(f"  ✓ Written {output_path}")


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
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code for model loading")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, args.trust_remote_code)
    print("Model loaded.")

    run_inference(
        model=model,
        tokenizer=tokenizer,
        data_dir=Path(args.data_dir),
        output_root=Path(args.output_root),
        layer=args.layer,
        date_str=args.date_str,
        max_new_tokens=args.max_new_tokens,
    )
    print("Done.")


if __name__ == "__main__":
    main()
