#!/usr/bin/env python3
"""
eval_local.py — Local evaluation without API calls.

Compares model inference output against ground truth from curation data.
Computes: parse success rate, JSON schema compliance, interest name matching,
topic count comparison, and text similarity.

Usage:
    python eval_local.py \
        --output-root /path/to/eval_output/ \
        --data-dir /path/to/data/maiprofile/ \
        --layer layer1_delta \
        [--date-str 20260512]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from maiprofile_parser import LAYER_PARSERS, safe_json_loads


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_ground_truth(data_dir: Path, layer: str) -> dict[str, str]:
    """Load ground truth: {user_id: assistant_response_text}"""
    # Try _test first, then _50k
    filename = f"curation_data_{layer}_test.jsonl"
    filepath = data_dir / filename
    if not filepath.exists():
        filename = f"curation_data_{layer}_50k.jsonl"
        filepath = data_dir / filename
    if not filepath.exists():
        print(f"[ERROR] No data file found for {layer}")
        return {}

    gt = {}
    for record in load_jsonl(filepath):
        metadata = record.get("metadata", {})
        user_id = metadata.get("user_id")
        messages = record.get("messages", [])
        if user_id and messages and messages[-1].get("role") == "assistant":
            gt[user_id] = messages[-1]["content"]
    return gt


def extract_interest_names(data: dict, layer: str) -> list[str]:
    """Extract interest names from a parsed layer output."""
    if layer in ("layer3_persona",):
        items = data.get("interest_personas", [])
    elif layer in ("layer3_seasonality",):
        items = data.get("interest_seasonality", [])
    elif layer == "layer0_signal":
        return []  # No interest names in layer0
    else:
        items = data.get("interests", [])
    return [item.get("interest_name", "") for item in items if isinstance(item, dict)]


def compute_name_overlap(pred_names: list[str], gt_names: list[str]) -> dict:
    """Compute precision/recall/f1 for interest names (case-insensitive exact match)."""
    pred_set = set(n.lower().strip() for n in pred_names if n)
    gt_set = set(n.lower().strip() for n in gt_names if n)

    if not pred_set and not gt_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gt_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matches = pred_set & gt_set
    precision = len(matches) / len(pred_set) if pred_set else 0
    recall = len(matches) / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_name_similarity(pred_names: list[str], gt_names: list[str]) -> float:
    """Compute fuzzy similarity between interest name lists using token overlap."""
    if not pred_names or not gt_names:
        return 0.0

    def tokenize(name):
        return set(name.lower().split())

    total_sim = 0.0
    for pn in pred_names:
        best_sim = 0.0
        p_tokens = tokenize(pn)
        for gn in gt_names:
            g_tokens = tokenize(gn)
            if p_tokens | g_tokens:
                sim = len(p_tokens & g_tokens) / len(p_tokens | g_tokens)
                best_sim = max(best_sim, sim)
        total_sim += best_sim
    return total_sim / len(pred_names)


def eval_layer0(pred: dict, gt_parsed: dict) -> dict:
    """Evaluate layer0_signal: compare signal annotations."""
    pred_signals = pred.get("signals", [])
    gt_signals = gt_parsed.get("result", gt_parsed.get("signals", []))

    if not gt_signals:
        return {"signal_count_pred": len(pred_signals), "signal_count_gt": 0}

    # Compare should_filter decisions
    correct = 0
    total = min(len(pred_signals), len(gt_signals))
    for i in range(total):
        p_filter = pred_signals[i].get("should_filter", False)
        g_filter = gt_signals[i].get("should_filter", False)
        if p_filter == g_filter:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return {
        "signal_count_pred": len(pred_signals),
        "signal_count_gt": len(gt_signals),
        "filter_accuracy": accuracy,
    }


def eval_interests_layer(pred: dict, gt_parsed: dict, layer: str) -> dict:
    """Evaluate interest-based layers."""
    pred_names = extract_interest_names(pred, layer)
    gt_names = extract_interest_names(gt_parsed, layer)

    name_overlap = compute_name_overlap(pred_names, gt_names)
    name_similarity = compute_name_similarity(pred_names, gt_names)

    return {
        "num_interests_pred": len(pred_names),
        "num_interests_gt": len(gt_names),
        "name_exact_precision": name_overlap["precision"],
        "name_exact_recall": name_overlap["recall"],
        "name_exact_f1": name_overlap["f1"],
        "name_fuzzy_similarity": name_similarity,
    }


def run_eval(output_root: Path, data_dir: Path, layer: str, date_str: str | None):
    """Main eval logic."""
    # Auto-detect date_str
    if date_str is None:
        dirs = [d for d in output_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        if dirs:
            date_str = sorted(dirs)[-1].name
        else:
            print("[ERROR] No date directory found in output_root. Specify --date-str.")
            return
        print(f"[INFO] Auto-detected date_str: {date_str}")

    # Load model outputs
    output_file = output_root / date_str / f"{layer}.jsonl"
    if not output_file.exists():
        print(f"[ERROR] Output file not found: {output_file}")
        return

    predictions = load_jsonl(output_file)
    print(f"[INFO] Loaded {len(predictions)} predictions from {output_file}")

    # Load ground truth
    gt_map = load_ground_truth(data_dir, layer)
    print(f"[INFO] Loaded {len(gt_map)} ground truth entries")

    # Metrics
    total = 0
    parse_success = 0
    parse_failures = 0
    per_sample_metrics = []

    for pred in predictions:
        user_id = pred.get("user_id")
        if not user_id or user_id not in gt_map:
            continue

        total += 1

        # Check parse success
        if pred.get("_parse_error"):
            parse_failures += 1
            continue
        parse_success += 1

        # Parse ground truth
        gt_text = gt_map[user_id]
        gt_parsed = safe_json_loads(gt_text)
        if gt_parsed is None:
            continue

        # Per-layer eval
        if layer == "layer0_signal":
            metrics = eval_layer0(pred, gt_parsed)
        else:
            metrics = eval_interests_layer(pred, gt_parsed, layer)

        per_sample_metrics.append(metrics)

    # Aggregate
    print(f"\n{'='*70}")
    print(f"  LOCAL EVALUATION RESULTS — {layer}")
    print(f"{'='*70}")
    print(f"  Total samples:         {total}")
    print(f"  Parse success:         {parse_success} ({parse_success/total*100:.1f}%)" if total > 0 else "")
    print(f"  Parse failures:        {parse_failures} ({parse_failures/total*100:.1f}%)" if total > 0 else "")

    if not per_sample_metrics:
        print("\n  [WARN] No valid samples to evaluate.")
        return

    # Average metrics
    print(f"\n  {'Metric':<35} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*65}")

    all_keys = per_sample_metrics[0].keys()
    summary = {}
    for key in all_keys:
        values = [m[key] for m in per_sample_metrics if key in m]
        if not values:
            continue
        mean_val = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        summary[key] = {"mean": mean_val, "min": min_val, "max": max_val}
        print(f"  {key:<35} {mean_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")

    print(f"{'='*70}")

    # Save summary
    summary_path = output_root / date_str / f"{layer}_local_eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "layer": layer,
            "total": total,
            "parse_success": parse_success,
            "parse_failures": parse_failures,
            "parse_success_rate": parse_success / total if total > 0 else 0,
            "metrics": summary,
        }, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Local evaluation (no API needed)")
    parser.add_argument("--output-root", required=True, help="Output root from inference")
    parser.add_argument("--data-dir", required=True, help="Directory with curation_data_*.jsonl (ground truth)")
    parser.add_argument("--layer", required=True,
                        choices=["layer0_signal", "layer1_delta", "layer1_actual", "layer1_intent",
                                 "layer2_temporal", "layer3_persona", "layer3_seasonality"],
                        help="Which layer to evaluate")
    parser.add_argument("--date-str", default=None, help="Date string (YYYYMMDD). Auto-detected if omitted.")
    args = parser.parse_args()

    run_eval(
        output_root=Path(args.output_root),
        data_dir=Path(args.data_dir),
        layer=args.layer,
        date_str=args.date_str,
    )


if __name__ == "__main__":
    main()
