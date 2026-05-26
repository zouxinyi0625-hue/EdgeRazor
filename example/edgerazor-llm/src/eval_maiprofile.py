#!/usr/bin/env python3
"""
eval_maiprofile.py — End-to-end evaluation wrapper.

1. Run inference with quantized model → generate layer outputs
2. Call MaiProfile's run_evaluation.py
3. Collect and display summary metrics

Usage:
    python eval_maiprofile.py \
        --model-path /path/to/quantized_model \
        --data-dir /path/to/curation_data/ \
        --output-root /path/to/output/ \
        --maiprofile-root /path/to/MaiProfile/maiprofilev3dev \
        --date-str 20260101 \
        [--eval-flags "--eval-layer1-delta-topics --eval-layer3-seasonality"] \
        [--baseline-root /path/to/baseline_output/] \
        [--skip-inference]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_inference_step(args):
    """Run inference_maiprofile.py to generate layer outputs."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "inference_maiprofile.py"),
        "--model-path", args.model_path,
        "--data-dir", args.data_dir,
        "--output-root", args.output_root,
        "--layer", args.layer,
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.date_str:
        cmd.extend(["--date-str", args.date_str])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    print("=" * 60)
    print("[Step 1/3] Running inference...")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_evaluation_step(args):
    """Call MaiProfile's run_evaluation.py on the generated outputs."""
    eval_script = Path(args.maiprofile_root) / "run_evaluation.py"
    if not eval_script.exists():
        print(f"[ERROR] run_evaluation.py not found at {eval_script}")
        return False

    cmd = [
        sys.executable, str(eval_script),
        "--output-root", args.output_root,
    ]
    # Add eval flags
    if args.eval_flags:
        cmd.extend(args.eval_flags.split())

    print("=" * 60)
    print("[Step 2/3] Running evaluation...")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def collect_summaries(output_root: str, baseline_root: str | None = None):
    """Collect summary JSONs and display comparison table."""
    print("=" * 60)
    print("[Step 3/3] Collecting evaluation summaries...")
    print("=" * 60)

    eval_dir = Path(output_root) / "eval"
    if not eval_dir.exists():
        print(f"  [WARN] No eval directory found at {eval_dir}")
        return

    # Find all summary files
    summaries = sorted(eval_dir.rglob("*_summary.json"))
    if not summaries:
        print("  [WARN] No summary files found.")
        return

    print(f"\n{'='*60}")
    print(f"{'Metric':<40} {'Quantized':>12}")
    if baseline_root:
        print(f"{'':40} {'Baseline':>12}")
    print(f"{'='*60}")

    for summary_path in summaries:
        with open(summary_path) as f:
            data = json.load(f)
        eval_name = summary_path.stem.replace("_summary", "")
        print(f"\n--- {eval_name} ---")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    print(f"  {key:<38} {value:>12.4f}")
                elif isinstance(value, dict):
                    for k2, v2 in value.items():
                        if isinstance(v2, (int, float)):
                            print(f"  {key}.{k2:<34} {v2:>12.4f}")

    # Compare with baseline if provided
    if baseline_root:
        baseline_eval_dir = Path(baseline_root) / "eval"
        if baseline_eval_dir.exists():
            print(f"\n{'='*60}")
            print("Baseline summaries:")
            baseline_summaries = sorted(baseline_eval_dir.rglob("*_summary.json"))
            for bp in baseline_summaries:
                with open(bp) as f:
                    bdata = json.load(f)
                print(f"\n--- {bp.stem.replace('_summary', '')} (baseline) ---")
                if isinstance(bdata, dict):
                    for key, value in bdata.items():
                        if isinstance(value, (int, float)):
                            print(f"  {key:<38} {value:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end MaiProfile evaluation")
    parser.add_argument("--model-path", required=True, help="Path to quantized model")
    parser.add_argument("--data-dir", required=True, help="Directory with curation_data_*.jsonl")
    parser.add_argument("--output-root", required=True, help="Output directory for inference + eval")
    parser.add_argument("--maiprofile-root", required=True,
                        help="Path to MaiProfile/maiprofilev3dev (containing run_evaluation.py)")
    parser.add_argument("--layer", required=True,
                        choices=["layer0_signal", "layer1_delta", "layer1_actual", "layer1_intent",
                                 "layer2_temporal", "layer3_persona", "layer3_seasonality"],
                        help="Which layer to evaluate")
    parser.add_argument("--date-str", default=None, help="Date string (YYYYMMDD). Auto-detected from data if omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--eval-flags", default="--eval-layer1-delta-topics --eval-layer3-seasonality",
                        help="Flags to pass to run_evaluation.py")
    parser.add_argument("--baseline-root", default=None,
                        help="Baseline output root for comparison")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference step (use existing outputs)")
    args = parser.parse_args()

    # Step 1: Inference
    if not args.skip_inference:
        run_inference_step(args)

    # Step 2: Evaluation
    run_evaluation_step(args)

    # Step 3: Collect and compare
    collect_summaries(args.output_root, args.baseline_root)

    print("\n✓ All done.")


if __name__ == "__main__":
    main()
