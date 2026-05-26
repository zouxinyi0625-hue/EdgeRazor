#!/usr/bin/env python3
"""
split_data.py — Split curation data into train/test sets (90/10) by user_id.

Usage:
    python split_data.py --data-dir /path/to/data/maiprofile/ [--train-ratio 0.9] [--seed 42]

Input:  curation_data_layer*_50k.jsonl
Output: curation_data_layer*_train.jsonl, curation_data_layer*_test.jsonl
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def split_file(filepath: Path, train_ratio: float, seed: int):
    """Split a single JSONL file by user_id into train/test."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Group by user_id
    user_ids = list(set(r.get("metadata", {}).get("user_id", f"unknown_{i}") for i, r in enumerate(records)))
    random.seed(seed)
    random.shuffle(user_ids)

    split_idx = int(len(user_ids) * train_ratio)
    train_users = set(user_ids[:split_idx])
    test_users = set(user_ids[split_idx:])

    train_records = [r for r in records if r.get("metadata", {}).get("user_id") in train_users]
    test_records = [r for r in records if r.get("metadata", {}).get("user_id") in test_users]

    # Write output files
    stem = filepath.stem.replace("_50k", "")
    train_path = filepath.parent / f"{stem}_train.jsonl"
    test_path = filepath.parent / f"{stem}_test.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "file": filepath.name,
        "total_records": len(records),
        "total_users": len(user_ids),
        "train_users": len(train_users),
        "test_users": len(test_users),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_path": str(train_path),
        "test_path": str(test_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Split curation data into train/test")
    parser.add_argument("--data-dir", required=True, help="Directory containing *_50k.jsonl files")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train ratio (default 0.9)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("curation_data_layer*_50k.jsonl"))

    if not files:
        print(f"[ERROR] No *_50k.jsonl files found in {data_dir}")
        return

    print(f"Found {len(files)} files to split (ratio: {args.train_ratio}/{1-args.train_ratio:.1f}, seed: {args.seed})")
    print("=" * 80)

    all_stats = []
    for filepath in files:
        print(f"\nProcessing: {filepath.name}")
        stats = split_file(filepath, args.train_ratio, args.seed)
        all_stats.append(stats)
        print(f"  Users:   {stats['train_users']} train / {stats['test_users']} test (total: {stats['total_users']})")
        print(f"  Records: {stats['train_records']} train / {stats['test_records']} test (total: {stats['total_records']})")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'File':<45} {'Total':>7} {'Train':>7} {'Test':>7} {'Users':>7}")
    print("-" * 80)
    for s in all_stats:
        name = s["file"].replace("curation_data_", "").replace("_50k.jsonl", "")
        print(f"{name:<45} {s['total_records']:>7} {s['train_records']:>7} {s['test_records']:>7} {s['total_users']:>7}")
    print("-" * 80)
    total_train = sum(s["train_records"] for s in all_stats)
    total_test = sum(s["test_records"] for s in all_stats)
    total_all = sum(s["total_records"] for s in all_stats)
    print(f"{'TOTAL':<45} {total_all:>7} {total_train:>7} {total_test:>7}")
    print("=" * 80)
    print("\nDone. Train files: *_train.jsonl, Test files: *_test.jsonl")


if __name__ == "__main__":
    main()
