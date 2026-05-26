#!/usr/bin/env python3
"""
run_evaluation.py — Run LLM-as-judge evaluation on EdgeRazor inference outputs.

Reads inference output JSONL and ground truth from curation data, then runs
per-layer evaluators using Azure OpenAI as judge.

Usage:
    python run_evaluation.py \
        --output-root /path/to/inference_output/ \
        --data-dir /path/to/curation_data/ \
        --layer layer1_delta \
        [--date-str 20260512] \
        [--max-samples 100] \
        [--workers 7] \
        [--model gpt-5.1] \
        [--reasoning-effort none]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for evaluation package
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.llm_client import build_client, DEFAULT_MODEL
from evaluation.io_utils import read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edgerazor.run_evaluation")


def load_ground_truth(data_dir: Path, layer: str) -> Dict[str, Dict[str, Any]]:
    """Load ground truth from curation data: {user_id: parsed_json}."""
    filename = f"curation_data_{layer}_test.jsonl"
    filepath = data_dir / filename
    if not filepath.exists():
        filename = f"curation_data_{layer}_50k.jsonl"
        filepath = data_dir / filename
    if not filepath.exists():
        logger.error("No data file found for %s in %s", layer, data_dir)
        return {}

    gt = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            user_id = metadata.get("user_id")
            messages = record.get("messages", [])
            if user_id and messages and messages[-1].get("role") == "assistant":
                content = messages[-1]["content"]
                try:
                    gt[user_id] = json.loads(content)
                except json.JSONDecodeError:
                    gt[user_id] = {"_raw": content}
    return gt


def load_predictions(output_file: Path) -> List[Dict[str, Any]]:
    """Load inference predictions JSONL."""
    return read_jsonl(output_file)


async def run_evaluation(args):
    """Main evaluation logic."""
    data_dir = Path(args.data_dir)
    output_root = Path(args.output_root)
    layer = args.layer

    # Auto-detect date_str
    date_str = args.date_str
    if date_str is None:
        dirs = [d for d in output_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
        if dirs:
            date_str = sorted(dirs)[-1].name
        else:
            logger.error("No date directory found. Specify --date-str.")
            return
    logger.info("Using date_str: %s", date_str)

    # Load predictions
    output_file = output_root / date_str / f"{layer}.jsonl"
    if not output_file.exists():
        logger.error("Output file not found: %s", output_file)
        return
    predictions = load_predictions(output_file)
    logger.info("Loaded %d predictions from %s", len(predictions), output_file)

    # Load ground truth
    gt_map = load_ground_truth(data_dir, layer)
    logger.info("Loaded %d ground truth entries", len(gt_map))

    # Limit samples
    if args.max_samples > 0:
        predictions = predictions[:args.max_samples]

    # Build LLM client
    model = args.model
    client = build_client(timeout=300.0)
    reasoning_effort = args.reasoning_effort if args.reasoning_effort != "none" else None

    # Setup eval output directory
    eval_output = output_root / "eval" / f"{date_str}_{model}"
    eval_output.mkdir(parents=True, exist_ok=True)

    # Run per-layer evaluation
    semaphore = asyncio.Semaphore(args.workers)

    if layer == "layer0_signal":
        await _eval_layer0(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore)
    elif layer == "layer1_delta":
        await _eval_layer1_delta(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore)
    elif layer == "layer1_actual":
        await _eval_layer1_actual(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore, data_dir)
    elif layer == "layer1_intent":
        await _eval_layer1_intent(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore, data_dir)
    elif layer == "layer2_temporal":
        await _eval_layer2_temporal(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore)
    elif layer == "layer3_seasonality":
        await _eval_layer3_seasonality(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore)
    else:
        logger.warning("No LLM evaluator implemented for layer: %s", layer)

    logger.info("Evaluation complete. Results in: %s", eval_output)


async def _eval_layer0(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore):
    """Evaluate layer0_signal using RawSignalDenoisingEvaluator."""
    from evaluation.eval_raw_signal_denoising import RawSignalDenoisingEvaluator

    evaluator = RawSignalDenoisingEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            signals = pred.get("signals", [])
            if not signals:
                return
            await evaluator.evaluate_user(user_id, date_str, signals)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


async def _eval_layer1_delta(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore):
    """Evaluate layer1_delta using topics evaluator."""
    from evaluation.eval_layer1_topics import Layer1DeltaTopicsEvaluator

    evaluator = Layer1DeltaTopicsEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            interests = pred.get("interests", [])
            if not interests:
                return
            await evaluator.evaluate_user(user_id, date_str, interests)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


async def _eval_layer1_actual(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore, data_dir):
    """Evaluate layer1_actual using actual_activity evaluator.

    Needs layer1_delta output as context for topics/evidence.
    """
    from evaluation.eval_layer1_actual_activity import Layer1ActualActivityEvaluator

    evaluator = Layer1ActualActivityEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    # Load layer1_delta predictions for context (topics/evidence)
    delta_file = Path(predictions[0].get("_source_dir", "")) if predictions else None
    # Try to load delta from same output directory
    delta_output = eval_output.parent.parent / date_str / "layer1_delta.jsonl"
    delta_map: Dict[str, List] = {}
    if delta_output.exists():
        for rec in read_jsonl(delta_output):
            uid = rec.get("user_id")
            if uid:
                delta_map[uid] = rec.get("interests", [])

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            actual_interests = pred.get("interests", [])
            delta_interests = delta_map.get(user_id, actual_interests)
            if not actual_interests:
                return
            await evaluator.evaluate_user(user_id, date_str, delta_interests, actual_interests)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


async def _eval_layer1_intent(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore, data_dir):
    """Evaluate layer1_intent using inferred_intent evaluator."""
    from evaluation.eval_layer1_inferred_intent import Layer1InferredIntentEvaluator

    evaluator = Layer1InferredIntentEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    # Load layer1_delta and layer1_actual for context
    output_dir = eval_output.parent.parent / date_str
    delta_map: Dict[str, List] = {}
    actual_map: Dict[str, List] = {}
    delta_file = output_dir / "layer1_delta.jsonl"
    actual_file = output_dir / "layer1_actual.jsonl"
    if delta_file.exists():
        for rec in read_jsonl(delta_file):
            uid = rec.get("user_id")
            if uid:
                delta_map[uid] = rec.get("interests", [])
    if actual_file.exists():
        for rec in read_jsonl(actual_file):
            uid = rec.get("user_id")
            if uid:
                actual_map[uid] = rec.get("interests", [])

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            intent_interests = pred.get("interests", [])
            delta_interests = delta_map.get(user_id, intent_interests)
            actual_interests = actual_map.get(user_id, [])
            if not intent_interests:
                return
            await evaluator.evaluate_user(user_id, date_str, delta_interests, actual_interests, intent_interests)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


async def _eval_layer2_temporal(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore):
    """Evaluate layer2_temporal using temporal type evaluator."""
    from evaluation.eval_layer1_temporal_type import Layer2TemporalTypeEvaluator

    evaluator = Layer2TemporalTypeEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            interests = pred.get("interests", [])
            if not interests:
                return
            await evaluator.evaluate_user(user_id, date_str, interests)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


async def _eval_layer3_seasonality(client, model, predictions, gt_map, date_str, eval_output, reasoning_effort, semaphore):
    """Evaluate layer3_seasonality."""
    from evaluation.eval_layer3_seasonality import Layer3SeasonalityEvaluator

    evaluator = Layer3SeasonalityEvaluator(
        client=client, model_name=model, output_root=eval_output,
        reasoning_effort=reasoning_effort,
    )

    async def _task(pred):
        async with semaphore:
            user_id = pred.get("user_id")
            interests = pred.get("interests", pred.get("interest_seasonality", []))
            if not interests:
                return
            await evaluator.evaluate_user(user_id, date_str, interests)

    tasks = [_task(p) for p in predictions if p.get("user_id") and not p.get("_parse_error")]
    await asyncio.gather(*tasks)
    evaluator.write_summary()


def main():
    parser = argparse.ArgumentParser(description="LLM-as-judge evaluation for EdgeRazor inference")
    parser.add_argument("--output-root", required=True, help="Inference output root directory")
    parser.add_argument("--data-dir", required=True, help="Directory with curation_data_*.jsonl (ground truth)")
    parser.add_argument("--layer", required=True,
                        choices=["layer0_signal", "layer1_delta", "layer1_actual", "layer1_intent",
                                 "layer2_temporal", "layer3_persona", "layer3_seasonality"],
                        help="Which layer to evaluate")
    parser.add_argument("--date-str", default=None, help="Date string (YYYYMMDD). Auto-detected if omitted.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples to evaluate (0=all)")
    parser.add_argument("--workers", type=int, default=7, help="Concurrent LLM evaluation workers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Azure OpenAI model deployment (default: {DEFAULT_MODEL})")
    parser.add_argument("--reasoning-effort", default="none",
                        choices=["none", "low", "medium", "high"],
                        help="Reasoning effort for judge model")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
