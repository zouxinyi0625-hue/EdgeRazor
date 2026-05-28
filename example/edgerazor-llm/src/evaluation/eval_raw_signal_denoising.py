"""eval_raw_signal_denoising.py — Raw Signal Denoising Quality evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.io_utils import append_jsonl, read_text, write_json
from evaluation.llm_client import AsyncAzureOpenAI, invoke_chat, safe_json_loads

logger = logging.getLogger("edgerazor.eval_raw_signal_denoising")

PROMPT_PATH = Path(__file__).parent / "prompts" / "eval_raw_signal_denoising.md"
CONSISTENCY_PROMPT_PATH = Path(__file__).parent / "prompts" / "eval_raw_signal_denoising_consistency.md"
JSONL_FILENAME = "raw_signal_denoising.jsonl"
SUMMARY_FILENAME = "raw_signal_denoising_summary.json"
SCORE_DIMENSIONS = ["accuracy", "precision", "recall", "consistency", "intent_accuracy"]
BATCH_SIZE = 50
CONSISTENCY_MAX_SIGNALS = 200


class RawSignalDenoisingEvaluator:
    def __init__(self, client, model_name, output_root, max_tokens=16_000, reasoning_effort=None):
        self._client = client
        self._model = model_name
        self._model_key = model_name
        self._output_root = output_root
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._prompt = read_text(PROMPT_PATH)
        self._consistency_prompt = read_text(CONSISTENCY_PROMPT_PATH)
        self._all_scores: List[Dict[str, Any]] = []
        from evaluation.llm_reliability import LLMReliabilityTracker
        self._llm_tracker = LLMReliabilityTracker()

    @staticmethod
    def _build_signal_batch(signals, start_row):
        items = []
        for i, s in enumerate(signals):
            row = start_row + i
            entry = {"row": row, "action": s.get("Action", ""), "source": s.get("Source", ""), "detailed_source": s.get("DetailedSource", ""), "should_filter": s.get("should_filter", False)}
            if s.get("should_filter"):
                entry["filter_reason"] = s.get("filter_reason", "")
            else:
                entry["intent"] = s.get("intent", "")
            items.append(entry)
        return json.dumps(items, ensure_ascii=False, indent=2)

    async def _score_batch(self, signals, start_row, user_id, date_str, batch_idx=0, total_batches=1):
        user_content = self._build_signal_batch(signals, start_row)
        messages = [{"role": "system", "content": self._prompt}, {"role": "user", "content": user_content}]
        batch_label = f"batch {batch_idx+1}/{total_batches}" if total_batches > 1 else "single"
        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client, model=self._model, messages=messages,
                max_completion_tokens=self._max_tokens, reasoning_effort=self._reasoning_effort,
                response_format=None, caller="eval_l0_denoising_scoring",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] denoising eval %s failed: %s", user_id, date_str, batch_label, exc)
            self._llm_tracker.record("phase1_scoring", succeeded=False, model=self._model_key)
            return []
        if isinstance(parsed, dict):
            for key in ("result", "results", "signals", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            self._llm_tracker.record("phase1_scoring", succeeded=False, model=self._model_key)
            return []
        self._llm_tracker.record("phase1_scoring", succeeded=True, model=self._model_key)
        return parsed

    async def _eval_consistency(self, signals, user_id, date_str):
        import random
        eval_signals = signals
        if len(signals) > CONSISTENCY_MAX_SIGNALS:
            rng = random.Random(hash(user_id))
            eval_signals = rng.sample(signals, CONSISTENCY_MAX_SIGNALS)
        items = []
        for i, s in enumerate(eval_signals):
            items.append({"row": i, "action": s.get("Action", ""), "source": s.get("Source", ""), "should_filter": s.get("should_filter", False),
                          "filter_reason": s.get("filter_reason", "") if s.get("should_filter") else None,
                          "intent": s.get("intent", "") if not s.get("should_filter") else None})
        user_content = json.dumps(items, ensure_ascii=False, indent=2)
        messages = [{"role": "system", "content": self._consistency_prompt}, {"role": "user", "content": user_content}]
        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client, model=self._model, messages=messages,
                max_completion_tokens=self._max_tokens, reasoning_effort=self._reasoning_effort,
                response_format=None, caller="eval_l0_denoising_consistency",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] denoising consistency eval failed: %s", user_id, date_str, exc)
            self._llm_tracker.record("phase2_consistency", succeeded=False, model=self._model_key)
            return {}
        if not isinstance(parsed, dict):
            self._llm_tracker.record("phase2_consistency", succeeded=False, model=self._model_key)
            return {}
        self._llm_tracker.record("phase2_consistency", succeeded=True, model=self._model_key)
        return parsed

    async def evaluate_user(self, user_id, date_str, signals):
        if not signals:
            return None

        # Phase 1: Per-signal scoring
        all_scored = []
        if len(signals) <= BATCH_SIZE:
            all_scored = await self._score_batch(signals, 0, user_id, date_str)
        else:
            batches = [signals[i:i + BATCH_SIZE] for i in range(0, len(signals), BATCH_SIZE)]
            batch_results = await asyncio.gather(*[self._score_batch(b, i * BATCH_SIZE, user_id, date_str, i, len(batches)) for i, b in enumerate(batches)])
            for br in batch_results:
                all_scored.extend(br)

        for item in all_scored:
            row = item.get("row", -1)
            if 0 <= row < len(signals):
                sig = signals[row]
                item["action"] = sig.get("Action", "")
                item["source"] = sig.get("Source", "")
                item["should_filter"] = sig.get("should_filter", False)

        # Phase 2: Consistency
        consistency_result = await self._eval_consistency(signals, user_id, date_str)

        # Phase 3: Aggregation
        all_dq, filter_dq, keep_dq, intent_acc = [], [], [], []
        for item in all_scored:
            dq = item.get("decision_quality")
            if dq is None:
                continue
            dq = float(dq)
            all_dq.append(dq)
            if item.get("should_filter"):
                filter_dq.append(dq)
            else:
                keep_dq.append(dq)
                ia = item.get("intent_accuracy")
                if ia is not None:
                    intent_acc.append(float(ia))

        filter_count = sum(1 for s in signals if s.get("should_filter", False))
        keep_count = len(signals) - filter_count

        user_averages: Dict[str, Any] = {}
        if all_dq:
            user_averages["accuracy"] = round(sum(all_dq) / len(all_dq), 4)
        if filter_dq:
            user_averages["precision"] = round(sum(filter_dq) / len(filter_dq), 4)
        if keep_dq:
            user_averages["recall"] = round(sum(keep_dq) / len(keep_dq), 4)
        if consistency_result.get("overall_consistency") is not None:
            user_averages["consistency"] = consistency_result["overall_consistency"]
        if intent_acc:
            user_averages["intent_accuracy"] = round(sum(intent_acc) / len(intent_acc), 4)

        dim_values = [user_averages.get(d) for d in SCORE_DIMENSIONS if user_averages.get(d) is not None]
        if dim_values:
            user_averages["final_score"] = round(sum(dim_values) / len(dim_values), 4)

        result = {
            "module": "raw_signal_denoising", "user_id": user_id, "date_str": date_str,
            "signal_count": len(signals), "filter_count": filter_count, "keep_count": keep_count,
            "num_signals_scored": len(all_scored), "user_averages": user_averages,
            "signal_scores": all_scored, "consistency": consistency_result,
        }
        jpath = self._output_root / date_str / JSONL_FILENAME
        append_jsonl(jpath, result)
        self._all_scores.append(result)
        logger.info("[%s][%s] denoising eval: %d/%d scored. final=%.3f", user_id, date_str, len(all_scored), len(signals), user_averages.get("final_score", 0))
        return result

    def write_summary(self) -> Dict[str, Any]:
        if not self._all_scores:
            return {}
        import numpy as np
        all_dq, filter_dq, keep_dq, intent_acc_global = [], [], [], []
        for rec in self._all_scores:
            for item in rec.get("signal_scores", []):
                v = item.get("decision_quality")
                if v is None:
                    continue
                v = float(v)
                all_dq.append(v)
                if item.get("should_filter"):
                    filter_dq.append(v)
                else:
                    keep_dq.append(v)
                    ia = item.get("intent_accuracy")
                    if ia is not None:
                        intent_acc_global.append(float(ia))
        global_stats = {}
        if all_dq:
            global_stats["accuracy"] = {"mean": round(float(np.mean(all_dq)), 4), "std": round(float(np.std(all_dq)), 4), "count": len(all_dq)}
        if filter_dq:
            global_stats["precision"] = {"mean": round(float(np.mean(filter_dq)), 4), "std": round(float(np.std(filter_dq)), 4), "count": len(filter_dq)}
        if keep_dq:
            global_stats["recall"] = {"mean": round(float(np.mean(keep_dq)), 4), "std": round(float(np.std(keep_dq)), 4), "count": len(keep_dq)}
        if intent_acc_global:
            global_stats["intent_accuracy"] = {"mean": round(float(np.mean(intent_acc_global)), 4), "std": round(float(np.std(intent_acc_global)), 4), "count": len(intent_acc_global)}
        final_vals = [r["user_averages"].get("final_score") for r in self._all_scores if r["user_averages"].get("final_score") is not None]
        if final_vals:
            global_stats["final_score"] = {"mean": round(float(np.mean(final_vals)), 4), "std": round(float(np.std(final_vals)), 4), "count": len(final_vals)}
        total_signals = sum(r["signal_count"] for r in self._all_scores)
        total_filtered = sum(r["filter_count"] for r in self._all_scores)
        summary = {
            "module": "raw_signal_denoising", "total_evaluations": len(self._all_scores),
            "total_signals_scored": total_signals, "total_filtered": total_filtered,
            "filter_rate": round(total_filtered / total_signals, 4) if total_signals else 0,
            "llm_reliability": self._llm_tracker.summary(), "global": global_stats,
        }
        write_json(self._output_root / SUMMARY_FILENAME, summary)
        return summary
