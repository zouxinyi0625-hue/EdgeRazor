"""eval_layer1_topics.py — Offline evaluation for layer1_delta topic quality.

Scores each topic on 4 dimensions: Utility, Precision, Coherence, Granularity.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.io_utils import append_jsonl, read_text, write_json
from evaluation.llm_client import AsyncAzureOpenAI, invoke_chat, safe_json_loads

logger = logging.getLogger("edgerazor.eval_topics")

SCORE_PILLARS = ["utility", "precision", "coherence", "granularity"]
PROMPT_PATH = Path(__file__).parent / "prompts" / "layer1_delta_eval_topics.md"
MODULE_NAME = "layer1_delta"
BATCH_SIZE = 50


class Layer1DeltaTopicsEvaluator:
    """Offline evaluator for layer1_delta topic quality."""

    def __init__(
        self,
        client: AsyncAzureOpenAI,
        model_name: str,
        output_root: Path,
        max_tokens: int = 16_000,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self._client = client
        self._model = model_name
        self._model_key = model_name
        self._output_root = output_root
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._prompt = read_text(PROMPT_PATH)
        self._all_scores: List[Dict[str, Any]] = []
        from evaluation.llm_reliability import LLMReliabilityTracker
        self._llm_tracker = LLMReliabilityTracker()

    @staticmethod
    def _flatten_topics(interests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat_topics = []
        for interest in interests:
            interest_name = interest.get("interest_name", "")
            for t in interest.get("topics", []):
                if not isinstance(t, dict):
                    continue
                evidence_actions = []
                for e in t.get("evidence", []):
                    if isinstance(e, dict):
                        evidence_actions.append(
                            f"{e.get('date', '')} | {', '.join(e.get('source', []))} | {e.get('action', '')}"
                        )
                    elif isinstance(e, str):
                        evidence_actions.append(e)
                flat_topics.append({
                    "interest_name": interest_name,
                    "topic": t.get("topic", ""),
                    "source": t.get("source", []),
                    "evidence": evidence_actions,
                })
        return flat_topics

    @staticmethod
    def _build_user_prompt(flat_topics: List[Dict[str, Any]]) -> str:
        stripped = []
        for ft in flat_topics:
            stripped.append({
                "topic": ft.get("topic", ""),
                "source": ft.get("source", []),
                "evidence": ft.get("evidence", []),
            })
        topics_str = json.dumps(stripped, ensure_ascii=False, indent=2)
        return f"topics:\n{topics_str}"

    async def _call_llm(
        self,
        flat_topics: List[Dict[str, Any]],
        user_id: str,
        date_str: str,
        batch_idx: int = 0,
        total_batches: int = 1,
    ) -> List[Dict[str, Any]]:
        user_prompt = self._build_user_prompt(flat_topics)
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": user_prompt},
        ]
        batch_label = f"batch {batch_idx+1}/{total_batches}" if total_batches > 1 else "single"

        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client,
                model=self._model,
                messages=messages,
                max_completion_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                response_format=None,
                caller="eval_l1_topics",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] eval_topics LLM call failed (%s): %s", user_id, date_str, batch_label, exc)
            self._llm_tracker.record("scoring", succeeded=False, model=self._model_key)
            return []

        if isinstance(parsed, dict):
            for key in ("result", "results", "evaluations", "topics", "scores", "data", "output"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            if isinstance(parsed, dict):
                lists = [v for v in parsed.values() if isinstance(v, list)]
                if len(lists) == 1:
                    parsed = lists[0]
        if not isinstance(parsed, list):
            logger.warning("[%s][%s] eval_topics: unexpected response format (%s).", user_id, date_str, batch_label)
            return []

        for idx, item in enumerate(parsed):
            if "interest_name" not in item and idx < len(flat_topics):
                item["interest_name"] = flat_topics[idx].get("interest_name", "")

        self._llm_tracker.record("scoring", succeeded=True, model=self._model_key)
        return parsed

    async def evaluate_user(
        self,
        user_id: str,
        date_str: str,
        interests: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        total_topics = sum(len(i.get("topics", [])) for i in interests if isinstance(i, dict))
        if total_topics == 0:
            return None

        flat_topics = self._flatten_topics(interests)

        import asyncio
        if len(flat_topics) <= BATCH_SIZE:
            parsed = await self._call_llm(flat_topics, user_id, date_str)
        else:
            batches = [flat_topics[i:i + BATCH_SIZE] for i in range(0, len(flat_topics), BATCH_SIZE)]
            coros = [self._call_llm(batch, user_id, date_str, idx, len(batches)) for idx, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*coros)
            parsed = []
            for br in batch_results:
                parsed.extend(br)

        if not isinstance(parsed, list):
            return None

        # Compute per-user averages
        pillar_sums = {p: 0.0 for p in SCORE_PILLARS}
        valid_count = 0
        for item in parsed:
            scores = item.get("scores", {})
            if not scores:
                continue
            valid_count += 1
            for p in SCORE_PILLARS:
                val = scores.get(p)
                if val is not None:
                    pillar_sums[p] += float(val)

        user_avg: Dict[str, float] = {}
        if valid_count > 0:
            user_avg = {p: round(pillar_sums[p] / valid_count, 4) for p in SCORE_PILLARS}
            def _norm(p, v):
                if p == "granularity":
                    return 10.0 * v
                return v
            user_avg["final_score"] = round(
                sum(_norm(p, user_avg[p]) for p in SCORE_PILLARS) / len(SCORE_PILLARS), 4
            )

        result = {
            "module": MODULE_NAME,
            "user_id": user_id,
            "date_str": date_str,
            "num_topics_evaluated": valid_count,
            "num_topics_expected": total_topics,
            "topic_scores": parsed,
            "user_averages": user_avg,
        }

        jpath = self._output_root / date_str / "layer1_delta_eval_topics.jsonl"
        append_jsonl(jpath, result)
        self._all_scores.append(result)

        logger.info(
            "[%s][%s] eval_topics: %d/%d topics scored. final=%.3f",
            user_id, date_str, valid_count, total_topics, user_avg.get("final_score", 0),
        )
        return result

    def write_summary(self) -> Dict[str, Any]:
        if not self._all_scores:
            return {}

        import numpy as np
        from collections import defaultdict

        all_pillars = SCORE_PILLARS + ["final_score"]

        pillar_all: Dict[str, List[float]] = {p: [] for p in all_pillars}
        for rec in self._all_scores:
            for item in rec.get("topic_scores", []):
                scores = item.get("scores", {})
                for p in SCORE_PILLARS:
                    val = scores.get(p)
                    if val is not None:
                        pillar_all[p].append(float(val))
                topic_vals = [
                    (10.0 * float(scores[p])) if p == "granularity" else float(scores[p])
                    for p in SCORE_PILLARS if scores.get(p) is not None
                ]
                if topic_vals:
                    pillar_all["final_score"].append(sum(topic_vals) / len(topic_vals))

        global_avg: Dict[str, Any] = {}
        for p in all_pillars:
            vals = pillar_all[p]
            if vals:
                global_avg[p] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "count": len(vals),
                }

        summary = {
            "module": MODULE_NAME,
            "total_evaluations": len(self._all_scores),
            "total_topics_scored": sum(r["num_topics_evaluated"] for r in self._all_scores),
            "llm_reliability": self._llm_tracker.summary(),
            "global": global_avg,
        }

        write_json(self._output_root / "layer1_delta_eval_topics_summary.json", summary)
        g_final = global_avg.get("final_score", {})
        logger.info(
            "eval_topics summary: %d evaluations, global final_score=%.3f ± %.3f",
            summary["total_evaluations"], g_final.get("mean", 0), g_final.get("std", 0),
        )
        return summary
