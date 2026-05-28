"""eval_layer1_actual_activity.py — Evaluate actual_activity quality (inference_quality 1-10)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.io_utils import append_jsonl, read_text, write_json
from evaluation.llm_client import AsyncAzureOpenAI, invoke_chat, safe_json_loads

logger = logging.getLogger("edgerazor.eval_actual_activity")

SCORE_PILLARS = ["inference_quality"]
PROMPT_PATH = Path(__file__).parent / "prompts" / "layer1_actual_eval_activity.md"
MODULE_NAME = "layer1_actual"
JSONL_FILENAME = "layer1_actual_eval_activity.jsonl"
SUMMARY_FILENAME = "layer1_actual_eval_activity_summary.json"


class Layer1ActualActivityEvaluator:
    def __init__(self, client, model_name, output_root, max_tokens=16_000, reasoning_effort=None):
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
    def _build_user_prompt(delta_interests, actual_interests):
        actual_map = {i.get("interest_name", ""): i.get("actual_activity", "") for i in actual_interests}
        interest_items = []
        for interest in delta_interests:
            if not isinstance(interest, dict):
                continue
            name = interest.get("interest_name", "")
            actual = actual_map.get(name, "")
            if not actual:
                continue
            topics_info = []
            for t in interest.get("topics", []):
                if not isinstance(t, dict):
                    continue
                evidence_actions = []
                for e in t.get("evidence", []):
                    if isinstance(e, dict):
                        evidence_actions.append(f"{e.get('date', '')} | {', '.join(e.get('source', []))} | {e.get('action', '')}")
                    elif isinstance(e, str):
                        evidence_actions.append(e)
                topics_info.append({"topic": t.get("topic", ""), "source": t.get("source", []), "evidence": evidence_actions})
            interest_items.append({"interest_name": name, "actual_activity": actual, "topics": topics_info})
        return f"interests:\n{json.dumps(interest_items, ensure_ascii=False, indent=2)}"

    async def evaluate_user(self, user_id, date_str, delta_interests, actual_interests):
        total_interests = sum(1 for i in actual_interests if isinstance(i, dict) and i.get("actual_activity"))
        if total_interests == 0:
            return None

        user_prompt = self._build_user_prompt(delta_interests, actual_interests)
        messages = [{"role": "system", "content": self._prompt}, {"role": "user", "content": user_prompt}]

        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client, model=self._model, messages=messages,
                max_completion_tokens=self._max_tokens, reasoning_effort=self._reasoning_effort,
                response_format=None, caller="eval_l1_actual_activity",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] eval_actual_activity failed: %s", user_id, date_str, exc)
            self._llm_tracker.record("scoring", succeeded=False, model=self._model_key)
            return None

        if isinstance(parsed, dict):
            for key in ("result", "results", "evaluations", "interests", "scores", "data", "output"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            if isinstance(parsed, dict):
                lists = [v for v in parsed.values() if isinstance(v, list)]
                if len(lists) == 1:
                    parsed = lists[0]
        if not isinstance(parsed, list):
            self._llm_tracker.record("scoring", succeeded=False, model=self._model_key)
            return None

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

        user_avg = {p: round(pillar_sums[p] / valid_count, 4) for p in SCORE_PILLARS} if valid_count > 0 else {}

        result = {
            "module": MODULE_NAME, "user_id": user_id, "date_str": date_str,
            "num_interests_evaluated": valid_count, "num_interests_expected": total_interests,
            "interest_scores": parsed, "user_averages": user_avg,
        }
        self._llm_tracker.record("scoring", succeeded=True, model=self._model_key)
        jpath = self._output_root / date_str / JSONL_FILENAME
        append_jsonl(jpath, result)
        self._all_scores.append(result)
        logger.info("[%s][%s] eval_actual_activity: %d/%d scored. iq=%.3f", user_id, date_str, valid_count, total_interests, user_avg.get("inference_quality", 0))
        return result

    def write_summary(self) -> Dict[str, Any]:
        if not self._all_scores:
            return {}
        import numpy as np
        pillar_all: Dict[str, List[float]] = {p: [] for p in SCORE_PILLARS}
        for rec in self._all_scores:
            for item in rec.get("interest_scores", []):
                scores = item.get("scores", {})
                for p in SCORE_PILLARS:
                    val = scores.get(p)
                    if val is not None:
                        pillar_all[p].append(float(val))
        global_avg = {}
        for p in SCORE_PILLARS:
            vals = pillar_all[p]
            if vals:
                global_avg[p] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "count": len(vals)}
        summary = {
            "module": MODULE_NAME, "total_evaluations": len(self._all_scores),
            "total_interests_scored": sum(r["num_interests_evaluated"] for r in self._all_scores),
            "llm_reliability": self._llm_tracker.summary(), "global": global_avg,
        }
        write_json(self._output_root / SUMMARY_FILENAME, summary)
        return summary
