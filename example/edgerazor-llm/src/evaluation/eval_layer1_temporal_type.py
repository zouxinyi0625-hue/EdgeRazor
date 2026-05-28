"""eval_layer1_temporal_type.py — Evaluate temporal type accuracy (binary 0/1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.io_utils import append_jsonl, read_text, write_json
from evaluation.llm_client import AsyncAzureOpenAI, invoke_chat, safe_json_loads

logger = logging.getLogger("edgerazor.eval_temporal_type")

PROMPT_PATH = Path(__file__).parent / "prompts" / "layer2_eval_temporal_type.md"
MODULE_NAME = "layer2_temporal"
JSONL_FILENAME = "layer2_eval_temporal_type.jsonl"
SUMMARY_FILENAME = "layer2_eval_temporal_type_summary.json"


class Layer2TemporalTypeEvaluator:
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
    def _build_user_prompt(interests):
        items = []
        for i in interests:
            temporal = i.get("temporal", "")
            if not temporal:
                continue
            evidence_list = []
            for t in i.get("topics", []):
                if isinstance(t, dict):
                    for e in t.get("evidence", []):
                        if isinstance(e, dict):
                            evidence_list.append(e.get("action", ""))
                        elif isinstance(e, str):
                            evidence_list.append(e)
            items.append({"interest_name": i.get("interest_name", ""), "evidence": evidence_list, "predicted_temporal": temporal})
        return json.dumps(items, ensure_ascii=False, indent=2)

    async def evaluate_user(self, user_id, date_str, interests):
        eval_interests = [i for i in interests if i.get("temporal")]
        if not eval_interests:
            return None

        user_prompt = self._build_user_prompt(eval_interests)
        messages = [{"role": "system", "content": self._prompt}, {"role": "user", "content": user_prompt}]

        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client, model=self._model, messages=messages,
                max_completion_tokens=self._max_tokens, reasoning_effort=self._reasoning_effort,
                response_format=None, caller="eval_l2_temporal_type",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] temporal type eval failed: %s", user_id, date_str, exc)
            self._llm_tracker.record("scoring", succeeded=False, model=self._model_key)
            return None

        if isinstance(parsed, dict):
            for key in ("result", "results", "evaluations", "interests", "data"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            if isinstance(parsed, dict):
                lists = [v for v in parsed.values() if isinstance(v, list)]
                if len(lists) == 1:
                    parsed = lists[0]
        if not isinstance(parsed, list):
            return None

        correct = sum(1 for item in parsed if item.get("accuracy", 0) == 1)
        total = len(parsed)
        accuracy_rate = round(correct / total, 4) if total else 0.0

        result = {
            "module": MODULE_NAME, "user_id": user_id, "date_str": date_str,
            "total_interests": total, "correct": correct, "accuracy_rate": accuracy_rate,
            "interest_scores": parsed,
        }
        self._llm_tracker.record("scoring", succeeded=True, model=self._model_key)
        jpath = self._output_root / date_str / JSONL_FILENAME
        append_jsonl(jpath, result)
        self._all_scores.append(result)
        logger.info("[%s][%s] temporal type eval: %d/%d correct (%.1f%%)", user_id, date_str, correct, total, accuracy_rate * 100)
        return result

    def write_summary(self) -> Dict[str, Any]:
        if not self._all_scores:
            return {}
        import numpy as np
        all_correct = sum(r["correct"] for r in self._all_scores)
        all_total = sum(r["total_interests"] for r in self._all_scores)
        all_rates = [r["accuracy_rate"] for r in self._all_scores]
        summary = {
            "total_evaluations": len(self._all_scores),
            "llm_reliability": self._llm_tracker.summary(),
            "global": {
                "total_interests": all_total, "total_correct": all_correct,
                "overall_accuracy": round(all_correct / all_total, 4) if all_total else 0,
                "mean_accuracy": round(float(np.mean(all_rates)), 4),
                "std_accuracy": round(float(np.std(all_rates)), 4),
            },
        }
        write_json(self._output_root / SUMMARY_FILENAME, summary)
        return summary
