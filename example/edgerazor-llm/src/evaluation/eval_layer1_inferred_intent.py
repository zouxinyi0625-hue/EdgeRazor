"""eval_layer1_inferred_intent.py — Evaluate inferred_intent quality (4 dimensions, 1-10 each)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.io_utils import append_jsonl, read_text, write_json
from evaluation.llm_client import AsyncAzureOpenAI, invoke_chat, safe_json_loads

logger = logging.getLogger("edgerazor.eval_inferred_intent")

SCORE_PILLARS = ["faithfulness_to_evidence", "intent_abstraction_quality", "distinctness_from_actual_activity", "specificity_actionability"]
PROMPT_PATH = Path(__file__).parent / "prompts" / "layer1_intent_eval_intent.md"
MODULE_NAME = "layer1_intent"
JSONL_FILENAME = "layer1_intent_eval_intent.jsonl"
SUMMARY_FILENAME = "layer1_intent_eval_intent_summary.json"


class Layer1InferredIntentEvaluator:
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
    def _build_user_prompt(delta_interests, actual_interests, intent_interests):
        actual_map = {i.get("interest_name", ""): i.get("actual_activity", "") for i in actual_interests if isinstance(i, dict)}
        intent_map = {i.get("interest_name", ""): i.get("inferred_intent", "") for i in intent_interests}
        interest_items = []
        for interest in delta_interests:
            if not isinstance(interest, dict):
                continue
            name = interest.get("interest_name", "")
            intent = intent_map.get(name, "")
            if not intent:
                continue
            actual = actual_map.get(name, "")
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
            interest_items.append({"interest_name": name, "actual_activity": actual, "inferred_intent": intent, "topics": topics_info})
        return f"interests:\n{json.dumps(interest_items, ensure_ascii=False, indent=2)}"

    async def evaluate_user(self, user_id, date_str, delta_interests, actual_interests, intent_interests):
        total_interests = sum(1 for i in intent_interests if isinstance(i, dict) and i.get("inferred_intent"))
        if total_interests == 0:
            return None

        user_prompt = self._build_user_prompt(delta_interests, actual_interests, intent_interests)
        messages = [{"role": "system", "content": self._prompt}, {"role": "user", "content": user_prompt}]

        try:
            response_text, usage, elapsed, resp_len = await invoke_chat(
                client=self._client, model=self._model, messages=messages,
                max_completion_tokens=self._max_tokens, reasoning_effort=self._reasoning_effort,
                response_format=None, caller="eval_l1_inferred_intent",
            )
            parsed = safe_json_loads(response_text)
        except Exception as exc:
            logger.warning("[%s][%s] eval_inferred_intent failed: %s", user_id, date_str, exc)
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

        user_avg = {}
        if valid_count > 0:
            user_avg = {p: round(pillar_sums[p] / valid_count, 4) for p in SCORE_PILLARS}
            user_avg["final_score"] = round(sum(user_avg[p] for p in SCORE_PILLARS) / len(SCORE_PILLARS), 4)

        result = {
            "module": MODULE_NAME, "user_id": user_id, "date_str": date_str,
            "num_interests_evaluated": valid_count, "num_interests_expected": total_interests,
            "interest_scores": parsed, "user_averages": user_avg,
        }
        self._llm_tracker.record("scoring", succeeded=True, model=self._model_key)
        jpath = self._output_root / date_str / JSONL_FILENAME
        append_jsonl(jpath, result)
        self._all_scores.append(result)
        logger.info("[%s][%s] eval_inferred_intent: %d/%d scored. final=%.3f", user_id, date_str, valid_count, total_interests, user_avg.get("final_score", 0))
        return result

    def write_summary(self) -> Dict[str, Any]:
        if not self._all_scores:
            return {}
        import numpy as np
        all_pillars = SCORE_PILLARS + ["final_score"]
        pillar_all: Dict[str, List[float]] = {p: [] for p in all_pillars}
        for rec in self._all_scores:
            for item in rec.get("interest_scores", []):
                scores = item.get("scores", {})
                for p in SCORE_PILLARS:
                    val = scores.get(p)
                    if val is not None:
                        pillar_all[p].append(float(val))
                interest_vals = [float(scores[p]) for p in SCORE_PILLARS if scores.get(p) is not None]
                if interest_vals:
                    pillar_all["final_score"].append(sum(interest_vals) / len(interest_vals))
        global_avg = {}
        for p in all_pillars:
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
