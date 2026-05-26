"""Lightweight LLM call reliability tracker for evaluation modules."""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, Optional


class LLMReliabilityTracker:
    """Track success/failure rates of LLM calls by call-point."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"attempted": 0, "succeeded": 0, "failed": 0, "model": None}
        )

    def record(self, call_point: str, succeeded: bool, model: Optional[str] = None) -> None:
        with self._lock:
            entry = self._data[call_point]
            entry["attempted"] += 1
            if succeeded:
                entry["succeeded"] += 1
            else:
                entry["failed"] += 1
            if model and entry["model"] is None:
                entry["model"] = model

    def summary(self) -> Dict[str, Any]:
        if not self._data:
            return {}
        result = {}
        for cp, entry in self._data.items():
            attempted = entry["attempted"]
            failed = entry["failed"]
            result[cp] = {
                "model": entry["model"],
                "attempted": attempted,
                "succeeded": entry["succeeded"],
                "failed": failed,
                "failure_rate": round(failed / attempted, 4) if attempted else 0.0,
            }
        return result
