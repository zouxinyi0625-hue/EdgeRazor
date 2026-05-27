"""
llm_client.py — Async Azure OpenAI client for EdgeRazor evaluation.

Uses direct API key authentication (no DefaultAzureCredential needed).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from openai import AsyncAzureOpenAI, APIStatusError

logger = logging.getLogger("edgerazor.eval.llm_client")

# ---------------------------------------------------------------------------
# Default config — override via environment variables or build_client params
# ---------------------------------------------------------------------------

API_VERSION = "2024-12-01-preview"

DEFAULT_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
DEFAULT_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
DEFAULT_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-5.1")

# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        _HTTP_CLIENT = httpx.AsyncClient(
            http2=False,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _HTTP_CLIENT


def build_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 300.0,
    max_retries: int = 2,
) -> AsyncAzureOpenAI:
    """Create an authenticated async Azure OpenAI client."""
    return AsyncAzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=endpoint or DEFAULT_ENDPOINT,
        api_key=api_key or DEFAULT_API_KEY,
        max_retries=max_retries,
        timeout=timeout,
        http_client=_get_http_client(),
    )


# ---------------------------------------------------------------------------
# LLM call counter
# ---------------------------------------------------------------------------

_llm_call_count = 0


def get_llm_call_count() -> int:
    return _llm_call_count


def reset_llm_call_count() -> None:
    global _llm_call_count
    _llm_call_count = 0


# ---------------------------------------------------------------------------
# Core chat invocation
# ---------------------------------------------------------------------------

async def invoke_chat(
    client: AsyncAzureOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_completion_tokens: int,
    reasoning_effort: Optional[str] = None,
    response_format: Optional[str] = "json_object",
    caller: Optional[str] = None,
    temperature: float = 0.2,
    seed: Optional[int] = None,
    extra_retry_delay: float = 1.0,
    max_retries: int = 2,
    verbose_llm_logging: bool = True,
) -> Tuple[str, Dict[str, Any], float, int]:
    """Call the chat completions endpoint.

    Returns (response_text, usage_dict, elapsed_seconds, response_len).
    """
    started = time.perf_counter()
    kwargs: Dict[str, Any] = {
        "messages": messages,
        "model": model,
        "max_completion_tokens": max_completion_tokens,
    }

    _is_reasoning_model = any(tag in model for tag in ("o1", "o3", "o4", "gpt-5"))
    _reasoning_active = _is_reasoning_model and reasoning_effort not in (None, "none")
    _supports_temperature = not _is_reasoning_model or ("gpt-5.4" in model and not _reasoning_active)

    if _is_reasoning_model:
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if _supports_temperature:
            kwargs["temperature"] = temperature
    else:
        kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed

    if response_format:
        kwargs["response_format"] = {"type": response_format}

    _EXTRA_RETRYABLE_CODES = {499}

    import asyncio as _asyncio
    for attempt in range(max_retries + 1):
        try:
            started = time.perf_counter()
            response = await client.chat.completions.create(**kwargs)
            elapsed = time.perf_counter() - started
            break
        except APIStatusError as e:
            elapsed = time.perf_counter() - started
            if e.status_code not in _EXTRA_RETRYABLE_CODES or attempt >= max_retries:
                raise
            wait = min(extra_retry_delay * (2 ** attempt), extra_retry_delay * 8)
            logger.warning(
                "invoke_chat HTTP %d (attempt %d/%d), retrying in %ds. model=%s%s",
                e.status_code, attempt + 1, max_retries + 1, wait, model,
                f" caller={caller}" if caller else "",
            )
            await _asyncio.sleep(wait)

    global _llm_call_count
    _llm_call_count += 1

    choice = response.choices[0]
    response_text = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", "unknown")
    usage = _usage_to_dict(response)

    if verbose_llm_logging:
        logger.info(
            "LLM call done. model=%s elapsed=%.2fs tokens=%s finish_reason=%s response_len=%d%s",
            model, elapsed, usage.get("total_tokens"),
            finish_reason, len(response_text),
            f" caller={caller}" if caller else "",
        )
    return response_text, usage, elapsed, len(response_text)


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def safe_json_loads(text: str) -> Any:
    """Parse JSON from LLM output, handling markdown fences and truncation."""
    payload = (text or "").strip()
    if not payload:
        return ""
    if payload.startswith("```"):
        payload = payload.strip("`")
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            return json.loads(payload + "}")
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(payload + "]}")
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(payload + "]")
        except json.JSONDecodeError:
            raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _usage_to_dict(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage)
