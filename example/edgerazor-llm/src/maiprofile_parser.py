#!/usr/bin/env python3
"""
maiprofile_parser.py — Parse raw LLM text output into structured JSONL
matching MaiProfile's expected per-layer format.
"""

import json
import re


def safe_json_loads(text: str) -> dict | list | None:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    if not text:
        return None
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try closing brackets
        closers = ["}", "]}", "]"]
        for suffix in closers:
            try:
                return json.loads(text + suffix)
            except json.JSONDecodeError:
                continue
    return None


def parse_layer0_signal(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer0 output: list of signal annotations."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    signals = data if isinstance(data, list) else data.get("signals", [])
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer0_signal",
        "signals": signals,
    }


def parse_layer1_delta(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer1_delta output: interest groupings with topics."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    interests = data if isinstance(data, list) else data.get("interests", [])
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer1_delta",
        "interests": interests,
    }


def parse_layer1_actual(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer1_actual output: actual_activity per interest."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    interests = data if isinstance(data, list) else data.get("interests", [])
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer1_actual",
        "interests": interests,
    }


def parse_layer1_intent(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer1_intent output: inferred_intent per interest."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    interests = data if isinstance(data, list) else data.get("interests", [])
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer1_intent",
        "interests": interests,
    }


def parse_layer2_temporal(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer2_temporal output: temporal classification per interest."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    interests = data if isinstance(data, list) else data.get("interests", [])
    # Add decay if not present
    decay_map = {
        "Ephemeral": 0.842,
        "ShortTerm": 0.9503,
        "LongTerm": 0.9851,
        "Persistent": 1.0,
    }
    for item in interests:
        if "decay" not in item:
            item["decay"] = decay_map.get(item.get("temporal", "Persistent"), 1.0)
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer2_temporal",
        "interests": interests,
    }


def parse_layer3_persona(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer3_persona output: persona per interest."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    personas = data if isinstance(data, list) else data.get("interest_personas", data.get("interests", []))
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer3_persona",
        "interest_personas": personas,
    }


def parse_layer3_seasonality(user_id: str, date: str, raw_text: str) -> dict | None:
    """Parse layer3_seasonality output: seasonality per interest."""
    data = safe_json_loads(raw_text)
    if data is None:
        return None
    seasonality = data if isinstance(data, list) else data.get("interest_seasonality", data.get("interests", []))
    return {
        "user_id": user_id,
        "date": date,
        "layer": "layer3_seasonality",
        "interest_seasonality": seasonality,
    }


# Registry: layer name → parser function
LAYER_PARSERS = {
    "layer0_signal": parse_layer0_signal,
    "layer1_delta": parse_layer1_delta,
    "layer1_actual": parse_layer1_actual,
    "layer1_intent": parse_layer1_intent,
    "layer2_temporal": parse_layer2_temporal,
    "layer3_persona": parse_layer3_persona,
    "layer3_seasonality": parse_layer3_seasonality,
}

# Map curation data filenames to layer keys (supports both _test and _50k suffixes)
CURATION_FILE_TO_LAYER = {
    "curation_data_layer0_signal_test.jsonl": "layer0_signal",
    "curation_data_layer1_delta_test.jsonl": "layer1_delta",
    "curation_data_layer1_actual_test.jsonl": "layer1_actual",
    "curation_data_layer1_intent_test.jsonl": "layer1_intent",
    "curation_data_layer2_temporal_test.jsonl": "layer2_temporal",
    "curation_data_layer3_persona_test.jsonl": "layer3_persona",
    "curation_data_layer3_seasonality_test.jsonl": "layer3_seasonality",
}
