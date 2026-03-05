import json
import re
from typing import Any


ALLOWED_TRANSFORMS = {
    "normalize",
    "standardize",
    "handle_missing",
    "one_hot_encode",
    "drop",
}

ALLOWED_CHARTS = {"scatter", "histogram", "box", "bar", "line", "heatmap"}


def parse_json_response(text: str, fallback: Any):
    if not text:
        return fallback

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Handle common LLM format: ```json ... ```
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            snippet = fenced.group(1).strip()
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass

        # Try to parse the first JSON object/array found in free text.
        start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx != -1]
        if start_candidates:
            start = min(start_candidates)
            snippet = text[start:].strip()
            for end in range(len(snippet), 1, -1):
                piece = snippet[:end]
                try:
                    return json.loads(piece)
                except json.JSONDecodeError:
                    continue
        return fallback


def validate_feature_plan(payload: Any, valid_columns: set[str]) -> dict[str, list[str]]:
    if not isinstance(payload, dict):
        return {}

    clean: dict[str, list[str]] = {}
    for column, actions in payload.items():
        if column not in valid_columns:
            continue

        if isinstance(actions, str):
            normalized_input = [actions]
        elif isinstance(actions, list):
            normalized_input = actions
        else:
            continue

        normalized_actions = []
        for action in normalized_input:
            if not isinstance(action, str):
                continue
            key = action.strip().lower()
            if key in ALLOWED_TRANSFORMS:
                normalized_actions.append(key)
        if normalized_actions:
            clean[column] = normalized_actions
    return clean


def validate_visualization_plan(payload: Any, valid_columns: set[str]) -> list[dict]:
    if not isinstance(payload, list):
        return []

    clean = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        chart_type = str(item.get("type", "")).lower().strip()
        x_col = item.get("x")
        y_col = item.get("y")

        if chart_type not in ALLOWED_CHARTS:
            continue
        if x_col is not None and x_col not in valid_columns:
            continue
        if y_col is not None and y_col not in valid_columns:
            continue

        # Type-specific field requirements to avoid runtime chart failures.
        if chart_type in {"scatter", "bar", "line"} and (x_col is None or y_col is None):
            continue
        if chart_type in {"histogram", "box"} and x_col is None:
            continue

        clean.append({"type": chart_type, "x": x_col, "y": y_col})
    return clean
