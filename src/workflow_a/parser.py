from __future__ import annotations

import json
import re
from json import JSONDecoder
from typing import Any


class WorkflowAParseError(ValueError):
    pass


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _candidate_strings(text: str) -> list[str]:
    text = text.strip()

    candidates: list[str] = []

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(_strip_code_fences(block) for block in fenced)

    candidates.append(_strip_code_fences(text))

    return [candidate for candidate in candidates if candidate]


def _decode_first_json_object(candidate: str) -> dict[str, Any] | None:
    decoder = JSONDecoder()

    for match in re.finditer(r"{", candidate):
        fragment = candidate[match.start():].strip()
        fragment = _remove_trailing_commas(fragment)

        try:
            obj, _ = decoder.raw_decode(fragment)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            return obj

    return None


def extract_json_block(text: str) -> dict[str, Any]:
    for candidate in _candidate_strings(text):
        obj = _decode_first_json_object(candidate)
        if obj is not None:
            return obj

    preview = text.strip().replace("\n", " ")[:500]
    raise WorkflowAParseError(
        "No valid JSON object found in model output. "
        f"Output preview: {preview}"
    )
