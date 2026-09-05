from __future__ import annotations

import json
import re
from json import JSONDecoder
from typing import Any


class WorkflowAParseError(ValueError):
    pass


CORE_KEYS = {
    "image_id",
    "scene",
    "entities",
    "observed_interactions",
    "spatial_relations",
    "environment",
    "caption",
    # audioset-core stage payloads (visual_terms/scene, nodes, caption) sent
    # one at a time by the three-call flow — only "nodes"/"visual_terms" are
    # new here, the rest overlap with the legacy keys above.
    "nodes",
    "visual_terms",
}


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


def _decode_json_objects(candidate: str) -> list[dict[str, Any]]:
    decoder = JSONDecoder()
    objects: list[dict[str, Any]] = []

    for match in re.finditer(r"{", candidate):
        fragment = candidate[match.start():].strip()
        fragment = _remove_trailing_commas(fragment)

        try:
            obj, _ = decoder.raw_decode(fragment)
        except json.JSONDecodeError:
            continue

        if isinstance(obj, dict):
            objects.append(obj)

    return objects


def _score_object(obj: dict[str, Any]) -> int:
    if isinstance(obj.get("core"), dict):
        core = obj["core"]
        return 100 + len(CORE_KEYS.intersection(core.keys()))

    return len(CORE_KEYS.intersection(obj.keys()))


def _select_best_object(objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not objects:
        return None

    scored = sorted(
        objects,
        key=lambda obj: (_score_object(obj), len(obj.keys())),
        reverse=True,
    )

    best = scored[0]

    if _score_object(best) < 1:
        return None

    return best


def extract_json_block(text: str) -> dict[str, Any]:
    all_objects: list[dict[str, Any]] = []

    for candidate in _candidate_strings(text):
        all_objects.extend(_decode_json_objects(candidate))

    obj = _select_best_object(all_objects)
    if obj is not None:
        return obj

    preview = text.strip().replace("\n", " ")[:500]
    raise WorkflowAParseError(
        "No complete Workflow A JSON object found in model output. "
        f"Output preview: {preview}"
    )
