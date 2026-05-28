from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

CAPTION_SPATIAL_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
}


def build_importance_by_label(
    extended_json: dict[str, Any] | None,
) -> dict[str, float]:
    if not extended_json:
        return {}

    importance: dict[str, float] = {}

    for entity in extended_json.get("entities_extended", []):
        label = entity.get("label", "")
        score = float(entity.get("semantic_importance", 0.0))

        if label:
            importance[label] = max(importance.get(label, 0.0), score)

    return importance


def rank_core_entities(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None,
    max_entities: int = 8,
) -> list[dict[str, Any]]:
    importance = build_importance_by_label(extended_json)

    entities = core_json.get("entities", [])

    ranked = sorted(
        entities,
        key=lambda e: (
            importance.get(e.get("label", ""), 0.0),
            float(e.get("confidence", 0.0)),
            int(e.get("count_estimate", 1)),
        ),
        reverse=True,
    )

    return ranked[:max_entities]


def select_caption_relevant_spatial_relations(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None,
    max_relations: int = 2,
) -> list[dict[str, Any]]:
    entities = {
        e.get("id"): e
        for e in core_json.get("entities", [])
    }

    importance_by_label = build_importance_by_label(extended_json)

    scored = []

    for rel in core_json.get("spatial_relations", []):
        relation = rel.get("relation")

        if relation not in CAPTION_SPATIAL_RELATIONS:
            continue

        subject = entities.get(rel.get("subject_id"))
        obj = entities.get(rel.get("object_id"))

        if subject is None or obj is None:
            continue

        if (
            subject.get("category") in {"object", "structure"}
            and obj.get("category") in {"object", "structure"}
        ):
            continue

        subject_importance = importance_by_label.get(
            subject.get("label", ""),
            float(subject.get("confidence", 0.0)),
        )
        object_importance = importance_by_label.get(
            obj.get("label", ""),
            float(obj.get("confidence", 0.0)),
        )

        if max(subject_importance, object_importance) < 0.45:
            continue

        score = (
            0.45 * float(rel.get("confidence", 0.0))
            + 0.35 * subject_importance
            + 0.20 * object_importance
        )

        cleaned = {
            "subject_id": rel.get("subject_id"),
            "subject_label": subject.get("label"),
            "relation": relation,
            "object_id": rel.get("object_id"),
            "object_label": obj.get("label"),
            "confidence": rel.get("confidence"),
        }

        scored.append((score, cleaned))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:max_relations]]


def build_caption_payload(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "scene": core_json.get("scene", {}),
        "entities": rank_core_entities(
            core_json=core_json,
            extended_json=extended_json,
            max_entities=8,
        ),
        "observed_interactions": core_json.get("observed_interactions", []),
        "spatial_relations": select_caption_relevant_spatial_relations(
            core_json=core_json,
            extended_json=extended_json,
            max_relations=2,
        ),
        "environment": core_json.get("environment", {}),
    }


def build_llm_caption_prompt(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
) -> str:
    payload = build_caption_payload(
        core_json=core_json,
        extended_json=extended_json,
    )

    return (
        "You are given a structured JSON representation extracted from an image.\n"
        "Write one grounded natural image caption using only the information present in the JSON.\n\n"
        "Rules:\n"
        "- Do not add objects, people, animals, locations, actions, or attributes not present in the JSON.\n"
        "- Do not infer geographic locations.\n"
        "- Do not invent actions not present in observed_interactions.\n"
        "- Prefer observed_interactions when available.\n"
        "- Use spatial_relations only when they improve naturalness.\n"
        "- Mention the scene when useful.\n"
        "- Include the most important entities and contextual elements.\n"
        "- The caption should contain enough semantic detail for CLIPScore and SPICE evaluation.\n"
        "- Keep the description grounded and factual, not poetic.\n"
        "- Avoid storytelling or speculation.\n"
        "- Do not mention technical metadata or JSON structure.\n"
        "- Return only the caption.\n\n"
        "JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def clean_caption(caption: str) -> str:
    caption = caption.strip()
    caption = caption.replace("<end_of_turn>", "").strip()
    caption = caption.strip('"').strip("'")

    if "\n" in caption:
        caption = caption.splitlines()[0].strip()

    if caption and not caption.endswith("."):
        caption += "."

    return caption or "A visual scene."
