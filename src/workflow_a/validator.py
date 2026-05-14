from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from src.schemas import CoreJSON
from src.workflow_b.constants import CATEGORY_MAP

ENTITY_CATEGORIES = {
    "human",
    "animal",
    "object",
    "structure",
    "vehicle",
    "vegetation",
    "tool",
    "food",
    "other",
}

INDOOR_OUTDOOR = {"indoor", "outdoor", "mixed", "unknown"}
CROWD_LEVELS = {"empty", "sparse", "moderate", "dense", "unknown"}
ACTIVITY_LEVELS = {"low", "medium", "high", "unknown"}
LIGHTING_LEVELS = {"bright", "moderate", "dim", "unknown"}
SPATIAL_RELATIONS = {"left_of", "right_of", "above", "below", "overlapping"}


def normalize_label(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    value = str(value).strip().lower().replace("_", " ")
    return value or default


def normalize_confidence(value: Any, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return max(0.0, min(confidence, 1.0))


def normalize_category(label: str, category: Any) -> str:
    category = normalize_label(category, default="")
    if category in ENTITY_CATEGORIES:
        return category
    return CATEGORY_MAP.get(label, "object")


def unwrap_vlm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept either {'caption': ..., 'core': {...}} or a direct CORE object."""
    if "core" in payload and isinstance(payload["core"], dict):
        core = deepcopy(payload["core"])
        if "caption" not in core and payload.get("caption"):
            core["caption"] = payload["caption"]
        return core
    return deepcopy(payload)


def normalize_core_payload(payload: dict[str, Any], image_id: str | None = None) -> dict[str, Any]:
    core = unwrap_vlm_payload(payload)

    if image_id is not None:
        core["image_id"] = image_id
    else:
        core["image_id"] = str(core.get("image_id") or "unknown")

    scene = core.get("scene") if isinstance(core.get("scene"), dict) else {}
    scene_label = normalize_label(scene.get("label"), default="unknown scene")
    indoor_outdoor = normalize_label(scene.get("indoor_outdoor"), default="unknown")
    if indoor_outdoor not in INDOOR_OUTDOOR:
        indoor_outdoor = "unknown"

    core["scene"] = {
        "label": scene_label,
        "indoor_outdoor": indoor_outdoor,
        "confidence": normalize_confidence(scene.get("confidence"), default=0.5),
    }

    normalized_entities = []
    original_to_new_id = {}

    for index, entity in enumerate(core.get("entities") or [], start=1):
        if not isinstance(entity, dict):
            continue

        label = normalize_label(entity.get("label"), default="object")
        new_id = f"e{len(normalized_entities) + 1}"
        old_id = str(entity.get("id") or new_id)
        original_to_new_id[old_id] = new_id

        try:
            count_estimate = int(entity.get("count_estimate", 1))
        except (TypeError, ValueError):
            count_estimate = 1

        normalized_entities.append({
            "id": new_id,
            "label": label,
            "category": normalize_category(label, entity.get("category")),
            "count_estimate": max(1, count_estimate),
            "confidence": normalize_confidence(entity.get("confidence"), default=0.5),
        })

    core["entities"] = normalized_entities
    valid_ids = {entity["id"] for entity in normalized_entities}

    normalized_interactions = []
    for interaction in core.get("observed_interactions") or []:
        if not isinstance(interaction, dict):
            continue
        subject_id = original_to_new_id.get(str(interaction.get("subject_id")), str(interaction.get("subject_id")))
        object_id = original_to_new_id.get(str(interaction.get("object_id")), str(interaction.get("object_id")))
        if subject_id not in valid_ids or object_id not in valid_ids or subject_id == object_id:
            continue
        normalized_interactions.append({
            "subject_id": subject_id,
            "verb": normalize_label(interaction.get("verb"), default="interact with"),
            "object_id": object_id,
            "confidence": normalize_confidence(interaction.get("confidence"), default=0.5),
        })
    core["observed_interactions"] = normalized_interactions

    normalized_relations = []
    for relation in core.get("spatial_relations") or []:
        if not isinstance(relation, dict):
            continue
        subject_id = original_to_new_id.get(str(relation.get("subject_id")), str(relation.get("subject_id")))
        object_id = original_to_new_id.get(str(relation.get("object_id")), str(relation.get("object_id")))
        relation_label = normalize_label(relation.get("relation"), default="")
        if subject_id not in valid_ids or object_id not in valid_ids or subject_id == object_id:
            continue
        if relation_label not in SPATIAL_RELATIONS:
            continue
        normalized_relations.append({
            "subject_id": subject_id,
            "relation": relation_label,
            "object_id": object_id,
            "confidence": normalize_confidence(relation.get("confidence"), default=0.5),
        })
    core["spatial_relations"] = normalized_relations

    environment = core.get("environment") if isinstance(core.get("environment"), dict) else {}
    crowd_level = normalize_label(environment.get("crowd_level"), default="unknown")
    activity_level = normalize_label(environment.get("activity_level"), default="unknown")
    lighting = normalize_label(environment.get("lighting"), default="unknown")

    core["environment"] = {
        "crowd_level": crowd_level if crowd_level in CROWD_LEVELS else "unknown",
        "activity_level": activity_level if activity_level in ACTIVITY_LEVELS else "unknown",
        "lighting": lighting if lighting in LIGHTING_LEVELS else "unknown",
    }

    caption = str(core.get("caption") or "A visual scene.").strip()
    core["caption"] = caption if caption.endswith(".") else caption + "."

    return core


def validate_workflow_a_output(payload: dict[str, Any], image_id: str | None = None) -> CoreJSON:
    normalized = normalize_core_payload(payload=payload, image_id=image_id)
    try:
        return CoreJSON.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Workflow A output is not compatible with CoreJSON: {exc}") from exc
