from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from metrics.audioset_leaf_vocab import audioset_detectable_terms
from src.schemas import AcousticSemantics, CoreJSON, AudioSetCoreJSON
from src.workflow_b.constants import CATEGORY_MAP
from src.workflow_b.vocabularies import infer_indoor_outdoor_from_scene

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
ACOUSTIC_INFERENCE_TYPES = {
    "visible_source",
    "visible_action",
    "scene_affordance",
    "uncertain",
}
# AudioSetCoreJSON's node_type (single/three/two call_mode) is stricter than
# legacy AcousticSemantics.inference_type above: no "scene_affordance" (was
# letting a node skip visual_evidence_terms) and no "uncertain" (the weakest
# evidentiary bucket, most prone to ungrounded id/evidence pairings). Every
# audioset-core node must name a specific visible object or action.
AUDIOSET_CORE_NODE_TYPES = {
    "visible_source",
    "visible_action",
}


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


def normalize_legacy_core_payload(payload: dict[str, Any], image_id: str | None = None) -> dict[str, Any]:
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


def validate_legacy_workflow_a_output(payload: dict[str, Any], image_id: str | None = None) -> CoreJSON:
    normalized = normalize_legacy_core_payload(payload=payload, image_id=image_id)
    try:
        return CoreJSON.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Workflow A output is not compatible with CoreJSON: {exc}") from exc


def normalize_legacy_acoustic_semantics_payload(
    payload: dict[str, Any],
    allowed_audioset_nodes: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> dict[str, Any]:
    """Normalize optional AudioSet pseudo-labels without changing CORE.

    These are acoustically plausible ontology nodes inferred from visual
    evidence, not claims about sounds actually heard in the image.
    """

    by_id = {node["id"]: node["name"] for node in allowed_audioset_nodes}
    by_name = {normalize_label(node["name"], default=""): node["id"] for node in allowed_audioset_nodes}

    raw_section = payload.get("acoustic_semantics")
    if not isinstance(raw_section, dict):
        return {"nodes": []}

    raw_nodes = raw_section.get("nodes", [])
    if not isinstance(raw_nodes, list):
        return {"nodes": []}

    normalized_nodes = []
    seen_ids = set()

    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue

        node_id = str(raw_node.get("id") or "").strip()
        node_name = normalize_label(raw_node.get("name"), default="")

        if node_id not in by_id and node_name in by_name:
            node_id = by_name[node_name]

        if node_id not in by_id or node_id in seen_ids:
            continue

        evidence = str(raw_node.get("evidence") or "").strip()
        if not evidence:
            continue

        inference_type = str(raw_node.get("inference_type") or "uncertain").strip().lower()
        if inference_type not in ACOUSTIC_INFERENCE_TYPES:
            inference_type = "uncertain"

        normalized_nodes.append({
            "id": node_id,
            "name": by_id[node_id],
            "evidence": evidence,
            "confidence": normalize_confidence(raw_node.get("confidence"), default=0.5),
            "inference_type": inference_type,
        })
        seen_ids.add(node_id)

    return {"nodes": normalized_nodes}


def validate_legacy_acoustic_semantics_output(
    payload: dict[str, Any],
    allowed_audioset_nodes: list[dict[str, str]] | tuple[dict[str, str], ...],
) -> AcousticSemantics:
    normalized = normalize_legacy_acoustic_semantics_payload(
        payload=payload,
        allowed_audioset_nodes=allowed_audioset_nodes,
    )
    try:
        return AcousticSemantics.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(
            f"Workflow A acoustic_semantics output is not compatible with schema: {exc}"
        ) from exc
    

def normalize_audioset_core_payload(
        payload: dict[str, Any],
        allowed_audioset_nodes: list[dict[str, str]] | tuple[dict[str, str], ...],
        allowed_scene_labels: list[str] | tuple[str, ...] = (),
        image_id: str | None = None,
) -> dict[str, Any]:
    core = unwrap_vlm_payload(payload)

    if image_id is not None:
        core["image_id"] = image_id
    else:
        core["image_id"] = str(core.get("image_id") or "unknown")


    raw_scene = core.get("scene") if isinstance(core.get("scene"), dict) else {}
    if not raw_scene:
        raise ValueError("Malformed JSON: 'scene' section was not found or is not a valid object.")

    by_id = {node["id"]: node["name"] for node in allowed_audioset_nodes}
    by_name = {normalize_label(node["name"], default=""): node["id"] for node in allowed_audioset_nodes}

    allowed_scene_labels_normalized = {normalize_label(label, default="") for label in allowed_scene_labels}
    allowed_visual_terms = set(audioset_detectable_terms())

    # The terms the model itself declared as visible (stage 1 of the three-call
    # flow, or the "visual_terms" scratch key of the single-call prompt). When
    # present, node evidence must stay within this set -- not just the global
    # detectable-term vocabulary -- so a node can't cite an object the model
    # never actually claimed to see. If missing/empty, skip this extra check
    # rather than rejecting every node's terms against an empty set.
    declared_visual_terms_list = []
    seen_declared_terms = set()
    for term in core.get("visual_terms") or []:
        if not isinstance(term, str):
            continue
        normalized_term = normalize_label(term, default="")
        if not normalized_term or normalized_term not in allowed_visual_terms:
            continue
        if normalized_term in seen_declared_terms:
            continue
        declared_visual_terms_list.append(normalized_term)
        seen_declared_terms.add(normalized_term)
    declared_visual_terms = set(declared_visual_terms_list)

    scene_label = normalize_label(raw_scene.get("label"), default="")
    if scene_label not in allowed_scene_labels_normalized:
        raise ValueError("Scene label not in allowed Places365 labels")

    indoor_outdoor = normalize_label(raw_scene.get("indoor_outdoor"), default="unknown")
    if indoor_outdoor not in INDOOR_OUTDOOR:
        indoor_outdoor = infer_indoor_outdoor_from_scene(scene_label)

    core["scene"] = {
        "label": scene_label,
        "indoor_outdoor": indoor_outdoor,
    }


    raw_nodes = core.get("nodes", [])
    if not isinstance(raw_nodes, list):
        raw_nodes = []

    normalized_nodes = []
    seen_ids = set()

    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue

        new_id = f"n{len(normalized_nodes) + 1}"

        audioset_id = str(raw_node.get("audioset_id") or "").strip()
        node_name = normalize_label(raw_node.get("audioset_name"), default="")

        if audioset_id not in by_id and node_name in by_name:
            audioset_id = by_name[node_name]

        if audioset_id not in by_id or audioset_id in seen_ids:
            continue

        evidence = str(raw_node.get("evidence") or "").strip()
        if not evidence:
            continue

        node_type = str(raw_node.get("node_type") or "visible_source").strip().lower()
        if node_type not in AUDIOSET_CORE_NODE_TYPES:
            node_type = "visible_source"

        raw_terms = raw_node.get("visual_evidence_terms")
        visual_evidence_terms = None
        if isinstance(raw_terms, list):
            seen_terms = set()
            filtered_terms = []
            for term in raw_terms:
                normalized_term = normalize_label(term, default="")
                if normalized_term not in allowed_visual_terms:
                    continue
                if declared_visual_terms and normalized_term not in declared_visual_terms:
                    continue
                if normalized_term not in seen_terms:
                    filtered_terms.append(normalized_term)
                    seen_terms.add(normalized_term)
            visual_evidence_terms = filtered_terms or None

        # No node_type is exempt from having evidence anymore (scene_affordance
        # was the only exemption and it's gone) -- a node that ends up with no
        # valid visual_evidence_terms after filtering is dropped entirely,
        # rather than kept with empty evidence.
        if visual_evidence_terms is None:
            continue

        normalized_nodes.append({
            "node_id": new_id,
            "audioset_id": audioset_id,
            "audioset_name": by_id[audioset_id],
            "node_type": node_type,
            "evidence": evidence,
            "visual_evidence_terms": visual_evidence_terms,
        })
        seen_ids.add(audioset_id)

    core["nodes"] = normalized_nodes

    caption = str(core.get("caption") or "A visual scene.").strip()
    core["caption"] = caption if caption.endswith(".") else caption + "."

    raw_acoustic_caption = core.get("acoustic_caption")
    if isinstance(raw_acoustic_caption, str) and raw_acoustic_caption.strip():
        acoustic_caption = raw_acoustic_caption.strip()
        core["acoustic_caption"] = acoustic_caption if acoustic_caption.endswith(".") else acoustic_caption + "."
    else:
        core["acoustic_caption"] = None

    core["visual_terms"] = declared_visual_terms_list

    return core


def validate_audioset_workflow_a_output(
    payload: dict[str, Any],
    allowed_audioset_nodes: list[dict[str, str]] | tuple[dict[str, str], ...],
    allowed_scene_labels: list[str] | tuple[str, ...] = (),
    image_id: str | None = None,
) -> CoreJSON:
    normalized = normalize_audioset_core_payload(
        payload=payload,
        allowed_audioset_nodes=allowed_audioset_nodes,
        allowed_scene_labels=allowed_scene_labels,
        image_id=image_id,
    )
    try:
        return AudioSetCoreJSON.model_validate(normalized)
    except ValidationError as exc:
        raise ValueError(f"Workflow A output is not compatible with CoreJSON: {exc}") from exc

