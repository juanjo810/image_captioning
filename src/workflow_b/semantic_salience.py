from __future__ import annotations

from src.workflow_b.constants import (
    CATEGORY_IMPORTANCE_WEIGHTS,
    SCENE_ENTITY_IMPORTANCE,
)
from src.workflow_b.vocabularies import scene_groups_for_label


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def scene_relevance_for_entity(label: str, scene_label: str | None) -> float:
    if scene_label is None:
        return 0.0

    label = normalize_label(label)
    groups = scene_groups_for_label(scene_label)

    scores = []

    for group in groups:
        group_scores = SCENE_ENTITY_IMPORTANCE.get(group, {})
        scores.append(group_scores.get(label, 0.0))

    return max(scores) if scores else 0.0


def compute_semantic_importance(
    entity: dict,
    scene_label: str | None,
    interaction_entity_ids: set[str],
) -> float:
    label = normalize_label(entity.get("label", ""))
    category = entity.get("category", "other")

    salience_score = float(entity.get("salience_score", 0.0))
    category_weight = CATEGORY_IMPORTANCE_WEIGHTS.get(category, 0.30)
    scene_relevance = scene_relevance_for_entity(label, scene_label)

    interaction_bonus = 1.0 if entity.get("id") in interaction_entity_ids else 0.0

    score = (
        salience_score * 0.45
        + category_weight * 0.30
        + scene_relevance * 0.20
        + interaction_bonus * 0.05
    )

    return round(max(0.0, min(score, 1.0)), 4)