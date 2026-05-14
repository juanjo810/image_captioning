from __future__ import annotations

from math import sqrt

from src.schemas import SpatialRelation
from src.workflow_b.constants import (
    INVERSE_RELATIONS,
    SYMMETRIC_RELATIONS,
    SPATIAL_RELATION_ALLOWED_LABELS,
    SPATIAL_RELATION_EXCLUDED_LABELS,
)


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_width(bbox: list[float]) -> float:
    x1, _, x2, _ = bbox
    return max(0.0, x2 - x1)


def bbox_height(bbox: list[float]) -> float:
    _, y1, _, y2 = bbox
    return max(0.0, y2 - y1)


def bbox_area(bbox: list[float]) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def bbox_iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)

    if inter <= 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union

def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def is_spatial_relation_candidate(entity: dict) -> bool:
    label = normalize_label(entity.get("label", ""))

    if label in SPATIAL_RELATION_EXCLUDED_LABELS:
        return False

    if label in SPATIAL_RELATION_ALLOWED_LABELS:
        return True

    # Fallback conservador:
    # si no conocemos la etiqueta, permitimos entidades no dominantes.
    # Esto evita romper generalidad con nuevos detectores/vocabularios.
    return entity.get("relative_size", "") != "dominant"


def should_skip_pair(a: dict, b: dict) -> bool:
    return not (
        is_spatial_relation_candidate(a)
        and is_spatial_relation_candidate(b)
    )


def relation_confidence(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def build_pair_candidates(
    a: dict,
    b: dict,
    image_width: int,
    image_height: int,
    *,
    axis_threshold: float,
    overlap_threshold: float,
    min_confidence: float,
) -> list[SpatialRelation]:
    bbox_a = a.get("bbox")
    bbox_b = b.get("bbox")

    if not bbox_a or not bbox_b:
        return []

    ax, ay = bbox_center(bbox_a)
    bx, by = bbox_center(bbox_b)

    dx_norm = abs(ax - bx) / image_width
    dy_norm = abs(ay - by) / image_height
    overlap = bbox_iou(bbox_a, bbox_b)

    candidates: list[SpatialRelation] = []

    if overlap >= overlap_threshold:
        candidates.append(
            SpatialRelation(
                subject_id=a["id"],
                relation="overlapping",
                object_id=b["id"],
                confidence=relation_confidence(overlap),
            )
        )

    if dx_norm >= axis_threshold and dx_norm >= dy_norm:
        candidates.append(
            SpatialRelation(
                subject_id=a["id"],
                relation="left_of" if ax < bx else "right_of",
                object_id=b["id"],
                confidence=relation_confidence(dx_norm),
            )
        )

    if dy_norm >= axis_threshold and dy_norm > dx_norm:
        candidates.append(
            SpatialRelation(
                subject_id=a["id"],
                relation="above" if ay < by else "below",
                object_id=b["id"],
                confidence=relation_confidence(dy_norm),
            )
        )

    return [
        rel for rel in candidates
        if rel.confidence >= min_confidence
    ]


def remove_inverse_duplicates(
    relations: list[SpatialRelation],
) -> list[SpatialRelation]:
    kept: list[SpatialRelation] = []
    seen: set[tuple[str, str, str]] = set()

    for rel in sorted(relations, key=lambda r: r.confidence, reverse=True):
        key = (rel.subject_id, rel.relation, rel.object_id)

        inverse_relation = INVERSE_RELATIONS.get(rel.relation)
        inverse_key = (
            rel.object_id,
            inverse_relation,
            rel.subject_id,
        )

        if inverse_relation and inverse_key in seen:
            continue

        if rel.relation in SYMMETRIC_RELATIONS:
            symmetric_key = (
                rel.object_id,
                rel.relation,
                rel.subject_id,
            )

            if symmetric_key in seen:
                continue

        kept.append(rel)
        seen.add(key)

    return kept


def limit_relations_per_subject(
    relations: list[SpatialRelation],
    max_per_subject: int,
) -> list[SpatialRelation]:
    grouped: dict[str, list[SpatialRelation]] = {}

    for rel in relations:
        grouped.setdefault(rel.subject_id, []).append(rel)

    output: list[SpatialRelation] = []

    for subject_id, subject_relations in grouped.items():
        subject_relations = sorted(
            subject_relations,
            key=lambda r: r.confidence,
            reverse=True,
        )
        output.extend(subject_relations[:max_per_subject])

    return output


def build_spatial_relations(
    entities_extended: list[dict],
    image_width: int,
    image_height: int,
    *,
    axis_threshold: float = 0.15,
    overlap_threshold: float = 0.35,
    min_confidence: float = 0.15,
    max_relations_per_pair: int = 1,
    max_relations_per_subject: int = 3,
    min_entity_salience: float = 0.30
) -> list[SpatialRelation]:
    """Infer deterministic spatial relations from entity bounding boxes.

    This version does not exclude labels.
    It only applies geometric thresholds and duplicate suppression.
    """

    if image_width <= 0 or image_height <= 0:
        return []

    relations: list[SpatialRelation] = []

    for i, a in enumerate(entities_extended):
        for j, b in enumerate(entities_extended):
            if i == j:
                continue

            if should_skip_pair(a, b):
                continue

            if a.get("salience_score", 0.0) < min_entity_salience:
                continue

            if b.get("salience_score", 0.0) < min_entity_salience:
                continue

            candidates = build_pair_candidates(
                a=a,
                b=b,
                image_width=image_width,
                image_height=image_height,
                axis_threshold=axis_threshold,
                overlap_threshold=overlap_threshold,
                min_confidence=min_confidence,
            )

            if not candidates:
                continue

            candidates = sorted(
                candidates,
                key=lambda rel: rel.confidence,
                reverse=True,
            )

            relations.extend(candidates[:max_relations_per_pair])

    relations = remove_inverse_duplicates(relations)
    relations = limit_relations_per_subject(
        relations,
        max_per_subject=max_relations_per_subject,
    )

    return relations