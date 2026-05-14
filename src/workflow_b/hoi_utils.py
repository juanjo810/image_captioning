"""Utilities for HOI fusion and entity matching."""

from __future__ import annotations


def compute_iou(box_a, box_b) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def match_bbox_to_entity(
    bbox,
    entities_extended,
    category: str | None = None,
    min_iou: float = 0.20,
):
    best_entity = None
    best_iou = 0.0

    for entity in entities_extended:

        if category is not None and entity["category"] != category:
            continue

        iou = compute_iou(bbox, entity["bbox"])

        if iou > best_iou:
            best_iou = iou
            best_entity = entity

    if best_iou < min_iou:
        return None

    return best_entity
