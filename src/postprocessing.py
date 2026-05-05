"""Post-processing utilities for detector outputs."""

from __future__ import annotations

from src.geometry import Detection, bbox_area, bbox_iou


SEMANTIC_GROUPS = {
    "human": {"person", "man", "woman", "child"},
    "animal": {"horse", "donkey", "cow", "sheep"},
    "vehicle": {"cart", "wagon", "cart wagon"},
}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def group_for_label(label: str) -> str | None:
    label = normalize_label(label)
    for group_name, labels in SEMANTIC_GROUPS.items():
        if label in labels:
            return group_name
    return None


def containment_ratio(inner_bbox: list[float], outer_bbox: list[float]) -> float:
    """How much of inner_bbox is covered by outer_bbox."""
    x1 = max(inner_bbox[0], outer_bbox[0])
    y1 = max(inner_bbox[1], outer_bbox[1])
    x2 = min(inner_bbox[2], outer_bbox[2])
    y2 = min(inner_bbox[3], outer_bbox[3])

    inter = bbox_area([x1, y1, x2, y2])
    inner_area = bbox_area(inner_bbox)

    return 0.0 if inner_area <= 0 else inter / inner_area


def deduplicate_detections(
    detections: list[Detection],
    iou_threshold: float = 0.85,
) -> list[Detection]:
    """Conservative label-agnostic NMS."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in ordered:
        duplicate = any(
            bbox_iou(det.bbox, existing.bbox) >= iou_threshold
            for existing in kept
        )
        if not duplicate:
            kept.append(det)

    return kept


def suppress_semantic_aliases(
    detections: list[Detection],
    iou_threshold: float = 0.30,
    containment_threshold: float = 0.65,
) -> list[Detection]:
    """Suppress lower-confidence semantic aliases.

    Example:
    - person/man/woman/child over the same visual region
    - horse/donkey/cow/sheep over the same animal region
    """
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in ordered:
        det_group = group_for_label(det.label)

        should_remove = False

        for existing in kept:
            existing_group = group_for_label(existing.label)

            if det_group is None or det_group != existing_group:
                continue

            if det_group == "human":
                # For humans we only suppress almost-identical boxes.
                # This avoids deleting different people in crowded scenes.
                same_region = bbox_iou(det.bbox, existing.bbox) >= 0.75
            else:
                same_region = (
                    bbox_iou(det.bbox, existing.bbox) >= iou_threshold
                    or containment_ratio(det.bbox, existing.bbox) >= containment_threshold
                    or containment_ratio(existing.bbox, det.bbox) >= containment_threshold
    )

            if same_region:
                should_remove = True
                break

        if not should_remove:
            kept.append(det)

    return kept


def filter_detections(
    detections: list[Detection],
    min_confidence: float = 0.35,
    nms_iou_threshold: float = 0.85,
    semantic_iou_threshold: float = 0.30,
    semantic_containment_threshold: float = 0.65,
) -> list[Detection]:
    """Full post-processing pipeline for open-vocabulary detections."""
    filtered = [
        Detection(
            label=normalize_label(d.label),
            bbox=d.bbox,
            confidence=d.confidence,
            source=d.source,
        )
        for d in detections
        if d.confidence >= min_confidence
    ]

    filtered = deduplicate_detections(
        filtered,
        iou_threshold=nms_iou_threshold,
    )

    filtered = suppress_semantic_aliases(
        filtered,
        iou_threshold=semantic_iou_threshold,
        containment_threshold=semantic_containment_threshold,
    )

    return sorted(filtered, key=lambda d: d.confidence, reverse=True)