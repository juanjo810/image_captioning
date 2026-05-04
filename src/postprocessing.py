"""Post-processing utilities for detector outputs.

The first use case is Grounding DINO, which can produce several highly
overlapping boxes for the same physical object when the prompt contains
semantically related labels such as person/man/woman or horse/donkey/cow.
"""

from __future__ import annotations

from src.geometry import Detection, bbox_iou


def deduplicate_detections(
    detections: list[Detection],
    iou_threshold: float = 0.85,
) -> list[Detection]:
    """Apply conservative label-agnostic NMS.

    Detections are sorted by confidence. A detection is removed only when it
    strongly overlaps with an already-kept detection. This is intentionally
    label-agnostic because open-vocabulary detectors often assign different
    labels to the same physical object.

    Parameters
    ----------
    detections:
        Detector outputs normalized as Detection objects.
    iou_threshold:
        IoU threshold above which a detection is treated as a duplicate.
        A high default value avoids merging nearby but distinct objects.
    """
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []

    for det in ordered:
        is_duplicate = any(
            bbox_iou(det.bbox, existing.bbox) >= iou_threshold
            for existing in kept
        )
        if not is_duplicate:
            kept.append(det)

    return kept
