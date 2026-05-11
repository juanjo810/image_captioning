"""Deterministic geometric features for detected entities.

This module is independent of any specific detector. It operates on
simple dictionaries or lightweight objects describing bounding boxes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    label: str
    bbox: list[float]
    confidence: float
    source: str = "unknown"


def bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area_ratio(bbox: list[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return bbox_area(bbox) / float(width * height)


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = bbox_area([ix1, iy1, ix2, iy2])
    union = bbox_area(a) + bbox_area(b) - inter

    return 0.0 if union <= 0 else inter / union


def relative_size(ratio: float) -> str:
    if ratio < 0.01:
        return "tiny"
    if ratio < 0.05:
        return "small"
    if ratio < 0.15:
        return "medium"
    if ratio < 0.35:
        return "large"
    return "dominant"


def position_coarse(bbox: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    col = "left" if cx < width / 3 else "center" if cx < 2 * width / 3 else "right"
    row = "top" if cy < height / 3 else "middle" if cy < 2 * height / 3 else "bottom"

    return f"{col}-{row}"


def is_central(bbox: list[float], width: int, height: int) -> bool:
    return position_coarse(bbox, width, height) == "center-middle"


def salience(bbox: list[float], conf: float, width: int, height: int) -> float:
    ratio = area_ratio(bbox, width, height)
    central_bonus = 0.15 if is_central(bbox, width, height) else 0.0
    score = 0.65 * conf + 0.35 * min(ratio / 0.35, 1.0) + central_bonus
    return min(score, 1.0)

def clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox

    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]

def entity_geometry(det: Detection, entity_id: str, width: int, height: int) -> dict:
    """Compute all deterministic geometry fields for a detection."""
    bbox = clip_bbox(det.bbox, width, height)
    ratio = min(area_ratio(bbox, width, height), 1.0)

    return {
        "id": entity_id,
        "label": det.label.strip().lower().replace("_", " "),
        "bbox": [float(v) for v in bbox],
        "confidence": float(det.confidence),
        "bbox_area_ratio": float(ratio),
        "relative_size": relative_size(ratio),
        "position_coarse": position_coarse(bbox, width, height),
        "is_central": is_central(bbox, width, height),
        "salience_score": float(salience(bbox, det.confidence, width, height)),
        "source": det.source,
    }


def compute_global_geometry(entities: list[dict]) -> dict:
    counts = {}
    coverage = {}
    total_coverage = 0.0

    for e in entities:
        # ⚠️ categoría viene de fusion (no de geometry)
        cat = e.get("category", "object")

        counts[cat] = counts.get(cat, 0) + 1
        coverage[cat] = coverage.get(cat, 0.0) + e["bbox_area_ratio"]

        total_coverage += e["bbox_area_ratio"]

    return {
        "total_object_coverage": min(total_coverage, 1.0),
        "object_density_proxy": min(len(entities) / 20.0, 1.0),
        "category_counts": counts,
        "category_coverage": coverage
    }
