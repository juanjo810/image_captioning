from src.geometry import entity_geometry, compute_global_geometry, bbox_iou
from src.schemas import (
    CoreJSON, ExtendedJSON,
    Scene, Entity, ObservedInteraction, Environment
)
from src.captioning import build_caption


CATEGORY_MAP = {
    "person": "human",
    "man": "human",
    "woman": "human",
    "child": "human",
    "horse": "animal",
    "dog": "animal",
    "cat": "animal",
    "tree": "vegetation",
    "plant": "vegetation",
    "building": "structure",
    "house": "structure",
    "cart": "vehicle",
    "car": "vehicle",
    "bicycle": "vehicle",
    "basket": "tool",
    "tool": "tool",
    "bread": "food",
    "food": "food",
}


def category_from_label(label: str) -> str:
    return CATEGORY_MAP.get(label.strip().lower().replace("_", " "), "object")


def match_bbox(bbox, entities_ext):
    best_id = None
    best_iou = 0.0

    for e in entities_ext:
        iou = bbox_iou(bbox, e["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = e["id"]

    return best_id if best_iou > 0.2 else None


def strip_extended_schema_fields(entity_geom: dict) -> dict:
    """Keep only fields allowed by EntityExtended.

    Geometry internally keeps label/confidence because fusion needs them, but the
    EXTENDED JSON schema intentionally stores only geometry-related fields.
    """
    allowed = {
        "id",
        "bbox",
        "bbox_area_ratio",
        "relative_size",
        "position_coarse",
        "is_central",
        "salience_score",
        "source",
    }
    return {key: value for key, value in entity_geom.items() if key in allowed}


def build_from_modules(
    image_id,
    width,
    height,
    detections,
    scene_label,
    scene_conf,
    hoi
):
    entity_records = []

    for i, det in enumerate(detections):
        rec = entity_geometry(det, f"e{i+1}", width, height)
        rec["category"] = category_from_label(rec["label"])
        entity_records.append(rec)

    global_geom = compute_global_geometry(entity_records)

    entities = [
        Entity(
            id=e["id"],
            label=e["label"],
            category=category_from_label(e["label"]),
            confidence=e["confidence"]
        )
        for e in entity_records
    ]

    interactions = []
    for h in hoi:
        sid = match_bbox(h["human_bbox"], entity_records)
        oid = match_bbox(h["object_bbox"], entity_records)

        if sid and oid:
            interactions.append(
                ObservedInteraction(
                    subject_id=sid,
                    verb=h["verb"],
                    object_id=oid,
                    confidence=h["confidence"]
                )
            )

    human_count = global_geom["category_counts"].get("human", 0)

    crowd_level = (
        "empty" if human_count == 0 else
        "sparse" if human_count <= 2 else
        "moderate"
    )
    env = Environment(
        crowd_level=crowd_level,
        activity_level="low" if not interactions else "medium",
        lighting="unknown"
    )

    caption = build_caption(scene_label, entities, interactions)

    core = CoreJSON(
        image_id=image_id,
        scene=Scene(label=scene_label, confidence=scene_conf),
        entities=entities,
        observed_interactions=interactions,
        environment=env,
        caption=caption
    )

    extended = ExtendedJSON(
        entities_extended=[strip_extended_schema_fields(e) for e in entity_records],
        global_geometry=global_geom
    )

    return core, extended
