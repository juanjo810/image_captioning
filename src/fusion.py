from src.geometry import entity_geometry, compute_global_geometry, bbox_iou
from src.schemas import (
    CoreJSON, ExtendedJSON,
    Scene, Entity, ObservedInteraction, Environment
)
from src.captioning import build_caption


def match_bbox(bbox, entities_ext):
    best_id = None
    best_iou = 0.0

    for e in entities_ext:
        iou = bbox_iou(bbox, e["bbox"])
        if iou > best_iou:
            best_iou = iou
            best_id = e["id"]

    return best_id if best_iou > 0.2 else None


def build_from_modules(
    image_id,
    width,
    height,
    detections,
    scene_label,
    scene_conf,
    hoi
):
    entities_ext = []

    for i, det in enumerate(detections):
        entities_ext.append(entity_geometry(det, f"e{i+1}", width, height))

    global_geom = compute_global_geometry(entities_ext)

    entities = [
        Entity(
            id=e["id"],
            label=e["label"],
            category="object",
            confidence=e["confidence"]
        )
        for e in entities_ext
    ]

    interactions = []
    for h in hoi:
        sid = match_bbox(h["human_bbox"], entities_ext)
        oid = match_bbox(h["object_bbox"], entities_ext)

        if sid and oid:
            interactions.append(
                ObservedInteraction(
                    subject_id=sid,
                    verb=h["verb"],
                    object_id=oid,
                    confidence=h["confidence"]
                )
            )

    env = Environment(
        crowd_level="sparse" if global_geom["human_count"] <= 2 else "moderate",
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
        entities_extended=entities_ext,
        global_geometry=global_geom
    )

    return core, extended
