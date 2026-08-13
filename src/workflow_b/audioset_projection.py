"""Project Workflow B detections onto AudioSet core nodes.

Builds ``AudioSetCoreJSON`` / ``AudioSetExtendedJSON`` from the detections that
``filter_detections(...)`` already produced, instead of the legacy visual
``CoreJSON`` built by ``src/fusion.py``.
"""

from __future__ import annotations

from metrics.audioset_leaf_vocab import audioset_leaf_rules
from metrics.audioset_ontology import load_audioset_ontology
from scripts.evaluation.vg_utils import normalize_text
from src.geometry import Detection, compute_global_geometry, entity_geometry
from src.captioning import build_audioset_caption
from src.schemas import (
    AudioSetCoreJSON,
    AudioSetCoreNode,
    AudioSetExtendedJSON,
    AudioSetGrounding,
    AudioSetScene,
    GlobalGeometry,
)
from src.workflow_b.vocabularies import (
    infer_indoor_outdoor_from_scene,
    scene_groups_for_label,
)

_INDOOR_AUDIOSET_NAME = "Inside, small room"
_URBAN_OUTDOOR_AUDIOSET_NAME = "Outside, urban or manmade"
_RURAL_OUTDOOR_AUDIOSET_NAME = "Outside, rural or natural"

_RURAL_OUTDOOR_SCENE_GROUPS = {
    "rural_traditional",
    "natural_outdoor",
    "coastal_water",
    "garden_park",
}
_URBAN_OUTDOOR_SCENE_GROUPS = {
    "market_public",
    "religious_heritage",
    "urban_transport",
    "sports_recreation",
    "industrial_workshop",
    "entertainment_culture",
}


def _normalize_detection_labels(
    detections: list[Detection]
) -> dict[int, str]:
    
    normalized_labels: dict[int, str] = {}
    for i, det in enumerate(detections):
        normalized_labels[i] = normalize_text(det.label)

    return normalized_labels

        
def _match_leaf_rules(
    detections: list[Detection],
    normalized_labels: dict[int, str],
) -> list[tuple[AudioSetCoreNode, list[Detection], int | None]]:
    results = []
    count = 0

    for rule in audioset_leaf_rules():
        group_matches: list[list[int]] = []
        for alternatives in rule.all_of:
            matched = [i for i, label in normalized_labels.items() if label in alternatives]
            if not matched:
                group_matches = []
                break
            group_matches.append(matched)

        if not group_matches:
            continue

        group_confidences = [max(detections[i].confidence for i in idxs) for idxs in group_matches]
        confidence = min(group_confidences)

        firing_detections = [detections[i] for idxs in group_matches for i in idxs]

        # Only a single-group rule (e.g. "Dog") counts one kind of object -- a
        # composite rule (e.g. "man" + "pedestrian" + "street") infers one event
        # from several different objects, so there is no instance count to give it.
        instance_count = len(group_matches[0]) if len(group_matches) == 1 else None

        count += 1
        node = AudioSetCoreNode(
            node_id=f"n{count}",
            audioset_id=rule.id,
            audioset_name=rule.name,
            node_type="visible_source",
            evidence=f"Detected visual source: {', '.join(sorted({d.label for d in firing_detections}))}",
            confidence=confidence,
        )
        results.append((node, firing_detections, instance_count))

    return results


def _scene_to_audioset_scene(
    scene_label: str,
    scene_confidence: float,
) -> AudioSetScene:
    indoor_outdoor = infer_indoor_outdoor_from_scene(scene_label)

    if indoor_outdoor == "indoor":
        audioset_name = _INDOOR_AUDIOSET_NAME
    else:
        groups = set(scene_groups_for_label(scene_label))
        if groups & _RURAL_OUTDOOR_SCENE_GROUPS:
            audioset_name = _RURAL_OUTDOOR_AUDIOSET_NAME
        else:
            audioset_name = _URBAN_OUTDOOR_AUDIOSET_NAME

    ontology = load_audioset_ontology()
    node_id = ontology.resolve_name(audioset_name)

    return AudioSetScene(
        audioset_id=node_id,
        audioset_name=audioset_name,
        confidence=scene_confidence,
        evidence=f"Places365 scene: {scene_label}",
    )


def project_detections_to_audioset(
    detections: list[Detection], 
    scene_label: str, 
    scene_confidence: float, 
    width: int, 
    height: int, 
    image_id: str,
) -> tuple[AudioSetCoreJSON, AudioSetExtendedJSON]:
    
    normalized_detection_labels = _normalize_detection_labels(detections)

    matching_rules = _match_leaf_rules(detections, normalized_detection_labels)    
    ontology = load_audioset_ontology()

    node_instance_counts = {rule[0].node_id: rule[2] for rule in matching_rules}

    nodes = []
    for rule in matching_rules:
        node = rule[0]
        parent_ids = sorted(ontology.parents_or_self(node.audioset_id))
        top_level_ids = sorted(ontology.top_levels(node.audioset_id))
        node = node.model_copy(
            update={"parent_ids": parent_ids, "top_level_ids": top_level_ids}
        )
        nodes.append(node)

    audioset_scene = _scene_to_audioset_scene(scene_label, scene_confidence)

    caption = build_audioset_caption(audioset_scene, nodes, node_instance_counts)

    core = AudioSetCoreJSON(
        image_id=image_id,
        scene=audioset_scene,
        nodes=nodes,
        caption=caption,
    )

    entity_records = [
        entity_geometry(det, f"e{i+1}", width, height)
        for i, det in enumerate(detections)
    ]

    geometry = compute_global_geometry(entity_records)
    global_geometry = GlobalGeometry(
        total_object_coverage=geometry.get("total_object_coverage"),
        object_density_proxy=geometry.get("object_density_proxy"),
        category_counts=geometry.get("category_counts"),
        category_coverage=geometry.get("category_coverage"),
    )

    grounding = [
        AudioSetGrounding(
            node_id=rule[0].node_id,
            visual_label_raw=det.label,
            bbox=det.bbox,
            source=det.source,
            detector_confidence=det.confidence,
        )
        for rule in matching_rules
        for det in rule[1] 
    ]
    
    extended = AudioSetExtendedJSON(
        grounding=grounding,
        global_geometry=global_geometry,
    )

    return core, extended

    

   
