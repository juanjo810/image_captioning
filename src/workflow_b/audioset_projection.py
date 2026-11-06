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
from src.captioning import build_audioset_caption, build_audioset_acoustic_caption
from src.schemas import (
    AudioSetCoreJSON,
    AudioSetCoreNode,
    AudioSetExtendedJSON,
    AudioSetGrounding,
    Scene,
    GlobalGeometry,
)
from src.workflow_b.vocabularies import infer_indoor_outdoor_from_scene


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
) -> list[tuple[AudioSetCoreNode, list[AudioSetGrounding], int | None]]:
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
        node_id = f"n{count}"
        node = AudioSetCoreNode(
            node_id=node_id,
            audioset_id=rule.id,
            audioset_name=rule.name,
            node_type="visible_source",
            evidence=f"Detected visual source: {', '.join(sorted({d.label for d in firing_detections}))}",
            visual_evidence_terms=sorted({normalize_text(d.label) for d in firing_detections}),
            confidence=confidence,
        )
        groundings = [
            AudioSetGrounding(
                node_id=node_id,
                visual_label_raw=det.label,
                bbox=det.bbox,
                source=det.source,
                detector_confidence=det.confidence,
            )
            for det in firing_detections
        ]
        results.append((node, groundings, instance_count))

    return results


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

    audioset_scene = Scene(
        label=scene_label,
        indoor_outdoor=infer_indoor_outdoor_from_scene(scene_label),
        confidence=scene_confidence
    )

    # Every filtered detection's normalized label, not just the ones that
    # fired a leaf rule -- an object with no associated sound rule is still a
    # perfectly valid VG object, and leaving it out of visual_terms (and thus
    # out of the caption) was recall given away for free in CIDEr/SPICE/CHAIR.
    visual_terms = list(dict.fromkeys(normalized_detection_labels.values()))

    caption = build_audioset_caption(audioset_scene, visual_terms)
    acoustic_caption = build_audioset_acoustic_caption(audioset_scene, nodes, node_instance_counts)

    core = AudioSetCoreJSON(
        image_id=image_id,
        scene=audioset_scene,
        visual_terms=visual_terms,
        nodes=nodes,
        caption=caption,
        acoustic_caption=acoustic_caption,
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
        grounding_entry
        for rule in matching_rules
        for grounding_entry in rule[1]
    ]

    extended = AudioSetExtendedJSON(
        grounding=grounding,
        global_geometry=global_geometry,
    )

    return core, extended

    

   
