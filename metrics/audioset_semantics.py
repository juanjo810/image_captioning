from __future__ import annotations

"""Ontology-driven AudioSet semantic layer for acoustic scene evaluation.

Visual Genome does not provide real AudioSet labels. This module maps visible
scene evidence to plausible acoustic semantics with deterministic rules, then
projects those pseudo-labels onto the AudioSet ontology. It does not claim that
the corresponding sound is audible in the image.
"""

from collections import Counter
from typing import Any, Iterable, Literal

from metrics.audioset_ontology import AudioSetOntology, load_audioset_ontology
from scripts.evaluation.evaluate_structured_vg import prf
from scripts.evaluation.vg_utils import canonicalize, normalize_text


AudioSetPredictionSource = Literal["core_rules", "vlm_nodes", "union"]


# Visual Genome has no AudioSet labels. These deterministic concepts are
# acoustic-semantic pseudo-labels inferred from visible objects and
# relationships, then resolved to real AudioSet ontology node IDs.
AUDIOSET_CONCEPT_TO_NODE_NAME: dict[str, str] = {
    "speech": "Speech",
    "singing": "Singing",
    "crowd": "Crowd",
    "footsteps": "Walk, footsteps",
    "animal_vocalization": "Animal",
    "bird_vocalization": "Bird vocalization, bird call, bird song",
    "dog_bark": "Bark",
    "cat_meow": "Meow",
    "livestock": "Livestock, farm animals, working animals",
    "horse": "Horse",
    "music": "Music",
    "musical_instrument": "Musical instrument",
    "percussion": "Percussion",
    "bell": "Bell",
    "church_bell": "Church bell",
    "water": "Water",
    "stream": "Stream",
    "waves": "Waves, surf",
    "waterfall": "Waterfall",
    "wind": "Wind",
    "rain": "Rain",
    "thunder": "Thunder",
    "engine": "Engine",
    "vehicle": "Vehicle",
    "train": "Train",
    "boat": "Boat, Water vehicle",
    "tools": "Tools",
    "machinery": "Mechanisms",
    "wood_impact": "Chop",
    "market_activity": "Crowd",
    "cooking": "Chopping (food)",
}


# Entity/object evidence. Values are internal acoustic concepts that must
# resolve to ontology node IDs before evaluation.
ENTITY_TO_AUDIOSET_TAGS: dict[str, set[str]] = {
    # Human presence and activity
    "person": {"speech", "footsteps"},
    "people": {"speech", "crowd", "footsteps"},
    "man": {"speech", "footsteps"},
    "woman": {"speech", "footsteps"},
    "child": {"speech", "footsteps"},
    "boy": {"speech", "footsteps"},
    "girl": {"speech", "footsteps"},
    "crowd": {"speech", "crowd", "footsteps"},
    "audience": {"crowd"},
    "vendor": {"speech", "market_activity"},
    "pedestrian": {"footsteps"},
    "worker": {"tools"},
    # Animals
    "animal": {"animal_vocalization"},
    "bird": {"bird_vocalization"},
    "duck": {"bird_vocalization"},
    "chicken": {"bird_vocalization", "livestock"},
    "dog": {"dog_bark"},
    "cat": {"cat_meow"},
    "horse": {"horse", "livestock"},
    "donkey": {"livestock"},
    "cow": {"livestock"},
    "sheep": {"livestock"},
    "goat": {"livestock"},
    "pig": {"livestock"},
    # Music and performance
    "instrument": {"musical_instrument", "music"},
    "musical instrument": {"musical_instrument", "music"},
    "guitar": {"musical_instrument", "music"},
    "piano": {"musical_instrument", "music"},
    "violin": {"musical_instrument", "music"},
    "accordion": {"musical_instrument", "music"},
    "flute": {"musical_instrument", "music"},
    "trumpet": {"musical_instrument", "music"},
    "saxophone": {"musical_instrument", "music"},
    "drum": {"percussion", "music"},
    "microphone": {"speech", "singing", "music"},
    "speaker": {"music", "speech"},
    "stage": {"music", "speech"},
    # Bells and resonant heritage cues
    "bell": {"bell"},
    "bell tower": {"bell", "church_bell"},
    "church": {"church_bell"},
    "cathedral": {"church_bell"},
    "temple": {"bell"},
    # Water and weather
    "water": {"water"},
    "river": {"water", "stream"},
    "creek": {"water", "stream"},
    "stream": {"water", "stream"},
    "lake": {"water"},
    "pond": {"water"},
    "waterfall": {"water", "waterfall"},
    "sea": {"water", "waves"},
    "ocean": {"water", "waves"},
    "wave": {"waves"},
    "beach": {"water", "waves", "wind"},
    "coast": {"water", "waves", "wind"},
    "harbor": {"water", "boat"},
    "rain": {"rain"},
    "storm": {"rain", "thunder", "wind"},
    "cloud": {"wind"},
    "tree": {"wind"},
    "forest": {"wind", "bird_vocalization"},
    # Vehicles, tools, and machinery
    "vehicle": {"vehicle", "engine"},
    "car": {"vehicle", "engine"},
    "bus": {"vehicle", "engine"},
    "truck": {"vehicle", "engine"},
    "van": {"vehicle", "engine"},
    "taxi": {"vehicle", "engine"},
    "motorcycle": {"vehicle", "engine"},
    "scooter": {"vehicle", "engine"},
    "tractor": {"vehicle", "engine", "machinery"},
    "train": {"train", "engine"},
    "tram": {"train", "engine"},
    "boat": {"boat", "engine", "water"},
    "ship": {"boat", "engine", "water"},
    "cart": {"wood_impact"},
    "wagon": {"wood_impact"},
    "bicycle": {"vehicle"},
    "tool": {"tools"},
    "farm tool": {"tools"},
    "plow": {"tools"},
    "machine": {"machinery", "engine"},
    "engine": {"engine"},
    "crane": {"machinery", "engine"},
    "forklift": {"machinery", "engine"},
    "equipment": {"tools"},
    "wood": {"wood_impact"},
    "log": {"wood_impact"},
    # Public/cultural activity cues
    "market stall": {"market_activity", "speech"},
    "stand": {"market_activity"},
    "basket": {"market_activity"},
    "crate": {"market_activity"},
    "food": {"market_activity", "cooking"},
    "bread": {"market_activity"},
    "fruit": {"market_activity"},
    "vegetable": {"market_activity"},
    "vegetables": {"market_activity"},
    "cooking pot": {"cooking"},
    "pot": {"cooking"},
}


PREDICATE_TO_AUDIOSET_TAGS: dict[str, set[str]] = {
    "talk": {"speech"},
    "talking": {"speech"},
    "speak": {"speech"},
    "speaking": {"speech"},
    "listen": {"speech"},
    "listening": {"speech"},
    "sing": {"singing", "music"},
    "singing": {"singing", "music"},
    "play": {"music"},
    "playing": {"music"},
    "perform": {"music"},
    "performing": {"music"},
    "walk": {"footsteps"},
    "walking": {"footsteps"},
    "run": {"footsteps"},
    "running": {"footsteps"},
    "ride": {"vehicle"},
    "riding": {"vehicle"},
    "drive": {"vehicle", "engine"},
    "driving": {"vehicle", "engine"},
    "operate": {"machinery"},
    "operating": {"machinery"},
    "use": {"tools"},
    "using": {"tools"},
    "cut": {"tools", "wood_impact"},
    "cutting": {"tools", "wood_impact"},
    "cook": {"cooking"},
    "cooking": {"cooking"},
    "eat": {"market_activity"},
    "eating": {"market_activity"},
    "feed": {"animal_vocalization"},
    "feeding": {"animal_vocalization"},
}


# Scene labels are weaker evidence than object labels, but they let the
# predicted JSON contribute its scene field as requested.
SCENE_KEYWORD_TO_AUDIOSET_TAGS: dict[str, set[str]] = {
    "market": {"market_activity", "speech", "crowd"},
    "bazaar": {"market_activity", "speech", "crowd"},
    "street": {"vehicle", "engine", "footsteps"},
    "road": {"vehicle", "engine"},
    "highway": {"vehicle", "engine"},
    "railway": {"train", "engine"},
    "train": {"train", "engine"},
    "harbor": {"boat", "water"},
    "marina": {"boat", "water"},
    "beach": {"waves", "water", "wind"},
    "coast": {"waves", "water", "wind"},
    "river": {"stream", "water"},
    "waterfall": {"waterfall", "water"},
    "forest": {"wind", "bird_vocalization"},
    "field": {"wind", "livestock"},
    "farm": {"livestock", "machinery"},
    "barn": {"livestock"},
    "church": {"church_bell"},
    "cathedral": {"church_bell"},
    "temple": {"bell"},
    "stage": {"music", "speech"},
    "theater": {"music", "speech"},
    "auditorium": {"music", "speech", "crowd"},
    "kitchen": {"cooking"},
    "restaurant": {"speech", "market_activity", "cooking"},
    "workshop": {"tools", "machinery"},
    "factory": {"machinery", "engine"},
}


def _add_tags(
    counts: Counter[str],
    tags: Iterable[str],
    *,
    amount: int = 1,
    ontology: AudioSetOntology,
) -> None:
    for tag in tags:
        node_id = ontology.resolve_name(AUDIOSET_CONCEPT_TO_NODE_NAME.get(tag, tag))
        if node_id:
            counts[node_id] += max(1, amount)


def tags_for_label(label: str) -> set[str]:
    """Return detailed AudioSet-inspired tags for an object/entity label."""

    label = normalize_text(label)
    tags = set(ENTITY_TO_AUDIOSET_TAGS.get(label, set()))

    # Lightweight fallbacks keep the mapping useful when aliases are imperfect.
    if "bell" in label:
        tags.add("bell")
    if "church" in label or "cathedral" in label:
        tags.add("church_bell")
    if "boat" in label or "ship" in label:
        tags.update({"boat", "water"})
    if "water" in label:
        tags.add("water")
    if "tree" in label or "forest" in label:
        tags.add("wind")

    return tags


def tags_for_predicate(predicate: str) -> set[str]:
    """Return detailed tags implied by an interaction predicate."""

    predicate = normalize_text(predicate)
    tags = set(PREDICATE_TO_AUDIOSET_TAGS.get(predicate, set()))

    if "sing" in predicate:
        tags.update({"singing", "music"})
    if "play" in predicate:
        tags.add("music")
    if "talk" in predicate or "speak" in predicate:
        tags.add("speech")
    if "drive" in predicate or "ride" in predicate:
        tags.add("vehicle")

    return tags


def tags_for_scene(scene_label: str) -> set[str]:
    """Return scene-level acoustic tags from normalized keyword matches."""

    label = normalize_text(scene_label)
    tags: set[str] = set()

    for keyword, keyword_tags in SCENE_KEYWORD_TO_AUDIOSET_TAGS.items():
        if keyword in label:
            tags.update(keyword_tags)

    return tags


def parent_nodes_for_nodes(
    node_ids: set[str],
    ontology: AudioSetOntology,
) -> set[str]:
    parent_ids: set[str] = set()
    for node_id in node_ids:
        parent_ids.update(ontology.parents_or_self(node_id))
    return parent_ids


def top_level_nodes_for_nodes(
    node_ids: set[str],
    ontology: AudioSetOntology,
) -> set[str]:
    top_level_ids: set[str] = set()
    for node_id in node_ids:
        top_level_ids.update(ontology.top_levels(node_id))
    return top_level_ids


def _node_names(node_ids: Iterable[str], ontology: AudioSetOntology) -> str:
    return "|".join(
        ontology.node_by_id[node_id].name
        for node_id in sorted(node_ids, key=lambda item: ontology.node_by_id[item].name)
    )


def _node_paths(node_ids: Iterable[str], ontology: AudioSetOntology) -> str:
    paths = []
    for node_id in sorted(node_ids, key=lambda item: ontology.node_by_id[item].name):
        node_paths = [
            " > ".join(path_names)
            for path_names in ontology.path_names_to_roots(node_id)
        ]
        paths.append(" / ".join(node_paths))
    return "||".join(paths)


def _average_best_pairwise_similarity(
    pred_nodes: set[str],
    ref_nodes: set[str],
    similarity_fn,
) -> float:
    if not pred_nodes and not ref_nodes:
        return 1.0
    if not pred_nodes or not ref_nodes:
        return 0.0

    pred_best = [
        max(similarity_fn(pred_node, ref_node) for ref_node in ref_nodes)
        for pred_node in pred_nodes
    ]
    ref_best = [
        max(similarity_fn(ref_node, pred_node) for pred_node in pred_nodes)
        for ref_node in ref_nodes
    ]
    return (sum(pred_best) + sum(ref_best)) / (len(pred_best) + len(ref_best))


def _prediction_entity_by_id(
    pred_json: dict[str, Any],
    object_alias: dict[str, str],
) -> dict[str, str]:
    entities = pred_json.get("core", {}).get("entities", [])
    return {
        entity.get("id"): canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("id") and entity.get("label")
    }


def prediction_audioset_tag_counts(
    pred_json: dict[str, Any],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
    ontology: AudioSetOntology | None = None,
) -> dict[str, int]:
    """Infer AudioSet ontology node counts from predicted CORE JSON."""

    ontology = ontology or load_audioset_ontology()
    counts: Counter[str] = Counter()
    core = pred_json.get("core", {})

    scene_label = core.get("scene", {}).get("label", "")
    _add_tags(counts, tags_for_scene(scene_label), ontology=ontology)

    for entity in core.get("entities", []):
        label = canonicalize(entity.get("label", ""), object_alias)
        if not label:
            continue

        try:
            count_estimate = int(entity.get("count_estimate", 1))
        except (TypeError, ValueError):
            count_estimate = 1

        _add_tags(
            counts,
            tags_for_label(label),
            amount=count_estimate,
            ontology=ontology,
        )

    entity_by_id = _prediction_entity_by_id(pred_json, object_alias)

    for interaction in core.get("observed_interactions", []):
        verb = canonicalize(interaction.get("verb", ""), relationship_alias)
        _add_tags(counts, tags_for_predicate(verb), ontology=ontology)

        for endpoint_key in ("subject_id", "object_id"):
            endpoint_label = entity_by_id.get(interaction.get(endpoint_key))
            if endpoint_label:
                _add_tags(counts, tags_for_label(endpoint_label), ontology=ontology)

    return dict(counts)


def vlm_audioset_tag_counts(
    pred_json: dict[str, Any],
    ontology: AudioSetOntology | None = None,
) -> dict[str, int]:
    """Read direct VLM AudioSet nodes from ``acoustic_semantics.nodes``.

    The section is optional and represents visually inferred acoustic semantics.
    Invalid, unknown, duplicated, or blacklisted ontology IDs are ignored.
    """

    ontology = ontology or load_audioset_ontology()
    raw_nodes = pred_json.get("acoustic_semantics", {}).get("nodes", [])
    if not isinstance(raw_nodes, list):
        return {}

    node_ids: set[str] = set()

    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            continue

        node_id = str(raw_node.get("id") or "").strip()
        if ontology.is_usable_label(node_id):
            node_ids.add(node_id)

    return {node_id: 1 for node_id in sorted(node_ids)}


def predicted_audioset_tag_counts(
    pred_json: dict[str, Any],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
    ontology: AudioSetOntology,
    pred_source: AudioSetPredictionSource = "core_rules",
) -> dict[str, int]:
    if pred_source == "core_rules":
        return prediction_audioset_tag_counts(
            pred_json=pred_json,
            object_alias=object_alias,
            relationship_alias=relationship_alias,
            ontology=ontology,
        )

    if pred_source == "vlm_nodes":
        return vlm_audioset_tag_counts(pred_json=pred_json, ontology=ontology)

    if pred_source == "union":
        counts: Counter[str] = Counter()
        counts.update(
            prediction_audioset_tag_counts(
                pred_json=pred_json,
                object_alias=object_alias,
                relationship_alias=relationship_alias,
                ontology=ontology,
            )
        )
        for node_id in vlm_audioset_tag_counts(
            pred_json=pred_json,
            ontology=ontology,
        ):
            counts[node_id] = max(1, counts[node_id])
        return dict(counts)

    raise ValueError(f"Unknown AudioSet prediction source: {pred_source}")


def reference_audioset_tag_counts(
    gt_objects: set[str],
    gt_relationships: set[tuple[str, str, str]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
    ontology: AudioSetOntology | None = None,
) -> dict[str, int]:
    """Infer AudioSet ontology node counts from Visual Genome pseudo-refs."""

    ontology = ontology or load_audioset_ontology()
    counts: Counter[str] = Counter()

    for label in gt_objects:
        canonical = canonicalize(label, object_alias)
        _add_tags(counts, tags_for_label(canonical), ontology=ontology)

    for subject, predicate, object_ in gt_relationships:
        canonical_subject = canonicalize(subject, object_alias)
        canonical_object = canonicalize(object_, object_alias)
        canonical_predicate = canonicalize(predicate, relationship_alias)

        _add_tags(
            counts,
            tags_for_predicate(canonical_predicate),
            ontology=ontology,
        )
        _add_tags(counts, tags_for_label(canonical_subject), ontology=ontology)
        _add_tags(counts, tags_for_label(canonical_object), ontology=ontology)

    return dict(counts)


def compute_audioset_metrics(
    pred_json: dict[str, Any],
    gt_objects: set[str],
    gt_relationships: set[tuple[str, str, str]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
    pred_source: AudioSetPredictionSource = "core_rules",
) -> dict[str, Any]:
    """Compute ontology-driven hierarchical AudioSet pseudo-reference metrics."""

    ontology = load_audioset_ontology()

    pred_counts = predicted_audioset_tag_counts(
        pred_json=pred_json,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
        ontology=ontology,
        pred_source=pred_source,
    )
    ref_counts = reference_audioset_tag_counts(
        gt_objects=gt_objects,
        gt_relationships=gt_relationships,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
        ontology=ontology,
    )

    pred_nodes = set(pred_counts)
    ref_nodes = set(ref_counts)
    node_p, node_r, node_f1 = prf(pred_nodes, ref_nodes)

    pred_parent_nodes = parent_nodes_for_nodes(pred_nodes, ontology)
    ref_parent_nodes = parent_nodes_for_nodes(ref_nodes, ontology)
    parent_p, parent_r, parent_f1 = prf(pred_parent_nodes, ref_parent_nodes)

    pred_top_nodes = top_level_nodes_for_nodes(pred_nodes, ontology)
    ref_top_nodes = top_level_nodes_for_nodes(ref_nodes, ontology)
    top_p, top_r, top_f1 = prf(pred_top_nodes, ref_top_nodes)

    lca_similarity = _average_best_pairwise_similarity(
        pred_nodes,
        ref_nodes,
        ontology.lca_similarity,
    )
    tree_distance_similarity = _average_best_pairwise_similarity(
        pred_nodes,
        ref_nodes,
        ontology.tree_distance_similarity,
    )

    return {
        "audioset_exact_node_precision": node_p,
        "audioset_exact_node_recall": node_r,
        "audioset_exact_node_f1": node_f1,
        "audioset_parent_precision": parent_p,
        "audioset_parent_recall": parent_r,
        "audioset_parent_f1": parent_f1,
        "audioset_top_level_precision": top_p,
        "audioset_top_level_recall": top_r,
        "audioset_top_level_f1": top_f1,
        "audioset_lca_similarity": lca_similarity,
        "audioset_tree_distance_similarity": tree_distance_similarity,
        "audioset_pred_source": pred_source,
        # Backward-compatible aliases retained for existing consumers.
        "audioset_tag_precision": node_p,
        "audioset_tag_recall": node_r,
        "audioset_tag_f1": node_f1,
        "audioset_category_precision": top_p,
        "audioset_category_recall": top_r,
        "audioset_category_f1": top_f1,
        "n_pred_audioset_tags": len(pred_nodes),
        "n_gt_audioset_tags": len(ref_nodes),
        "n_pred_audioset_categories": len(pred_top_nodes),
        "n_gt_audioset_categories": len(ref_top_nodes),
        "n_pred_audioset_parent_nodes": len(pred_parent_nodes),
        "n_gt_audioset_parent_nodes": len(ref_parent_nodes),
        "pred_audioset_tags": "|".join(sorted(pred_nodes)),
        "gt_audioset_tags": "|".join(sorted(ref_nodes)),
        "pred_audioset_categories": "|".join(sorted(pred_top_nodes)),
        "gt_audioset_categories": "|".join(sorted(ref_top_nodes)),
        "pred_audioset_node_names": _node_names(pred_nodes, ontology),
        "gt_audioset_node_names": _node_names(ref_nodes, ontology),
        "pred_audioset_parent_node_names": _node_names(pred_parent_nodes, ontology),
        "gt_audioset_parent_node_names": _node_names(ref_parent_nodes, ontology),
        "pred_audioset_top_level_names": _node_names(pred_top_nodes, ontology),
        "gt_audioset_top_level_names": _node_names(ref_top_nodes, ontology),
        "pred_audioset_node_paths": _node_paths(pred_nodes, ontology),
        "gt_audioset_node_paths": _node_paths(ref_nodes, ontology),
        "pred_audioset_tag_counts": "|".join(
            f"{tag}:{pred_counts[tag]}" for tag in sorted(pred_counts)
        ),
        "gt_audioset_tag_counts": "|".join(
            f"{tag}:{ref_counts[tag]}" for tag in sorted(ref_counts)
        ),
    }
