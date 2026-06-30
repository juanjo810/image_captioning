from __future__ import annotations

"""AudioSet-inspired semantic layer for acoustic scene evaluation.

This module intentionally uses a reduced, editable subset of the AudioSet
ontology. It maps visible scene evidence to plausible acoustic semantics with
deterministic rules only; it does not claim that a sound is audible in the
image. The goal is to evaluate whether predicted structured JSON preserves the
same soundscape-relevant concepts as Visual Genome references.
"""

from collections import Counter
from typing import Any, Iterable

from scripts.evaluation.evaluate_structured_vg import prf
from scripts.evaluation.vg_utils import canonicalize, normalize_text


# Reduced AudioSet-inspired tag taxonomy. Keys are detailed tags; values are
# high-level categories used for the coarser evaluation layer.
AUDIOSET_TAG_TO_CATEGORY: dict[str, str] = {
    "speech": "human_sounds",
    "singing": "human_sounds",
    "crowd": "human_sounds",
    "footsteps": "human_sounds",
    "animal_vocalization": "animal_sounds",
    "bird_vocalization": "animal_sounds",
    "dog_bark": "animal_sounds",
    "cat_meow": "animal_sounds",
    "livestock": "animal_sounds",
    "horse": "animal_sounds",
    "music": "music",
    "musical_instrument": "music",
    "percussion": "music",
    "bell": "bells",
    "church_bell": "bells",
    "water": "water",
    "stream": "water",
    "waves": "water",
    "waterfall": "water",
    "wind": "wind_weather",
    "rain": "wind_weather",
    "thunder": "wind_weather",
    "engine": "tools_vehicles",
    "vehicle": "tools_vehicles",
    "train": "tools_vehicles",
    "boat": "tools_vehicles",
    "tools": "tools_vehicles",
    "machinery": "tools_vehicles",
    "wood_impact": "tools_vehicles",
    "market_activity": "public_activity",
    "cooking": "public_activity",
}


# Entity/object evidence. Values are detailed AudioSet-inspired tags.
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
) -> None:
    for tag in tags:
        if tag in AUDIOSET_TAG_TO_CATEGORY:
            counts[tag] += max(1, amount)


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


def categories_for_tags(tags: set[str]) -> set[str]:
    return {
        AUDIOSET_TAG_TO_CATEGORY[tag]
        for tag in tags
        if tag in AUDIOSET_TAG_TO_CATEGORY
    }


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
) -> dict[str, int]:
    """Infer AudioSet-inspired detailed tag counts from predicted CORE JSON."""

    counts: Counter[str] = Counter()
    core = pred_json.get("core", {})

    scene_label = core.get("scene", {}).get("label", "")
    _add_tags(counts, tags_for_scene(scene_label))

    for entity in core.get("entities", []):
        label = canonicalize(entity.get("label", ""), object_alias)
        if not label:
            continue

        try:
            count_estimate = int(entity.get("count_estimate", 1))
        except (TypeError, ValueError):
            count_estimate = 1

        _add_tags(counts, tags_for_label(label), amount=count_estimate)

    entity_by_id = _prediction_entity_by_id(pred_json, object_alias)

    for interaction in core.get("observed_interactions", []):
        verb = canonicalize(interaction.get("verb", ""), relationship_alias)
        _add_tags(counts, tags_for_predicate(verb))

        for endpoint_key in ("subject_id", "object_id"):
            endpoint_label = entity_by_id.get(interaction.get(endpoint_key))
            if endpoint_label:
                _add_tags(counts, tags_for_label(endpoint_label))

    return dict(counts)


def reference_audioset_tag_counts(
    gt_objects: set[str],
    gt_relationships: set[tuple[str, str, str]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
) -> dict[str, int]:
    """Infer AudioSet-inspired detailed tag counts from Visual Genome refs."""

    counts: Counter[str] = Counter()

    for label in gt_objects:
        canonical = canonicalize(label, object_alias)
        _add_tags(counts, tags_for_label(canonical))

    for subject, predicate, object_ in gt_relationships:
        canonical_subject = canonicalize(subject, object_alias)
        canonical_object = canonicalize(object_, object_alias)
        canonical_predicate = canonicalize(predicate, relationship_alias)

        _add_tags(counts, tags_for_predicate(canonical_predicate))
        _add_tags(counts, tags_for_label(canonical_subject))
        _add_tags(counts, tags_for_label(canonical_object))

    return dict(counts)


def compute_audioset_metrics(
    pred_json: dict[str, Any],
    gt_objects: set[str],
    gt_relationships: set[tuple[str, str, str]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
) -> dict[str, Any]:
    """Compute detailed-tag and high-level-category PRF metrics."""

    pred_counts = prediction_audioset_tag_counts(
        pred_json=pred_json,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )
    ref_counts = reference_audioset_tag_counts(
        gt_objects=gt_objects,
        gt_relationships=gt_relationships,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )

    pred_tags = set(pred_counts)
    ref_tags = set(ref_counts)
    tag_p, tag_r, tag_f1 = prf(pred_tags, ref_tags)

    pred_categories = categories_for_tags(pred_tags)
    ref_categories = categories_for_tags(ref_tags)
    category_p, category_r, category_f1 = prf(pred_categories, ref_categories)

    return {
        "audioset_tag_precision": tag_p,
        "audioset_tag_recall": tag_r,
        "audioset_tag_f1": tag_f1,
        "audioset_category_precision": category_p,
        "audioset_category_recall": category_r,
        "audioset_category_f1": category_f1,
        "n_pred_audioset_tags": len(pred_tags),
        "n_gt_audioset_tags": len(ref_tags),
        "n_pred_audioset_categories": len(pred_categories),
        "n_gt_audioset_categories": len(ref_categories),
        "pred_audioset_tags": "|".join(sorted(pred_tags)),
        "gt_audioset_tags": "|".join(sorted(ref_tags)),
        "pred_audioset_categories": "|".join(sorted(pred_categories)),
        "gt_audioset_categories": "|".join(sorted(ref_categories)),
        "pred_audioset_tag_counts": "|".join(
            f"{tag}:{pred_counts[tag]}" for tag in sorted(pred_counts)
        ),
        "gt_audioset_tag_counts": "|".join(
            f"{tag}:{ref_counts[tag]}" for tag in sorted(ref_counts)
        ),
    }
