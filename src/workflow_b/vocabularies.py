from __future__ import annotations
from src.workflow_b.constants import(
    SCENE_GROUPS,
    INDOOR_SCENE_GROUPS,
    OUTDOOR_SCENE_GROUPS,
    UNIVERSAL_GROUNDING_PROMPT_BATCHES,
    SCENE_EXPANSION_VOCABS
)
from src.workflow_b.audioset_vocab import audioset_prompt_batches
from src.workflow_b.places365_mapping import PLACES365_TO_SCENE_GROUP



def normalize_scene_label(label: str) -> str:
    return label.strip().lower().replace("_", " ").replace("/", " ")



def scene_groups_for_label(scene_label: str) -> list[str]:
    normalized = normalize_scene_label(scene_label)

    # -------------------------------------------------------------
    # First try the exhaustive Places365 mapping.
    # -------------------------------------------------------------
    mapped_group = PLACES365_TO_SCENE_GROUP.get(normalized)

    if mapped_group is not None:
        return [mapped_group]

    # -------------------------------------------------------------
    # Fallback to legacy manual group matching.
    # -------------------------------------------------------------
    groups = []

    for group_name, labels in SCENE_GROUPS.items():
        if normalized in labels:
            groups.append(group_name)

    return groups



def infer_indoor_outdoor_from_scene(scene_label: str) -> str:
    normalized = normalize_scene_label(scene_label)

    if "indoor" in normalized:
        return "indoor"

    if "outdoor" in normalized:
        return "outdoor"

    groups = set(scene_groups_for_label(scene_label))

    if groups & INDOOR_SCENE_GROUPS:
        return "indoor"

    if groups & OUTDOOR_SCENE_GROUPS:
        return "outdoor"

    return "unknown"



def build_prompt_from_terms(terms: list[str]) -> str:
    return ", ".join(terms)



def iter_universal_prompt_batches():
    """Yield reproducible open-vocabulary detection batches.

    Batching avoids very long prompts and improves recall for small or frequent
    entities such as people in street scenes.
    """
    for batch_name, batch in UNIVERSAL_GROUNDING_PROMPT_BATCHES.items():
        yield {
            "name": batch_name,
            "prompt": build_prompt_from_terms(batch["terms"]),
            "box_threshold": batch["box_threshold"],
            "text_threshold": batch["text_threshold"],
        }


def iter_scene_expansion_prompt_batches(scene_label: str | None):
    """Yield scene-aware prompt batches derived from Places365 scene groups."""
    if scene_label is None:
        return

    groups = scene_groups_for_label(scene_label)

    for group in groups:
        terms = SCENE_EXPANSION_VOCABS.get(group)

        if not terms:
            continue

        yield {
            "name": f"scene_expansion:{group}",
            "prompt": build_prompt_from_terms(terms),
            "box_threshold": 0.28,
            "text_threshold": 0.22,
        }


def iter_audioset_prompt_batches():
    """Yield ontology-validated AudioSet-backed detector prompt batches."""
    for batch in audioset_prompt_batches():
        yield {
            "name": batch["name"],
            "prompt": build_prompt_from_terms(list(batch["terms"])),
            "box_threshold": batch["box_threshold"],
            "text_threshold": batch["text_threshold"],
        }


def iter_legacy_prompt_batches(scene_label: str | None = None):
    """Yield universal + optional scene-aware detector prompts."""
    yield from iter_universal_prompt_batches()

    if scene_label is not None:
        yield from iter_scene_expansion_prompt_batches(scene_label)


def iter_grounding_prompt_batches(
    scene_label: str | None = None,
    vocab_mode: str = "legacy",
):
    """Yield detector prompts for the selected Workflow B vocabulary mode."""
    if vocab_mode == "legacy":
        yield from iter_legacy_prompt_batches(scene_label)
        return

    if vocab_mode == "audioset":
        yield from iter_audioset_prompt_batches()
        return

    if vocab_mode == "hybrid":
        yield from iter_legacy_prompt_batches(scene_label)
        yield from iter_audioset_prompt_batches()
        return

    raise ValueError(f"Unsupported vocab_mode: {vocab_mode}")
