from __future__ import annotations

"""Small ontology-validated AudioSet vocabulary for Workflow B detectors."""

from functools import lru_cache
from typing import TypedDict

from metrics.audioset_ontology import load_audioset_ontology
from metrics.audioset_semantics import AUDIOSET_CONCEPT_TO_NODE_NAME


class AudioSetDetectionNode(TypedDict):
    concept: str
    audioset_id: str
    audioset_name: str
    prompt_terms: tuple[str, ...]


class AudioSetPromptBatch(TypedDict):
    name: str
    terms: tuple[str, ...]
    box_threshold: float
    text_threshold: float


_PROMPT_TERMS_BY_CONCEPT: dict[str, tuple[str, ...]] = {
    "speech": ("speech", "speaking person", "talking person"),
    "singing": ("singing", "singer"),
    "crowd": ("crowd", "group of people"),
    "footsteps": ("walking person", "pedestrian"),
    "animal_vocalization": ("animal",),
    "bird_vocalization": ("bird",),
    "dog_bark": ("dog",),
    "cat_meow": ("cat",),
    "livestock": ("livestock", "farm animal"),
    "horse": ("horse",),
    "music": ("music", "musical performance"),
    "musical_instrument": ("musical instrument",),
    "percussion": ("percussion", "drum"),
    "bell": ("bell",),
    "church_bell": ("church bell", "bell tower"),
    "water": ("water",),
    "stream": ("stream", "river"),
    "waves": ("waves", "surf", "ocean"),
    "waterfall": ("waterfall",),
    "wind": ("wind", "windy scene"),
    "rain": ("rain", "rainy scene"),
    "thunder": ("storm", "thunderstorm"),
    "engine": ("engine", "motor"),
    "vehicle": ("vehicle",),
    "train": ("train",),
    "boat": ("boat", "ship"),
    "tools": ("tool", "tools"),
    "machinery": ("machine", "machinery"),
    "wood_impact": ("chop", "wood chopping", "axe"),
    "market_activity": ("crowd", "market"),
    "cooking": ("chopping food", "cooking"),
}

_AUDIOSET_BATCH_SPECS: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("audioset_human_activity", ("speech", "singing", "crowd", "footsteps"), 0.20, 0.20),
    ("audioset_animals", ("animal_vocalization", "bird_vocalization", "dog_bark", "cat_meow", "livestock", "horse"), 0.25, 0.20),
    ("audioset_music_bells", ("music", "musical_instrument", "percussion", "bell", "church_bell"), 0.25, 0.20),
    ("audioset_water_weather", ("water", "stream", "waves", "waterfall", "wind", "rain", "thunder"), 0.28, 0.22),
    ("audioset_transport_machinery", ("engine", "vehicle", "train", "boat", "tools", "machinery", "wood_impact"), 0.30, 0.25),
    ("audioset_domestic_food", ("market_activity", "cooking"), 0.28, 0.22),
)


@lru_cache(maxsize=1)
def audioset_detection_nodes() -> tuple[AudioSetDetectionNode, ...]:
    """Return detector vocabulary nodes after resolving them in ontology.json."""

    ontology = load_audioset_ontology()
    nodes: list[AudioSetDetectionNode] = []

    for concept, node_name in AUDIOSET_CONCEPT_TO_NODE_NAME.items():
        node_id = ontology.resolve_name(node_name)
        if node_id is None:
            raise ValueError(f"AudioSet node is not usable or missing: {node_name}")

        canonical_name = ontology.node_by_id[node_id].name
        prompt_terms = _PROMPT_TERMS_BY_CONCEPT.get(concept, (canonical_name,))
        nodes.append(
            {
                "concept": concept,
                "audioset_id": node_id,
                "audioset_name": canonical_name,
                "prompt_terms": prompt_terms,
            }
        )

    return tuple(nodes)


@lru_cache(maxsize=1)
def audioset_prompt_batches() -> tuple[AudioSetPromptBatch, ...]:
    """Return reproducible AudioSet-backed prompt batches for Workflow B."""

    nodes_by_concept = {
        node["concept"]: node
        for node in audioset_detection_nodes()
    }
    batches: list[AudioSetPromptBatch] = []

    for name, concepts, box_threshold, text_threshold in _AUDIOSET_BATCH_SPECS:
        terms: list[str] = []
        seen: set[str] = set()

        for concept in concepts:
            node = nodes_by_concept[concept]
            for term in node["prompt_terms"]:
                normalized = term.strip().lower().replace("_", " ")
                if normalized and normalized not in seen:
                    terms.append(normalized)
                    seen.add(normalized)

        batches.append(
            {
                "name": name,
                "terms": tuple(terms),
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
            }
        )

    return tuple(batches)
