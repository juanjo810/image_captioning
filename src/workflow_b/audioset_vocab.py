from __future__ import annotations

"""AudioSet leaf vocabulary for Workflow B open-vocabulary detectors."""

from functools import lru_cache
from typing import TypedDict

from metrics.audioset_leaf_vocab import (
    audioset_detectable_terms,
    audioset_leaf_rules,
)


class AudioSetDetectionNode(TypedDict):
    audioset_id: str
    audioset_name: str
    evidence: str
    all_of: tuple[tuple[str, ...], ...]


class AudioSetPromptBatch(TypedDict):
    name: str
    terms: tuple[str, ...]
    box_threshold: float
    text_threshold: float


_MAX_TERMS_PER_BATCH = 30


@lru_cache(maxsize=1)
def audioset_detection_nodes() -> tuple[AudioSetDetectionNode, ...]:
    """Return the shared AudioSet leaf-node rules used by Workflow B."""

    return tuple(
        {
            "audioset_id": rule.id,
            "audioset_name": rule.name,
            "evidence": rule.evidence,
            "all_of": rule.all_of,
        }
        for rule in audioset_leaf_rules()
    )


@lru_cache(maxsize=1)
def audioset_prompt_batches() -> tuple[AudioSetPromptBatch, ...]:
    """Return detector prompt batches covering every detectable leaf-rule term."""

    terms = audioset_detectable_terms()
    batches: list[AudioSetPromptBatch] = []

    for batch_index in range(0, len(terms), _MAX_TERMS_PER_BATCH):
        batch_terms = terms[batch_index:batch_index + _MAX_TERMS_PER_BATCH]
        batches.append(
            {
                "name": f"audioset_leaf_terms:{len(batches) + 1:02d}",
                "terms": batch_terms,
                "box_threshold": 0.25,
                "text_threshold": 0.20,
            }
        )

    return tuple(batches)
