"""Fusion utilities for HOI -> CORE interaction conversion."""

from __future__ import annotations

from src.schemas import ObservedInteraction
from src.workflow_b.hoi_utils import match_bbox_to_entity


def build_observed_interactions(
    raw_hois,
    entities_extended,
):
    interactions = []

    for hoi in raw_hois:

        subject = match_bbox_to_entity(
            hoi.human_bbox,
            entities_extended,
            category="human",
        )

        object_ = match_bbox_to_entity(
            hoi.object_bbox,
            entities_extended,
        )

        if subject is None or object_ is None:
            continue

        interactions.append(
            ObservedInteraction(
                subject_id=subject["id"],
                verb=hoi.verb,
                object_id=object_["id"],
                confidence=hoi.confidence,
            )
        )

    return interactions
