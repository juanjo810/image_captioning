"""Human-Object Interaction adapter contracts.

This module follows the same flat adapter philosophy used for Grounding DINO
and Places365. The model-specific adapter returns raw HOI triplets expressed in
image coordinates. The fusion layer later maps those boxes to project entities.

The current DummyHOIAdapter is intentionally simple: it allows testing the HOI
fusion path before integrating a fragile external HOI model such as UPT-R50.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawHOITriplet:
    """Raw output of an HOI model before entity matching."""

    human_bbox: list[float]
    object_bbox: list[float]
    verb: str
    confidence: float


class DummyHOIAdapter:
    """Minimal HOI adapter for integration tests.

    It returns no interactions by default. If handcrafted triplets are provided,
    they are returned as-is. This keeps the production pipeline conservative
    while allowing controlled tests of HOI matching and fusion.
    """

    def __init__(self, triplets: list[RawHOITriplet] | None = None):
        self.triplets = triplets or []

    def predict(self, image_path: str | Path) -> list[RawHOITriplet]:
        return self.triplets
