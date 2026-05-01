"""Pydantic schemas for Image -> Structured Semantic Representation.

The CORE schema is shared by Workflow A (direct VLM) and Workflow B
(modular vision pipeline). The EXTENDED schema is mainly intended for
Workflow B, where bounding boxes and deterministic geometry are available.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


EntityCategory = Literal[
    "human",
    "animal",
    "object",
    "structure",
    "vehicle",
    "vegetation",
    "tool",
    "food",
    "other",
]

IndoorOutdoor = Literal["indoor", "outdoor", "mixed", "unknown"]
CrowdLevel = Literal["empty", "sparse", "moderate", "dense", "unknown"]
ActivityLevel = Literal["low", "medium", "high", "unknown"]
LightingLevel = Literal["bright", "moderate", "dim", "unknown"]
RelativeSize = Literal["tiny", "small", "medium", "large", "dominant"]


class StrictBaseModel(BaseModel):
    """Base model that rejects unexpected fields.

    This is useful for evaluation because it prevents silent JSON drift.
    """

    model_config = ConfigDict(extra="forbid")


class Scene(StrictBaseModel):
    label: str = Field(min_length=1)
    indoor_outdoor: IndoorOutdoor = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)


class Entity(StrictBaseModel):
    id: str = Field(pattern=r"^e[0-9]+$")
    label: str = Field(min_length=1)
    category: EntityCategory
    count_estimate: int = Field(default=1, ge=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        return value.strip().lower().replace("_", " ")


class ObservedInteraction(StrictBaseModel):
    subject_id: str = Field(pattern=r"^e[0-9]+$")
    verb: str = Field(min_length=1)
    object_id: str = Field(pattern=r"^e[0-9]+$")
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("verb")
    @classmethod
    def normalize_verb(cls, value: str) -> str:
        return value.strip().lower().replace("_", " ")


class Environment(StrictBaseModel):
    crowd_level: CrowdLevel = "unknown"
    activity_level: ActivityLevel = "unknown"
    lighting: LightingLevel = "unknown"


class CoreJSON(StrictBaseModel):
    image_id: str = Field(min_length=1)
    scene: Scene
    entities: list[Entity] = Field(default_factory=list)
    observed_interactions: list[ObservedInteraction] = Field(default_factory=list)
    environment: Environment
    caption: str = Field(min_length=1)


class EntityExtended(StrictBaseModel):
    id: str = Field(pattern=r"^e[0-9]+$")
    bbox: list[float] = Field(min_length=4, max_length=4)
    bbox_area_ratio: float = Field(ge=0.0, le=1.0)
    relative_size: RelativeSize
    position_coarse: str = Field(min_length=1)
    is_central: bool
    salience_score: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        x1, y1, x2, y2 = value
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2 > x1 and y2 > y1")
        return [float(v) for v in value]


class GlobalGeometry(StrictBaseModel):
    total_object_coverage: float = Field(ge=0.0, le=1.0)
    human_count: int = Field(ge=0)
    human_coverage_ratio: float = Field(ge=0.0, le=1.0)
    object_density_proxy: float = Field(ge=0.0, le=1.0)


class ExtendedJSON(StrictBaseModel):
    entities_extended: list[EntityExtended] = Field(default_factory=list)
    global_geometry: GlobalGeometry


class FullOutput(StrictBaseModel):
    core: CoreJSON
    extended: ExtendedJSON | None = None
    metadata: dict = Field(default_factory=dict)
