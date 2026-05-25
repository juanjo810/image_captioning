"""Semantic embedding metrics for structured visual scene representations.

This module evaluates structured JSON outputs directly, instead of evaluating only
final captions. It is designed for the image -> structured semantics -> soundscape
pipeline, where scene type, detected entities and observed interactions have
separate downstream relevance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "intfloat/e5-large-v2"
DEFAULT_PROMPT_TEMPLATE = "Represent the semantic content for similarity: {text}"


@dataclass(frozen=True)
class SemanticEmbeddingMetricResult:
    scene_embedding_similarity: float
    entity_embedding_similarity: float
    interaction_embedding_similarity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    for token in ("_", "-", "/"):
        text = text.replace(token, " ")

    return " ".join(text.split())


def canonicalize_scene(scene_json: Mapping[str, Any] | None) -> str:
    if not scene_json:
        return "The scene is unknown."

    label = normalize_label(scene_json.get("label", "unknown"))
    indoor_outdoor = normalize_label(scene_json.get("indoor_outdoor", ""))

    if indoor_outdoor:
        return f"The scene is {label}. The setting is {indoor_outdoor}."

    return f"The scene is {label}."


def canonicalize_entities(entities_json: Sequence[Mapping[str, Any]] | None) -> str:
    if not entities_json:
        return "The scene contains no entities."

    labels = []

    for entity in entities_json:
        label = normalize_label(entity.get("label", ""))

        if label:
            labels.append(label)

    labels = sorted(set(labels))

    if not labels:
        return "The scene contains no entities."

    return "The scene contains " + ", ".join(labels) + "."


def canonicalize_interactions(interactions_json: Sequence[Mapping[str, Any]] | None) -> str:
    if not interactions_json:
        return "No interactions are observed."

    triplets = []

    for interaction in interactions_json:
        subject = normalize_label(interaction.get("subject") or interaction.get("subject_label") or "")
        relation = normalize_label(interaction.get("relation") or interaction.get("predicate") or interaction.get("action") or "")
        obj = normalize_label(interaction.get("object") or interaction.get("object_label") or "")

        triplet = " ".join(x for x in [subject, relation, obj] if x)

        if triplet:
            triplets.append(triplet)

    triplets = sorted(set(triplets))

    if not triplets:
        return "No interactions are observed."

    return "Observed interactions: " + "; ".join(triplets) + "."


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def apply_prompt(text: str, prompt_template: str | None = DEFAULT_PROMPT_TEMPLATE) -> str:
    if not prompt_template:
        return text

    return prompt_template.format(text=text)


def encode_pair(
    text_a: str,
    text_b: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    prompt_template: str | None = DEFAULT_PROMPT_TEMPLATE,
):
    model = load_model(model_name)

    texts = [
        apply_prompt(text_a, prompt_template),
        apply_prompt(text_b, prompt_template),
    ]

    embeddings = model.encode(texts, normalize_embeddings=True)

    return np.asarray(embeddings, dtype=np.float32)


def cosine_similarity_texts(
    text_a: str,
    text_b: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    prompt_template: str | None = DEFAULT_PROMPT_TEMPLATE,
) -> float:
    embeddings = encode_pair(
        text_a,
        text_b,
        model_name=model_name,
        prompt_template=prompt_template,
    )

    return float(np.dot(embeddings[0], embeddings[1]))


def compute_semantic_embedding_metrics(
    pred_json: Mapping[str, Any],
    gt_json: Mapping[str, Any],
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    prompt_template: str | None = DEFAULT_PROMPT_TEMPLATE,
) -> SemanticEmbeddingMetricResult:

    pred_scene = canonicalize_scene(pred_json.get("scene"))
    gt_scene = canonicalize_scene(gt_json.get("scene"))

    pred_entities = canonicalize_entities(pred_json.get("entities", []))
    gt_entities = canonicalize_entities(gt_json.get("entities", []))

    pred_interactions = canonicalize_interactions(
        pred_json.get("observed_interactions", [])
        or pred_json.get("interactions", [])
    )

    gt_interactions = canonicalize_interactions(
        gt_json.get("observed_interactions", [])
        or gt_json.get("interactions", [])
    )

    return SemanticEmbeddingMetricResult(
        scene_embedding_similarity=cosine_similarity_texts(
            pred_scene,
            gt_scene,
            model_name=model_name,
            prompt_template=prompt_template,
        ),
        entity_embedding_similarity=cosine_similarity_texts(
            pred_entities,
            gt_entities,
            model_name=model_name,
            prompt_template=prompt_template,
        ),
        interaction_embedding_similarity=cosine_similarity_texts(
            pred_interactions,
            gt_interactions,
            model_name=model_name,
            prompt_template=prompt_template,
        ),
    )
