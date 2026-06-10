"""Semantic embedding metrics for structured visual scene representations.

This module evaluates structured JSON outputs directly, instead of evaluating only
final captions. It is designed for the image -> structured semantics -> soundscape
pipeline, where scene type, detected entities and observed interactions have
separate downstream relevance.

The default encoder is MPNet because local experiments on scene/entity/interaction
matrices showed better discriminative separation than larger retrieval-oriented
embedding models for this structured evaluation setting.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_PROMPT_TEMPLATE: str | None = None


@dataclass(frozen=True)
class SemanticEmbeddingMetricResult:
    scene_embedding_similarity: float
    entity_embedding_similarity: float
    interaction_embedding_similarity: float
    structured_description_embedding_similarity: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def normalize_label(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    for token in ("_", "-", "/"):
        text = text.replace(token, " ")

    return " ".join(text.split())


def _join_natural(items: Sequence[str]) -> str:
    clean_items = [item for item in items if item]

    if not clean_items:
        return ""
    if len(clean_items) == 1:
        return clean_items[0]

    return ", ".join(clean_items[:-1]) + f" and {clean_items[-1]}"


def canonicalize_scene(scene_json: Mapping[str, Any] | None) -> str:
    if not scene_json:
        return "unknown scene"

    label = normalize_label(scene_json.get("label", "unknown"))
    indoor_outdoor = normalize_label(scene_json.get("indoor_outdoor", ""))

    if indoor_outdoor:
        # return f"The scene is {label}. The setting is {indoor_outdoor}."
        return f"{label} {indoor_outdoor}."

    # return f"The scene is {label}."
    return f"{label}"


def canonicalize_entities(entities_json: Sequence[Mapping[str, Any]] | None) -> str:
    if not entities_json:
        return "entities: none"

    labels = []

    for entity in entities_json:
        label = normalize_label(entity.get("label", ""))

        if label:
            labels.append(label)

    labels = sorted(set(labels))

    if not labels:
        return "entities: none"

    # return "The scene contains " + ", ".join(labels) + "."
    return " ".join(labels)


def canonicalize_interactions(interactions_json: Sequence[Mapping[str, Any]] | None) -> str:
    if not interactions_json:
        return "interactions: none"

    triplets = []

    for interaction in interactions_json:
        subject = normalize_label(interaction.get("subject") or interaction.get("subject_label") or "")
        relation = normalize_label(interaction.get("relation") or interaction.get("predicate") or interaction.get("action") or "")
        obj = normalize_label(interaction.get("object") or interaction.get("object_label") or "")
        description = normalize_label(interaction.get("description", ""))

        triplet = " ".join(x for x in [subject, relation, obj] if x)

        if triplet:
            triplets.append(triplet)
        elif description:
            triplets.append(description)

    triplets = sorted(set(triplets))

    if not triplets:
        return "interactions: none"

    return "interactions: " + " ; ".join(triplets)


def build_structured_description(data: Mapping[str, Any]) -> str:
    """Build a deterministic rich description from structured JSON.

    This is not a free LLM caption. It is a canonical natural-language rendering
    of the JSON, intended to give embedding models enough context without adding
    unsupported information.
    """

    parts: list[str] = []

    scene = data.get("scene", {})
    scene_label = normalize_label(scene.get("label", "")) if isinstance(scene, Mapping) else ""
    indoor_outdoor = normalize_label(scene.get("indoor_outdoor", "")) if isinstance(scene, Mapping) else ""

    if scene_label:
        if indoor_outdoor:
            parts.append(f"The image shows a {scene_label} scene in an {indoor_outdoor} setting.")
        else:
            parts.append(f"The image shows a {scene_label} scene.")

    entities = data.get("entities", [])
    entity_labels = []

    if isinstance(entities, Sequence):
        for entity in entities:
            if isinstance(entity, Mapping):
                label = normalize_label(entity.get("label", ""))
                if label:
                    entity_labels.append(label)

    entity_labels = sorted(set(entity_labels))

    if entity_labels:
        parts.append(f"Visible entities include {_join_natural(entity_labels)}.")

    interactions = data.get("observed_interactions", []) or data.get("interactions", [])
    interaction_texts = []

    if isinstance(interactions, Sequence):
        for interaction in interactions:
            if not isinstance(interaction, Mapping):
                continue

            subject = normalize_label(interaction.get("subject") or interaction.get("subject_label") or "")
            relation = normalize_label(interaction.get("relation") or interaction.get("predicate") or interaction.get("action") or "")
            obj = normalize_label(interaction.get("object") or interaction.get("object_label") or "")
            description = normalize_label(interaction.get("description", ""))

            if subject and relation and obj:
                interaction_texts.append(f"{subject} {relation} {obj}")
            elif description:
                interaction_texts.append(description)

    interaction_texts = sorted(set(interaction_texts))

    if interaction_texts:
        parts.append(f"Observed interactions include {_join_natural(interaction_texts)}.")

    return " ".join(parts).strip() or "No structured visual content is available."


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

    pred_description = build_structured_description(pred_json)
    gt_description = build_structured_description(gt_json)

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
        structured_description_embedding_similarity=cosine_similarity_texts(
            pred_description,
            gt_description,
            model_name=model_name,
            prompt_template=prompt_template,
        ),
    )
