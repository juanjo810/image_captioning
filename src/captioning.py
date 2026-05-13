from __future__ import annotations

import json
import os
from typing import Any

from src.schemas import Entity, ObservedInteraction


from src.schemas import Entity, ObservedInteraction

def describe_spatial_relations(
    spatial_relations: list[dict],
    entities: list,
    max_relations: int = 1,
) -> str:
    entity_by_id = {e.id: e for e in entities}

    verbalizable = {
        "left_of": "to the left of",
        "right_of": "to the right of",
        "above": "above",
        "below": "below",
    }

    phrases = []

    for rel in sorted(
        spatial_relations,
        key=lambda r: r.get("confidence", 0.0),
        reverse=True,
    ):
        relation = rel.get("relation")

        if relation not in verbalizable:
            continue

        subj = entity_by_id.get(rel.get("subject_id"))
        obj = entity_by_id.get(rel.get("object_id"))

        if subj is None or obj is None:
            continue

        if subj.category in {"object", "structure"} and obj.category in {"object", "structure"}:
            continue

        phrases.append(
            f"The {subj.label} is {verbalizable[relation]} the {obj.label}"
        )

        if len(phrases) >= max_relations:
            break

    return ". ".join(phrases)

def _importance_by_label(
    entities_extended: list[dict] | None,
) -> dict[str, float]:
    if not entities_extended:
        return {}

    scores: dict[str, float] = {}

    for entity in entities_extended:
        label = entity.get("label", "")
        score = float(entity.get("semantic_importance", 0.0))

        if label:
            scores[label] = max(scores.get(label, 0.0), score)

    return scores


def _rank_entities_for_caption(
    entities: list[Entity],
    entities_extended: list[dict] | None = None,
) -> list[Entity]:
    importance = _importance_by_label(entities_extended)

    return sorted(
        entities,
        key=lambda e: (
            importance.get(e.label, 0.0),
            e.confidence,
            e.count_estimate,
        ),
        reverse=True,
    )

def build_caption(
    scene_label: str,
    entities: list[Entity],
    interactions: list[ObservedInteraction],
    indoor_outdoor: str | None = None,
    spatial_relations: list | None = None,
    entities_extended: list[dict] | None = None,
) -> str:
    """Deterministic but more natural template caption."""

    entity_by_id = {e.id: e for e in entities}

    ranked_entities = _rank_entities_for_caption(
        entities=entities,
        entities_extended=entities_extended,
    )

    interaction_sentences = []
    used_entity_ids = set()

    for interaction in interactions:
        subject = entity_by_id.get(interaction.subject_id)
        obj = entity_by_id.get(interaction.object_id)

        if subject is None or obj is None:
            continue

        subject_text = _entity_phrase(subject)
        object_text = _entity_phrase(obj)
        verb_text = _verb_to_caption_form(interaction.verb)

        interaction_sentences.append(
            f"{subject_text.capitalize()} {verb_text} {object_text}"
        )

        used_entity_ids.add(subject.id)
        used_entity_ids.add(obj.id)

    context_entities = [
        e for e in ranked_entities
        if e.id not in used_entity_ids
        and e.category in {"object", "structure", "vegetation", "vehicle", "tool", "food"}
    ]

    context_labels = []
    for entity in context_entities:
        if entity.label not in context_labels:
            context_labels.append(entity.label)

    scene_text = _scene_phrase(scene_label, indoor_outdoor)

    spatial_sentence = ""
    if spatial_relations:
        spatial_sentence = describe_spatial_relations(
            spatial_relations=spatial_relations,
            entities=entities,
            max_relations=1,
        )

    if interaction_sentences:
        caption = interaction_sentences[0]

        if scene_text:
            caption += f" in {scene_text}"

        if context_labels:
            caption += " with " + _join_labels(context_labels[:3])

    else:
        main_entities = [_entity_phrase(e) for e in ranked_entities[:5]]

        if main_entities:
            caption = f"{_capitalize_article(scene_text)} with {_join_labels(main_entities)}"
        else:
            caption = f"{_capitalize_article(scene_text)}"

    if spatial_sentence:
        caption += f". {spatial_sentence}"

    return caption.strip() + "."


def _entity_phrase(entity: Entity) -> str:
    label = entity.label

    if entity.count_estimate > 1:
        return f"{entity.count_estimate} {label}s"

    article = "an" if label[0].lower() in "aeiou" else "a"
    return f"{article} {label}"


def _verb_to_caption_form(verb: str) -> str:
    verb = verb.strip().lower()

    mapping = {
        "hold": "holds",
        "holding": "holds",
        "ride": "rides",
        "riding": "rides",
        "sit on": "sits on",
        "sitting on": "sits on",
        "stand next to": "stands next to",
        "standing next to": "stands next to",
        "look at": "looks at",
        "looking at": "looks at",
        "carry": "carries",
        "carrying": "carries",
        "use": "uses",
        "using": "uses",
    }

    return mapping.get(verb, verb)


def _scene_phrase(scene_label: str, indoor_outdoor: str | None = None) -> str:
    scene_label = scene_label.strip().replace("_", " ")

    if indoor_outdoor and indoor_outdoor != "unknown":
        return f"an {indoor_outdoor} {scene_label} scene"

    return f"a {scene_label} scene"


def _join_labels(labels: list[str]) -> str:
    if not labels:
        return ""

    if len(labels) == 1:
        return labels[0]

    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"

    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _capitalize_article(text: str) -> str:
    if not text:
        return "A scene"

    return text[0].upper() + text[1:]

def _clean_caption(caption: str) -> str:
    caption = caption.strip().strip('"').strip("'")

    if "\n" in caption:
        caption = caption.splitlines()[0].strip()

    if caption and not caption.endswith("."):
        caption += "."

    return caption or "A visual scene."