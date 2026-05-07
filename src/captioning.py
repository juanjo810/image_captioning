from __future__ import annotations

import json
import os
from typing import Any

from src.schemas import Entity, ObservedInteraction


from src.schemas import Entity, ObservedInteraction


def build_caption(
    scene_label: str,
    entities: list[Entity],
    interactions: list[ObservedInteraction],
    indoor_outdoor: str | None = None,
) -> str:
    """Deterministic but more natural template caption."""

    entity_by_id = {e.id: e for e in entities}

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
        e for e in entities
        if e.id not in used_entity_ids
        and e.category in {"object", "structure", "vegetation", "vehicle", "tool", "food"}
    ]

    context_labels = []
    for entity in context_entities:
        if entity.label not in context_labels:
            context_labels.append(entity.label)

    scene_text = _scene_phrase(scene_label, indoor_outdoor)

    if interaction_sentences:
        caption = interaction_sentences[0]

        if scene_text:
            caption += f" in {scene_text}"

        if context_labels:
            caption += " with " + _join_labels(context_labels[:3])

        return caption + "."

    main_entities = [_entity_phrase(e) for e in entities[:5]]

    if main_entities:
        return f"{_capitalize_article(scene_text)} with {_join_labels(main_entities)}."

    return f"{_capitalize_article(scene_text)}."


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


def build_llm_caption_prompt(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"core": core_json}

    if extended_json is not None:
        payload["extended"] = extended_json

    return (
        "You are given a structured JSON representation extracted from an image.\n"
        "Write one concise natural image caption using only the information present in the JSON.\n\n"
        "Rules:\n"
        "- Do not add objects, people, animals, locations, actions, or attributes not present in the JSON.\n"
        "- Do not infer geographic locations.\n"
        "- Prefer observed_interactions when available.\n"
        "- Mention the scene if useful.\n"
        "- Keep the caption natural and concise.\n"
        "- If confidence is low or information is sparse, remain generic.\n"
        "- Return only the caption, with no explanation.\n\n"
        "JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_llm_caption_from_json(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
    fallback_caption: str | None = None,
) -> str:
    """Generate a JSON-grounded LLM caption.

    If no LLM provider is configured, returns fallback_caption.
    """

    prompt = build_llm_caption_prompt(core_json, extended_json)

    provider = os.getenv("WORKFLOW_B_LLM_PROVIDER", "").strip().lower()

    if provider == "openai":
        return _call_openai_caption_llm(prompt)

    if fallback_caption is not None:
        return fallback_caption

    return core_json.get("caption", "A visual scene.")


def _call_openai_caption_llm(prompt: str) -> str:
    from openai import OpenAI

    model = os.getenv("WORKFLOW_B_OPENAI_MODEL", "gpt-4.1-mini")

    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=80,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write concise image captions from structured JSON only. "
                    "Do not invent objects, actions, places, or attributes."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    caption = response.choices[0].message.content or ""

    return _clean_caption(caption)


def _clean_caption(caption: str) -> str:
    caption = caption.strip().strip('"').strip("'")

    if "\n" in caption:
        caption = caption.splitlines()[0].strip()

    if caption and not caption.endswith("."):
        caption += "."

    return caption or "A visual scene."