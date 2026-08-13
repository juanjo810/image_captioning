from __future__ import annotations

from typing import Any

from src.schemas import Entity, ObservedInteraction, AudioSetCoreNode, AudioSetScene

_AUDIOSET_SCENE_PHRASES = {
    "Inside, small room": "An indoor",
    "Outside, urban or manmade": "An outdoor urban",
    "Outside, rural or natural": "An outdoor rural",
}


def _audioset_node_phrase(audioset_name: str, instance_count: int | None) -> str:
    """Verbalize one AudioSet node, singular/plural when the instance count is known.

    ``instance_count`` is only meaningful for single-evidence-group rules (e.g. "Dog"
    triggered purely by detected dogs) -- there it is the number of matched detections.
    For composite rules (e.g. "Walk, footsteps" needs man + pedestrian + street at once)
    there is no single object being counted, so the caller passes ``None`` and the term
    is used as-is, the same way "speech" or "footsteps" already read fine without an
    article.
    """

    primary_term = audioset_name.split(",")[0].strip()
    label = primary_term.lower()

    if instance_count is None:
        return label

    if instance_count > 1:
        return f"{label}s"

    article = "an" if label[0] in "aeiou" else "a"
    return f"{article} {label}"


def describe_spatial_relations(
    spatial_relations: list[dict],
    entities: list,
    max_relations: int = 2,
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
    """Build a richer deterministic caption from structured semantics.

    The caption remains grounded in the JSON, but it is intentionally richer than
    a one-sentence template so that caption metrics such as CLIPScore and SPICE
    receive enough visual and relational context.
    """

    entity_by_id = {e.id: e for e in entities}

    ranked_entities = _rank_entities_for_caption(
        entities=entities,
        entities_extended=entities_extended,
    )

    scene_text = _scene_phrase(scene_label, indoor_outdoor)
    sentences: list[str] = []

    if scene_text:
        sentences.append(f"The image shows {scene_text}")
    else:
        sentences.append("The image shows a visual scene")

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

    main_entities = []
    for entity in ranked_entities[:8]:
        phrase = _entity_phrase(entity)
        if phrase not in main_entities:
            main_entities.append(phrase)

    if main_entities:
        sentences.append(f"Visible elements include {_join_labels(main_entities)}")

    if interaction_sentences:
        sentences.extend(interaction_sentences[:3])

    context_entities = [
        e for e in ranked_entities
        if e.id not in used_entity_ids
        and e.category in {"object", "structure", "vegetation", "vehicle", "tool", "food"}
    ]

    context_labels = []
    for entity in context_entities:
        if entity.label not in context_labels:
            context_labels.append(entity.label)

    if context_labels:
        sentences.append(f"Additional contextual elements include {_join_labels(context_labels[:5])}")

    if spatial_relations:
        spatial_sentence = describe_spatial_relations(
            spatial_relations=spatial_relations,
            entities=entities,
            max_relations=2,
        )
        if spatial_sentence:
            sentences.append(spatial_sentence)

    return _finalize_caption(sentences)

# "An outdoor urban scene where vehicle engine noise, footsteps and speech would plausibly be heard."
def build_audioset_caption(
    scene: AudioSetScene,
    nodes: list[AudioSetCoreNode],
    node_instance_counts: dict[str, int | None] | None = None,
) -> str:
    scene_phrase = _AUDIOSET_SCENE_PHRASES.get(scene.audioset_name, "A")

    if not nodes:
        return _finalize_caption([f"{scene_phrase} scene"])

    node_instance_counts = node_instance_counts or {}
    ranked_nodes = sorted(nodes, key=lambda n: n.confidence, reverse=True)
    sound_labels = [
        _audioset_node_phrase(node.audioset_name, node_instance_counts.get(node.node_id))
        for node in ranked_nodes
    ]

    sentence = (
        f"{scene_phrase} scene where {_join_labels(sound_labels)} "
        "would plausibly be heard"
    )

    return _finalize_caption([sentence])


def _finalize_caption(sentences: list[str]) -> str:
    cleaned = []

    for sentence in sentences:
        sentence = sentence.strip().rstrip(".")
        if sentence:
            cleaned.append(sentence + ".")

    return " ".join(cleaned) or "A visual scene."


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


def _clean_caption(caption: str) -> str:
    caption = caption.strip().strip('"').strip("'")

    if "\n" in caption:
        caption = caption.splitlines()[0].strip()

    if caption and not caption.endswith("."):
        caption += "."

    return caption or "A visual scene."
