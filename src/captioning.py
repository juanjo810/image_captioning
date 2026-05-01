from src.schemas import Entity, ObservedInteraction


def build_caption(scene_label: str, entities: list[Entity], interactions: list[ObservedInteraction]) -> str:
    labels = [e.label for e in entities[:5]]

    if labels:
        text = f"A {scene_label} scene with " + ", ".join(labels)
    else:
        text = f"A {scene_label} scene"

    if interactions:
        verbs = sorted(set(i.verb for i in interactions))
        text += f". People are {', '.join(verbs)}"

    return text + "."
