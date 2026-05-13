from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.evaluation.vg_utils import canonicalize, load_alias_map, load_json, normalize_text
from src.workflow_b.constants import CATEGORY_MAP, SCENE_GROUPS
from src.workflow_b.vocabularies import infer_indoor_outdoor_from_scene, scene_groups_for_label


# ---------------------------------------------------------------------
# Soundscape-oriented semantic hierarchy
# ---------------------------------------------------------------------

ACOUSTIC_ENTITY_FAMILIES: dict[str, set[str]] = {
    "human": {
        "person", "man", "woman", "child", "boy", "girl", "people", "crowd",
        "audience", "vendor", "worker", "player", "pedestrian", "rider",
    },
    "animal": {
        "animal", "horse", "donkey", "cow", "sheep", "goat", "dog", "cat", "bird",
        "zebra", "lion", "tiger", "giraffe", "elephant", "bear", "duck", "chicken",
    },
    "vehicle": {
        "vehicle", "car", "bus", "truck", "bicycle", "motorcycle", "cart", "wagon",
        "wooden cart", "boat", "ship", "train", "tram", "tractor", "wheelchair",
        "forklift", "van", "taxi", "scooter",
    },
    "water": {
        "water", "river", "lake", "pond", "creek", "waterfall", "sea", "ocean",
        "wave", "canal", "harbor", "beach", "coast",
    },
    "music_instrument": {
        "instrument", "microphone", "speaker", "guitar", "drum", "violin", "piano",
        "accordion", "flute", "trumpet", "saxophone",
    },
    "tool_machinery": {
        "tool", "farm tool", "plow", "machine", "engine", "pipe", "crane", "forklift",
        "equipment", "medical equipment", "computer", "monitor", "traffic light",
        "street lamp", "lamp", "cooking pot", "sink",
    },
    "food_market": {
        "food", "bread", "fruit", "vegetable", "vegetables", "grain", "hay", "basket",
        "crate", "market stall", "stand",
    },
    "built_context": {
        "building", "house", "church", "cathedral", "temple", "tower", "bell tower",
        "wall", "stone wall", "fence", "gate", "door", "window", "bridge", "barn",
        "stable", "corral", "animal pen", "monument", "statue", "arch", "column",
        "altar", "grave", "boathouse", "dock", "pier", "gazebo", "lifeguard tower",
        "street", "road", "sidewalk", "crosswalk", "stage", "desk", "table", "chair",
        "bench", "cabinet", "counter", "screen", "display",
    },
    "nature_context": {
        "tree", "plant", "grass", "flower", "crop", "field crop", "bush", "forest",
        "field", "rock", "sky", "mountain", "sand", "path", "trail", "lawn", "log",
    },
}

# These families are the most important for discrete sound events.
DISCRETE_SOUND_FAMILIES = {
    "human",
    "animal",
    "vehicle",
    "water",
    "music_instrument",
    "tool_machinery",
    "food_market",
}

# Background/context families are still important, but mainly for ambience.
BACKGROUND_SOUND_FAMILIES = {
    "built_context",
    "nature_context",
    "water",
}

INTERACTION_FAMILIES: dict[str, set[str]] = {
    "object_manipulation": {
        "hold", "holding", "carry", "carrying", "use", "using", "wear", "wearing",
        "touch", "touching", "grab", "grabbing",
    },
    "locomotion_transport": {
        "ride", "riding", "sit on", "sitting on", "drive", "driving", "walk", "walking",
        "stand", "standing", "stand next to", "standing next to", "push", "pushing",
    },
    "food_activity": {
        "eat", "eating", "drink", "drinking", "cook", "cooking", "cut", "cutting",
        "prepare", "preparing",
    },
    "music_performance": {
        "play", "playing", "sing", "singing", "perform", "performing",
    },
    "animal_handling": {
        "ride horse", "hold horse", "lead", "leading", "feed", "feeding",
    },
}


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def load_predictions(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_object_refs(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]] = {x for x in row["objects"].split("|") if x}

    return refs


def load_relationship_refs(path: Path) -> dict[str, set[tuple[str, str, str]]]:
    refs: dict[str, set[tuple[str, str, str]]] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            triples = set()

            for raw in row["relationships"].split("|"):
                parts = raw.split("::")
                if len(parts) == 3:
                    triples.add((parts[0], parts[1], parts[2]))

            refs[row["image_id"]] = triples

    return refs


# ---------------------------------------------------------------------
# Normalization and hierarchy
# ---------------------------------------------------------------------

def label_to_category(label: str) -> str:
    return CATEGORY_MAP.get(normalize_text(label), "object")


def label_to_acoustic_family(label: str) -> str | None:
    label = normalize_text(label)

    for family, labels in ACOUSTIC_ENTITY_FAMILIES.items():
        if label in labels:
            return family

    category = label_to_category(label)

    if category in {"human", "animal", "vehicle"}:
        return category

    if category == "structure":
        return "built_context"

    if category == "vegetation":
        return "nature_context"

    if category == "tool":
        return "tool_machinery"

    if category == "food":
        return "food_market"

    return None


def verb_to_interaction_family(verb: str) -> str | None:
    verb = normalize_text(verb)

    for family, verbs in INTERACTION_FAMILIES.items():
        if verb in verbs:
            return family

    return None


def scene_family(scene_label: str) -> str | None:
    groups = scene_groups_for_label(scene_label)
    return groups[0] if groups else None


def prf(pred: set[Any], gt: set[Any]) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0

    if not pred or not gt:
        return 0.0, 0.0, 0.0

    tp = len(pred & gt)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def count_bin(count: int) -> str:
    if count <= 0:
        return "none"
    if count == 1:
        return "single"
    if count <= 3:
        return "few"
    if count <= 10:
        return "group"
    return "dense"


# ---------------------------------------------------------------------
# Prediction extraction
# ---------------------------------------------------------------------

def prediction_labels(pred_json: dict[str, Any], object_alias: dict[str, str]) -> set[str]:
    labels = set()

    for entity in pred_json.get("core", {}).get("entities", []):
        label = canonicalize(entity.get("label", ""), object_alias)
        if label:
            labels.add(label)

    return labels


def prediction_family_counts(pred_json: dict[str, Any], object_alias: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    for entity in pred_json.get("core", {}).get("entities", []):
        label = canonicalize(entity.get("label", ""), object_alias)
        family = label_to_acoustic_family(label)

        if family is None:
            continue

        counts[family] += int(entity.get("count_estimate", 1))

    return dict(counts)


def prediction_interaction_families(
    pred_json: dict[str, Any],
    relationship_alias: dict[str, str],
) -> set[str]:
    families = set()

    for interaction in pred_json.get("core", {}).get("observed_interactions", []):
        verb = canonicalize(interaction.get("verb", ""), relationship_alias)
        family = verb_to_interaction_family(verb)

        if family:
            families.add(family)

    return families


# ---------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------

def gt_family_counts(gt_objects: set[str], object_alias: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    for label in gt_objects:
        canonical = canonicalize(label, object_alias)
        family = label_to_acoustic_family(canonical)

        if family is None:
            continue

        counts[family] += 1

    return dict(counts)


def gt_interaction_families(
    gt_relationships: set[tuple[str, str, str]],
    relationship_alias: dict[str, str],
) -> set[str]:
    families = set()

    for _, predicate, _ in gt_relationships:
        predicate = canonicalize(predicate, relationship_alias)
        family = verb_to_interaction_family(predicate)

        if family:
            families.add(family)

    return families


# ---------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------

def count_bin_accuracy(pred_counts: dict[str, int], gt_counts: dict[str, int]) -> float:
    families = sorted(set(pred_counts) | set(gt_counts))

    if not families:
        return 1.0

    correct = 0

    for family in families:
        if count_bin(pred_counts.get(family, 0)) == count_bin(gt_counts.get(family, 0)):
            correct += 1

    return correct / len(families)


def weighted_presence_recall(pred_families: set[str], gt_families: set[str]) -> float:
    relevant_gt = gt_families & DISCRETE_SOUND_FAMILIES

    if not relevant_gt:
        return 1.0

    return len(pred_families & relevant_gt) / len(relevant_gt)


def compute_image_metrics(
    row: dict[str, str],
    object_refs: dict[str, set[str]],
    relationship_refs: dict[str, set[tuple[str, str, str]]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
) -> dict[str, Any]:
    pred_json = load_json(row["json_path"])
    image_id = row["image_id"]

    core = pred_json.get("core", {})
    pred_scene_label = core.get("scene", {}).get("label", "")
    pred_indoor_outdoor = core.get("scene", {}).get("indoor_outdoor", "unknown")

    pred_scene_family = scene_family(pred_scene_label)
    pred_io_from_scene = infer_indoor_outdoor_from_scene(pred_scene_label)

    gt_objects = object_refs.get(image_id, set())
    gt_relationships = relationship_refs.get(image_id, set())

    pred_labels = prediction_labels(pred_json, object_alias)
    gt_labels = {canonicalize(label, object_alias) for label in gt_objects}

    exact_p, exact_r, exact_f1 = prf(pred_labels, gt_labels)

    pred_counts = prediction_family_counts(pred_json, object_alias)
    gt_counts = gt_family_counts(gt_objects, object_alias)

    pred_families = set(pred_counts)
    gt_families = set(gt_counts)

    family_p, family_r, family_f1 = prf(pred_families, gt_families)

    pred_discrete = pred_families & DISCRETE_SOUND_FAMILIES
    gt_discrete = gt_families & DISCRETE_SOUND_FAMILIES
    discrete_p, discrete_r, discrete_f1 = prf(pred_discrete, gt_discrete)

    pred_interaction_fams = prediction_interaction_families(pred_json, relationship_alias)
    gt_interaction_fams = gt_interaction_families(gt_relationships, relationship_alias)
    int_p, int_r, int_f1 = prf(pred_interaction_fams, gt_interaction_fams)

    # Visual Genome has no direct Places365 scene label. These scene fields are
    # therefore diagnostic predictions, not GT accuracy. They are still useful
    # to compare scene models and ambience-oriented distributions.
    scene_known = pred_scene_family is not None
    indoor_outdoor_known = pred_indoor_outdoor != "unknown" or pred_io_from_scene != "unknown"

    return {
        "image_id": image_id,
        "detector": row["detector"],
        "scene_model": row["scene_model"],
        "captioner": row["captioner"],
        "scene_label": pred_scene_label,
        "scene_family": pred_scene_family or "unknown",
        "scene_family_known": float(scene_known),
        "indoor_outdoor": pred_indoor_outdoor,
        "indoor_outdoor_known": float(indoor_outdoor_known),
        "entity_exact_precision": exact_p,
        "entity_exact_recall": exact_r,
        "entity_exact_f1": exact_f1,
        "entity_family_precision": family_p,
        "entity_family_recall": family_r,
        "entity_family_f1": family_f1,
        "discrete_family_precision": discrete_p,
        "discrete_family_recall": discrete_r,
        "discrete_family_f1": discrete_f1,
        "family_count_bin_accuracy": count_bin_accuracy(pred_counts, gt_counts),
        "weighted_discrete_presence_recall": weighted_presence_recall(pred_families, gt_families),
        "interaction_family_precision": int_p,
        "interaction_family_recall": int_r,
        "interaction_family_f1": int_f1,
        "n_pred_families": len(pred_families),
        "n_gt_families": len(gt_families),
        "n_pred_discrete_families": len(pred_discrete),
        "n_gt_discrete_families": len(gt_discrete),
        "n_pred_interaction_families": len(pred_interaction_fams),
        "n_gt_interaction_families": len(gt_interaction_fams),
    }


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["detector"], row["scene_model"], row["captioner"])].append(row)

    metric_keys = [
        key for key in rows[0].keys()
        if key not in {"image_id", "detector", "scene_model", "captioner", "scene_label", "scene_family", "indoor_outdoor"}
    ]

    output = []

    for (detector, scene_model, captioner), items in grouped.items():
        result = {
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
            "n": len(items),
        }

        for key in metric_keys:
            result[key] = sum(float(item[key]) for item in items) / len(items)

        output.append(result)

    return output


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--vg-relationship-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--relationship-alias", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-image-output",
        default=None,
        help="Optional path for per-image soundscape semantic metrics.",
    )

    args = parser.parse_args()

    predictions = load_predictions(Path(args.manifest))
    object_refs = load_object_refs(Path(args.vg_object_refs))
    relationship_refs = load_relationship_refs(Path(args.vg_relationship_refs))
    object_alias = load_alias_map(args.object_alias)
    relationship_alias = load_alias_map(args.relationship_alias)

    per_image_rows = [
        compute_image_metrics(
            row=row,
            object_refs=object_refs,
            relationship_refs=relationship_refs,
            object_alias=object_alias,
            relationship_alias=relationship_alias,
        )
        for row in predictions
    ]

    summary_rows = aggregate(per_image_rows)
    write_csv(summary_rows, Path(args.output))

    if args.per_image_output:
        write_csv(per_image_rows, Path(args.per_image_output))

    print(f"[OK] Soundscape semantic metrics saved to {args.output}")
    if args.per_image_output:
        print(f"[OK] Per-image metrics saved to {args.per_image_output}")


if __name__ == "__main__":
    main()
