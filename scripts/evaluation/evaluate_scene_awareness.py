from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.evaluation.vg_utils import canonicalize, load_alias_map, load_json, normalize_text
from src.workflow_b.constants import CATEGORY_MAP, SCENE_GROUPS
from src.workflow_b.vocabularies import infer_indoor_outdoor_from_scene, scene_groups_for_label
from scripts.evaluation.vg_utils import is_audioset_core_json


# ---------------------------------------------------------------------
# Scene-aware soundscape evaluation
# ---------------------------------------------------------------------
#
# This script evaluates semantic consistency between predicted scene groups
# and structured visual evidence.
#
# The refactored version intentionally removes several heuristic overlap
# metrics that added complexity without substantial interpretability gains.
#
# The focus is now on:
#   - scene consistency
#   - indoor/outdoor coherence
#   - grounded semantic plausibility
#


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
        "forklift", "van", "taxi", "scooter", "airplane", "plane",
    },
    "water": {
        "water", "river", "lake", "pond", "creek", "waterfall", "sea", "ocean",
        "wave", "canal", "harbor", "beach", "coast", "stream",
    },
    "music_instrument": {
        "instrument", "microphone", "speaker", "guitar", "drum", "violin", "piano",
        "accordion", "flute", "trumpet", "saxophone", "tambourine", "bongo",
    },
    "tool_machinery": {
        "tool", "farm tool", "plow", "machine", "engine", "pipe", "crane", "forklift",
        "equipment", "medical equipment", "computer", "monitor", "traffic light",
        "street lamp", "lamp", "cooking pot", "sink", "wrench", "turbine",
    },
    "food_market": {
        "food", "bread", "fruit", "vegetable", "vegetables", "grain", "hay", "basket",
        "crate", "market stall", "stand", "plate", "bowl", "drink", "sandwich",
    },
    "built_context": {
        "building", "house", "church", "cathedral", "temple", "tower", "bell tower",
        "wall", "stone wall", "fence", "gate", "door", "window", "bridge", "barn",
        "stable", "corral", "animal pen", "monument", "statue", "arch", "column",
        "altar", "grave", "boathouse", "dock", "pier", "gazebo", "lifeguard tower",
        "street", "road", "sidewalk", "crosswalk", "stage", "desk", "table", "chair",
        "bench", "cabinet", "counter", "screen", "display", "storefront", "shopfront",
    },
    "nature_context": {
        "tree", "plant", "grass", "flower", "crop", "field crop", "bush", "forest",
        "field", "rock", "sky", "mountain", "sand", "path", "trail", "lawn", "log",
        "hill", "shore", "snow", "beach",
    },
}

FAMILY_WEIGHTS = {
    "human": 1.00,
    "animal": 0.95,
    "vehicle": 0.95,
    "water": 0.90,
    "music_instrument": 0.90,
    "tool_machinery": 0.75,
    "food_market": 0.55,
    "nature_context": 0.45,
    "built_context": 0.35,
}

SCENE_EXPECTED_FAMILIES: dict[str, set[str]] = {
    "rural_traditional": {
        "human", "animal", "vehicle", "tool_machinery", "food_market",
        "nature_context", "built_context",
    },
    "natural_outdoor": {
        "human", "animal", "water", "nature_context",
    },
    "market_public": {
        "human", "food_market", "built_context", "vehicle", "tool_machinery",
    },
    "religious_heritage": {
        "human", "built_context", "music_instrument",
    },
    "indoor_domestic": {
        "human", "food_market", "tool_machinery", "built_context",
    },
    "urban_transport": {
        "human", "vehicle", "tool_machinery", "built_context",
    },
    "coastal_water": {
        "human", "vehicle", "water", "nature_context", "built_context",
    },
    "public_indoor": {
        "human", "built_context", "food_market", "music_instrument", "tool_machinery",
    },
    "education_health_office": {
        "human", "built_context", "tool_machinery",
    },
    "sports_recreation": {
        "human", "built_context", "tool_machinery", "nature_context",
    },
    "industrial_workshop": {
        "human", "vehicle", "tool_machinery", "built_context",
    },
    "garden_park": {
        "human", "animal", "water", "nature_context", "built_context",
    },
    "entertainment_culture": {
        "human", "music_instrument", "tool_machinery", "built_context",
    },
}

INDOOR_SCENE_GROUPS = {
    "indoor_domestic",
    "public_indoor",
    "education_health_office",
}

OUTDOOR_SCENE_GROUPS = set(SCENE_GROUPS) - INDOOR_SCENE_GROUPS


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


def label_to_category(label: str) -> str:
    return CATEGORY_MAP.get(normalize_text(label), "object")


def label_to_family(label: str) -> str | None:
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


def scene_group(scene_label: str) -> str | None:
    groups = scene_groups_for_label(scene_label)
    return groups[0] if groups else None


def group_expected_io(group: str | None) -> str:
    if group is None:
        return "unknown"
    if group in INDOOR_SCENE_GROUPS:
        return "indoor"
    if group in OUTDOOR_SCENE_GROUPS:
        return "outdoor"
    return "unknown"


def labels_to_family_counts(labels: set[str], object_alias: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    for label in labels:
        canonical = canonicalize(label, object_alias)
        family = label_to_family(canonical)
        if family is not None:
            counts[family] += 1

    return dict(counts)


def prediction_family_counts(pred_json: dict[str, Any], object_alias: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)

    if is_audioset_core_json(pred_json):
        for node in pred_json.get("core", {}).get("nodes", []):
            for entity in node.get("visual_evidence_terms", []):
                label = canonicalize(entity, object_alias)
                family = label_to_family(label)
                if family is not None:
                    counts[family] += 1
    else:
        for entity in pred_json.get("core", {}).get("entities", []):
            label = canonicalize(entity.get("label", ""), object_alias)
            family = label_to_family(label)
            if family is not None:
                counts[family] += int(entity.get("count_estimate", 1))

    return dict(counts)


def family_evidence_score(
    observed_counts: dict[str, int],
    expected_families: set[str],
) -> float:
    if not observed_counts:
        return 0.0

    total = 0.0
    supported = 0.0

    for family, count in observed_counts.items():
        weight = FAMILY_WEIGHTS.get(family, 0.20) * max(count, 1)
        total += weight

        if family in expected_families:
            supported += weight

    return supported / total if total > 0 else 0.0


def scene_confidence(pred_json: dict[str, Any]) -> float | str:
    """Workflow B always fills scene.confidence with the real Places365 score;
    Workflow A deliberately never does (asking the VLM for a confidence
    number it would be inventing, not measuring), so this is
    genuinely absent rather than zero. Reported as "n/a" -- not "" and not
    0.0 -- so it reads unambiguously as "not applicable" rather than being
    mistaken for a missing/empty value or a real zero confidence."""
    confidence = pred_json.get("core", {}).get("scene", {}).get("confidence")
    if confidence is None:
        return "n/a"
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return "n/a"


def compute_image_metrics(
    row: dict[str, str],
    object_refs: dict[str, set[str]],
    object_alias: dict[str, str],
) -> dict[str, Any]:
    pred_json = load_json(row["json_path"])
    image_id = row["image_id"]

    core = pred_json.get("core", {})
    scene = core.get("scene", {})
    scene_label = scene.get("label", "")
    declared_io = scene.get("indoor_outdoor", "unknown")

    predicted_group = scene_group(scene_label)
    expected_io = group_expected_io(predicted_group)
    inferred_io = infer_indoor_outdoor_from_scene(scene_label)

    expected_families = SCENE_EXPECTED_FAMILIES.get(predicted_group or "", set())

    pred_counts = prediction_family_counts(pred_json, object_alias)
    gt_counts = labels_to_family_counts(object_refs.get(image_id, set()), object_alias)

    pred_consistency = family_evidence_score(pred_counts, expected_families)
    gt_consistency = family_evidence_score(gt_counts, expected_families)

    io_consistent = float(
        expected_io != "unknown"
        and (
            declared_io == expected_io
            or inferred_io == expected_io
        )
    )

    return {
        "image_id": image_id,
        "detector": row["detector"],
        "scene_model": row["scene_model"],
        "captioner": row["captioner"],
        "scene_label": scene_label,
        "scene_group": predicted_group or "unknown",
        "scene_confidence": scene_confidence(pred_json),
        "scene_group_known": float(predicted_group is not None),
        "scene_indoor_outdoor_consistency": io_consistent,
        "pred_entity_scene_consistency": pred_consistency,
        "gt_entity_scene_consistency": gt_consistency,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["detector"], row["scene_model"], row["captioner"])] .append(row)

    excluded = {
        "image_id", "detector", "scene_model", "captioner", "scene_label", "scene_group",
    }

    metric_keys = [key for key in rows[0].keys() if key not in excluded]

    output = []

    for (detector, scene_model, captioner), items in grouped.items():
        result = {
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
            "n": len(items),
        }

        for key in metric_keys:
            values = [item[key] for item in items if item[key] not in ("", "n/a")]
            result[key] = sum(float(v) for v in values) / len(values) if values else ""

        output.append(result)

    return output


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to write.")

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--per-image-output",
        default=None,
        help="Optional path for per-image scene-aware metrics.",
    )

    args = parser.parse_args()

    predictions = load_predictions(Path(args.manifest))
    object_refs = load_object_refs(Path(args.vg_object_refs))
    object_alias = load_alias_map(args.object_alias)

    per_image_rows = [
        compute_image_metrics(
            row=row,
            object_refs=object_refs,
            object_alias=object_alias,
        )
        for row in predictions
    ]

    summary_rows = aggregate(per_image_rows)

    write_csv(summary_rows, Path(args.output))

    if args.per_image_output:
        write_csv(per_image_rows, Path(args.per_image_output))

    print(f"[OK] Scene-aware metrics saved to {args.output}")

    if args.per_image_output:
        print(f"[OK] Per-image scene-aware metrics saved to {args.per_image_output}")


if __name__ == "__main__":
    main()
