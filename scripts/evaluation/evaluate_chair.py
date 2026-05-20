from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.evaluation.vg_utils import load_alias_map, canonicalize


CORE_FIELDS = [
    "detector",
    "scene_model",
    "captioner",
    "CHAIRi_VG",
    "CHAIRi_JSON",
    "avg_caption_objects_mentioned",
    "avg_hallucinated_vg",
    "avg_hallucinated_json",
    "n",
]

EXTENDED_FIELDS = [
    "detector",
    "scene_model",
    "captioner",
    "CHAIRs_VG",
    "CHAIRi_VG",
    "CHAIRs_JSON",
    "CHAIRi_JSON",
    "avg_caption_objects_mentioned",
    "avg_hallucinated_vg",
    "avg_hallucinated_json",
    "n",
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_predictions(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_vg_object_refs(path: Path) -> dict[str, set[str]]:
    refs = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]] = {
                obj for obj in row["objects"].split("|") if obj
            }

    return refs


def load_json_entities(json_path: Path, object_alias: dict[str, str]) -> set[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    entities = data.get("core", {}).get("entities", [])

    return {
        canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("label")
    }


def build_object_vocabulary(
    vg_refs: dict[str, set[str]],
    object_alias: dict[str, str],
) -> set[str]:
    vocab = set()

    for objects in vg_refs.values():
        for obj in objects:
            canonical = canonicalize(obj, object_alias)
            if canonical:
                vocab.add(canonical)

    return vocab


def extract_caption_objects(
    caption: str,
    object_vocab: set[str],
    object_alias: dict[str, str],
) -> set[str]:
    caption_norm = normalize_text(caption)
    found = set()

    for obj in sorted(object_vocab, key=lambda x: len(x.split()), reverse=True):
        obj_norm = normalize_text(obj)

        if not obj_norm:
            continue

        pattern = r"\b" + re.escape(obj_norm) + r"s?\b"

        if re.search(pattern, caption_norm):
            found.add(canonicalize(obj_norm, object_alias))

    return found


def chair_scores(
    caption_objects: set[str],
    reference_objects: set[str],
) -> tuple[float, float, int, int]:
    if not caption_objects:
        return 0.0, 0.0, 0, 0

    hallucinated = caption_objects - reference_objects

    chair_i = len(hallucinated) / len(caption_objects)
    chair_s = 1.0 if hallucinated else 0.0

    return chair_s, chair_i, len(hallucinated), len(caption_objects)


def select_fieldnames(metrics_profile: str) -> list[str]:
    if metrics_profile == "core":
        return CORE_FIELDS
    return EXTENDED_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metrics-profile",
        choices=["core", "extended"],
        default="extended",
        help="Use 'core' for compact paper-ready metrics or 'extended' for legacy metrics.",
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    vg_refs_path = Path(args.vg_object_refs)

    object_alias = load_alias_map(args.object_alias)

    predictions = load_predictions(manifest_path)
    vg_refs = load_vg_object_refs(vg_refs_path)
    object_vocab = build_object_vocabulary(vg_refs, object_alias)

    grouped = defaultdict(list)

    for row in predictions:
        image_id = row["image_id"]
        caption = row["caption"]
        json_path = Path(row["json_path"])

        caption_objects = extract_caption_objects(
            caption=caption,
            object_vocab=object_vocab,
            object_alias=object_alias,
        )

        vg_objects = vg_refs.get(image_id, set())
        json_objects = load_json_entities(json_path, object_alias)

        chair_s_vg, chair_i_vg, halluc_vg, mentioned = chair_scores(
            caption_objects=caption_objects,
            reference_objects=vg_objects,
        )

        chair_s_json, chair_i_json, halluc_json, _ = chair_scores(
            caption_objects=caption_objects,
            reference_objects=json_objects,
        )

        condition = (
            row["detector"],
            row["scene_model"],
            row["captioner"],
        )

        grouped[condition].append(
            {
                "CHAIRs_VG": chair_s_vg,
                "CHAIRi_VG": chair_i_vg,
                "CHAIRs_JSON": chair_s_json,
                "CHAIRi_JSON": chair_i_json,
                "caption_objects_mentioned": mentioned,
                "hallucinated_vg": halluc_vg,
                "hallucinated_json": halluc_json,
            }
        )

    rows = []

    for (detector, scene_model, captioner), items in grouped.items():
        n = len(items)

        rows.append(
            {
                "detector": detector,
                "scene_model": scene_model,
                "captioner": captioner,
                "CHAIRs_VG": sum(x["CHAIRs_VG"] for x in items) / n,
                "CHAIRi_VG": sum(x["CHAIRi_VG"] for x in items) / n,
                "CHAIRs_JSON": sum(x["CHAIRs_JSON"] for x in items) / n,
                "CHAIRi_JSON": sum(x["CHAIRi_JSON"] for x in items) / n,
                "avg_caption_objects_mentioned": (
                    sum(x["caption_objects_mentioned"] for x in items) / n
                ),
                "avg_hallucinated_vg": (
                    sum(x["hallucinated_vg"] for x in items) / n
                ),
                "avg_hallucinated_json": (
                    sum(x["hallucinated_json"] for x in items) / n
                ),
                "n": n,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = select_fieldnames(args.metrics_profile)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)

    print(f"[OK] CHAIR metrics saved to {output_path}")


if __name__ == "__main__":
    main()
