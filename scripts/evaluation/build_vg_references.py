from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.evaluation.vg_utils import (
    canonicalize,
    load_alias_map,
    load_json,
    normalize_text,
)


def load_manifest_image_ids(manifest_path: Path) -> set[str]:
    image_ids = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_ids.add(str(row["image_id"]))

    return image_ids


def build_region_references(
    region_descriptions_path: Path,
    image_ids: set[str],
) -> list[dict]:
    data = load_json(region_descriptions_path)
    rows = []

    for item in data:
        image_id = str(item.get("id") or item.get("image_id"))

        if image_id not in image_ids:
            continue

        for region in item.get("regions", []):
            phrase = normalize_text(region.get("phrase", ""))

            if phrase:
                rows.append(
                    {
                        "image_id": image_id,
                        "reference_caption": phrase,
                    }
                )

    return rows


def build_object_references(
    objects_path: Path,
    image_ids: set[str],
    object_alias: dict[str, str],
) -> list[dict]:
    data = load_json(objects_path)
    rows = []

    for item in data:
        image_id = str(item.get("image_id"))

        if image_id not in image_ids:
            continue

        objects = set()

        for obj in item.get("objects", []):
            names = obj.get("names", [])

            for name in names:
                canonical = canonicalize(name, object_alias)

                if canonical:
                    objects.add(canonical)

        rows.append(
            {
                "image_id": image_id,
                "objects": "|".join(sorted(objects)),
            }
        )

    return rows


def build_relationship_references(
    relationships_path: Path,
    image_ids: set[str],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
) -> list[dict]:
    data = load_json(relationships_path)
    rows = []

    for item in data:
        image_id = str(item.get("image_id"))

        if image_id not in image_ids:
            continue

        triples = set()

        for rel in item.get("relationships", []):
            subj_names = rel.get("subject", {}).get("names", [])
            obj_names = rel.get("object", {}).get("names", [])
            predicate = canonicalize(rel.get("predicate", ""), relationship_alias)

            if not predicate:
                continue

            for subj in subj_names:
                for obj in obj_names:
                    subj_c = canonicalize(subj, object_alias)
                    obj_c = canonicalize(obj, object_alias)

                    if subj_c and obj_c:
                        triples.add(f"{subj_c}::{predicate}::{obj_c}")

        rows.append(
            {
                "image_id": image_id,
                "relationships": "|".join(sorted(triples)),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError(f"No rows to write: {output_path}")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--region-descriptions", required=True)
    parser.add_argument("--objects", required=True)
    parser.add_argument("--relationships", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--relationship-alias", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)

    image_ids = load_manifest_image_ids(manifest_path)

    object_alias = load_alias_map(args.object_alias)
    relationship_alias = load_alias_map(args.relationship_alias)

    region_rows = build_region_references(
        region_descriptions_path=Path(args.region_descriptions),
        image_ids=image_ids,
    )

    object_rows = build_object_references(
        objects_path=Path(args.objects),
        image_ids=image_ids,
        object_alias=object_alias,
    )

    relationship_rows = build_relationship_references(
        relationships_path=Path(args.relationships),
        image_ids=image_ids,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )

    write_csv(region_rows, output_dir / "vg_caption_references.csv")
    write_csv(object_rows, output_dir / "vg_object_references.csv")
    write_csv(relationship_rows, output_dir / "vg_relationship_references.csv")

    print(f"[OK] Caption refs: {len(region_rows)}")
    print(f"[OK] Object refs: {len(object_rows)}")
    print(f"[OK] Relationship refs: {len(relationship_rows)}")


if __name__ == "__main__":
    main()
  