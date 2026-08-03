from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import csv

from metrics.audioset_semantics import compute_audioset_metrics
from scripts.evaluation.vg_utils import load_alias_map, load_json


def compute_image_metrics(row, object_refs, relationship_refs, object_alias, relationship_alias):
    pred_json = load_json(row["json_path"])
    image_id = row["image_id"]

    metrics = compute_audioset_metrics(
        pred_json=pred_json,
        gt_objects=object_refs.get(image_id, set()),
        gt_relationships=relationship_refs.get(image_id, set()),
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )

    return {
        "image_id": image_id,
        "detector": row["detector"],
        "scene_model": row["scene_model"],
        "captioner": row["captioner"],
        **metrics,
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["detector"], row["scene_model"], row["captioner"])].append(row)

    metric_keys = [
        key for key in rows[0].keys()
        if key not in {
            "image_id", "detector", "scene_model", "captioner", "scene_label",
            "scene_family", "background_family", "indoor_outdoor",
        }
        and isinstance(rows[0][key], (int, float))
        and not isinstance(rows[0][key], bool)
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


def load_predictions(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
