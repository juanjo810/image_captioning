from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.evaluation.vg_utils import (
    get_prediction_entities,
    get_prediction_interactions,
    load_alias_map,
    load_json,
)


def load_object_refs(path: Path) -> dict[str, set[str]]:
    refs = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]] = {
                x for x in row["objects"].split("|") if x
            }

    return refs


def load_relationship_refs(path: Path) -> dict[str, set[tuple[str, str, str]]]:
    refs = {}

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


def prf(pred: set, gt: set) -> tuple[float, float, float]:
    if not pred and not gt:
        return 1.0, 1.0, 1.0

    if not pred:
        return 0.0, 0.0, 0.0

    if not gt:
        return 0.0, 0.0, 0.0

    tp = len(pred & gt)

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt) if gt else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--object-refs", required=True)
    parser.add_argument("--relationship-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--relationship-alias", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    object_alias = load_alias_map(args.object_alias)
    relationship_alias = load_alias_map(args.relationship_alias)

    object_refs = load_object_refs(Path(args.object_refs))
    relationship_refs = load_relationship_refs(Path(args.relationship_refs))

    rows = []

    with Path(args.manifest).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_id = row["image_id"]
            pred_json = load_json(row["json_path"])

            pred_entities = get_prediction_entities(pred_json, object_alias)
            gt_entities = object_refs.get(image_id, set())

            ep, er, ef1 = prf(pred_entities, gt_entities)

            pred_interactions = get_prediction_interactions(
                pred_json,
                object_alias=object_alias,
                relationship_alias=relationship_alias,
            )

            gt_interactions = relationship_refs.get(image_id, set())

            ip, ir, if1 = prf(pred_interactions, gt_interactions)

            out = dict(row)
            out.update(
                {
                    "entity_precision": ep,
                    "entity_recall": er,
                    "entity_f1": ef1,
                    "interaction_precision": ip,
                    "interaction_recall": ir,
                    "interaction_f1": if1,
                    "n_gt_entities": len(gt_entities),
                    "n_pred_entities_norm": len(pred_entities),
                    "n_gt_interactions": len(gt_interactions),
                    "n_pred_interactions_norm": len(pred_interactions),
                }
            )

            rows.append(out)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Structured VG metrics saved to {output_path}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()
