from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_references(path: Path) -> dict[str, list[str]]:
    refs = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]].append(row["reference_caption"])

    return dict(refs)


def load_predictions(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate_condition(rows: list[dict], refs_by_image: dict[str, list[str]]) -> dict:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice

    gts = {}
    res = {}

    for idx, row in enumerate(rows):
        image_id = row["image_id"]

        if image_id not in refs_by_image:
            continue

        key = str(idx)

        gts[key] = refs_by_image[image_id]
        res[key] = [row["caption"]]
        
    if not gts:
        return {
            "CIDEr": 0.0,
            "METEOR": 0.0,
            "SPICE": 0.0,
            "n": 0,
        }

    cider_score, _ = Cider().compute_score(gts, res)
    meteor_score, _ = Meteor().compute_score(gts, res)
    spice_score, _ = Spice().compute_score(gts, res)

    return {
        "CIDEr": cider_score,
        "METEOR": meteor_score,
        "SPICE": spice_score,
        "n": len(gts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--caption-refs", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    predictions = load_predictions(Path(args.manifest))
    refs_by_image = load_references(Path(args.caption_refs))

    grouped = defaultdict(list)

    for row in predictions:
        condition = (
            row["detector"],
            row["scene_model"],
            row["captioner"],
        )
        grouped[condition].append(row)

    out_rows = []

    for (detector, scene_model, captioner), rows in grouped.items():
        scores = evaluate_condition(rows, refs_by_image)

        out_rows.append(
            {
                "detector": detector,
                "scene_model": scene_model,
                "captioner": captioner,
                **scores,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["detector", "scene_model", "captioner", "CIDEr", "METEOR", "SPICE", "n"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[OK] Caption metrics saved to {output_path}")


if __name__ == "__main__":
    main()
