from __future__ import annotations

import argparse
import csv
from pathlib import Path

CORE_METRIC_FIELDS = {
    "caption": [
        "detector",
        "scene_model",
        "captioner",
        "CIDEr",
        "SPICE",
        "n",
    ],
    "chair": [
        "detector",
        "scene_model",
        "captioner",
        "CHAIRi_VG",
        "CHAIRi_JSON",
        "avg_caption_objects_mentioned",
        "avg_hallucinated_vg",
        "avg_hallucinated_json",
        "n",
    ],
    "scene": [
        "detector",
        "scene_model",
        "captioner",
        "scene_group_known",
        "pred_entity_scene_consistency",
        "n",
    ],
    "soundscape": [
        "detector",
        "scene_model",
        "captioner",
        "weighted_entity_family_f1",
        "weighted_discrete_family_f1",
        "acoustic_weighted_discrete_presence_recall",
        "interaction_family_f1",
        "n",
    ],
    "audioset": [
        "detector",
        "scene_model",
        "captioner",
        "audioset_tag_precision",
        "audioset_tag_recall",
        "audioset_tag_f1",
        "audioset_category_precision",
        "audioset_category_recall",
        "audioset_category_f1",
        "n",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metric-group",
        required=True,
        choices=CORE_METRIC_FIELDS.keys(),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("Input CSV is empty.")

    fields = CORE_METRIC_FIELDS[args.metric_group]

    missing = [field for field in fields if field not in rows[0]]
    if missing:
        raise RuntimeError(
            f"Missing required fields for metric group '{args.metric_group}': {missing}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"[OK] Filtered metrics CSV saved to {output_path}")


if __name__ == "__main__":
    main()
