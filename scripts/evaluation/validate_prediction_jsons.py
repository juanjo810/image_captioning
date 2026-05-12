from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def safe_load_json(path: Path) -> tuple[bool, dict | None, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return True, data, ""
    except Exception as exc:
        return False, None, str(exc)


def compute_json_metrics(json_path: Path) -> dict:
    valid, data, error = safe_load_json(json_path)

    if not valid or data is None:
        return {
            "json_path": str(json_path),
            "json_valid": False,
            "json_error": error,
        }

    core = data.get("core", {})
    extended = data.get("extended", {})

    entities = core.get("entities", [])
    interactions = core.get("observed_interactions", [])
    entities_extended = extended.get("entities_extended", [])

    entity_ids = {e.get("id") for e in entities}
    extended_ids = {e.get("id") for e in entities_extended}

    interaction_valid_count = 0

    for inter in interactions:
        if (
            inter.get("subject_id") in entity_ids
            and inter.get("object_id") in entity_ids
            and inter.get("verb")
        ):
            interaction_valid_count += 1

    n_interactions = len(interactions)

    return {
        "json_path": str(json_path),
        "image_id": core.get("image_id", json_path.stem),
        "json_valid": True,
        "json_error": "",
        "n_entities": len(entities),
        "n_entities_extended": len(entities_extended),
        "n_interactions": n_interactions,
        "interaction_valid_ratio": (
            interaction_valid_count / n_interactions
            if n_interactions > 0
            else 1.0
        ),
        "entity_id_consistency": entity_ids.issubset(extended_ids) if extended_ids else True,
        "has_caption": bool(core.get("caption")),
        "scene_label": core.get("scene", {}).get("label", ""),
        "scene_confidence": core.get("scene", {}).get("confidence", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Directory with prediction JSON files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    rows = [
        compute_json_metrics(path)
        for path in sorted(input_path.rglob("*.json"))
    ]

    if not rows:
        raise RuntimeError(f"No JSON files found in {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Structured metrics saved to {output_path}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()