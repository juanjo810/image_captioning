from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from scripts.evaluation.vg_utils import is_audioset_core_json


def parse_condition(path: Path) -> tuple[str, str, str]:
    name = path.parent.name

    captioner = "gemma" if "captions_gemma" in str(path) else "template"

    if name.startswith("grounding_dino_"):
        detector = "grounding_dino"
        scene_model = name.replace("grounding_dino_", "")
    elif name.startswith("owlv2_"):
        detector = "owlv2"
        scene_model = name.replace("owlv2_", "")
    elif name in {"json", "legacy"}:
        # Workflow A layout: <output_dir>/json/<id>.json (or
        # <output_dir>/legacy/json/<id>.json) — the immediate parent is never
        # the run's identifying name, so climb to <output_dir> instead.
        detector = "unknown"
        scene_model = path.parent.parent.name
    else:
        detector = "unknown"
        scene_model = "unknown"

    return detector, scene_model, captioner


def summarize_json(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    detector, scene_model, captioner = parse_condition(json_path)
    metadata = data.get("metadata", {})
    extended = data.get("extended", {})
    global_geometry = extended.get("global_geometry", {})

    core = data.get("core", {})
    if is_audioset_core_json(data):
        scene_node = core.get("scene", {})
        nodes = core.get("nodes", [])

        return {
            "image_id": core.get("image_id", json_path.stem),
            "json_path": str(json_path),
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
            "caption": core.get("caption", ""),
            "scene_audioset_id": scene_node.get("audioset_id", ""),
            "scene_audioset_name": scene_node.get("audioset_name", ""),
            "scene_confidence": scene_node.get("confidence", ""),
            "n_nodes": len(nodes),
            "total_object_coverage": global_geometry.get("total_object_coverage", ""),
            "object_density_proxy": global_geometry.get("object_density_proxy", ""),
            "metadata_detector": metadata.get("detector", ""),
            "metadata_scene_model": metadata.get("scene_model", {}).get("architecture", ""),
            "metadata_caption_mode": metadata.get("caption_mode", ""),   
        }

    else:
        scene = core.get("scene", {})
        entities = core.get("entities", [])
        interactions = core.get("observed_interactions", [])

        return {
            "image_id": core.get("image_id", json_path.stem),
            "json_path": str(json_path),
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
            "caption": core.get("caption", ""),
            "scene_label": scene.get("label", ""),
            "scene_confidence": scene.get("confidence", ""),
            "indoor_outdoor": scene.get("indoor_outdoor", ""),
            "n_entities": len(entities),
            "n_interactions": len(interactions),
            "crowd_level": core.get("environment", {}).get("crowd_level", ""),
            "activity_level": core.get("environment", {}).get("activity_level", ""),
            "total_object_coverage": global_geometry.get("total_object_coverage", ""),
            "object_density_proxy": global_geometry.get("object_density_proxy", ""),
            "metadata_detector": metadata.get("detector", ""),
            "metadata_scene_model": metadata.get("scene_model", {}).get("architecture", ""),
            "metadata_caption_mode": metadata.get("caption_mode", ""),
        }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Prediction JSON directories.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV manifest path.",
    )

    args = parser.parse_args()

    rows = []

    for input_dir in args.inputs:
        input_path = Path(input_dir)

        for json_path in sorted(input_path.rglob("*.json")):
            rows.append(summarize_json(json_path))

    if not rows:
        raise RuntimeError("No JSON files found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Manifest saved to {output_path}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()