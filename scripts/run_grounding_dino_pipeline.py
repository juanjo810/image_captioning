"""Run Workflow B with real Grounding DINO detections.

This script validates the first real end-to-end path:

image -> Grounding DINO -> geometry -> fusion -> CORE + EXTENDED JSON

Scene classification and HOI are still placeholders at this stage. This is
intentional: we validate object detection and fusion before adding more models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from src.fusion import build_from_modules
from src.postprocessing import filter_detections
from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter
from src.workflow_b.places365_adapter import Places365Adapter
from src.workflow_b.vocabularies import (
    infer_indoor_outdoor_from_scene,
    iter_grounding_prompt_batches
)


def infer_indoor_outdoor(scene_label: str) -> str:
    outdoor_keywords = {
        "forest", "path", "pasture", "farm", "field", "broadleaf",
        "rainforest", "bamboo forest", "yard", "garden", "road",
        "mountain", "valley", "village"
    }
    indoor_keywords = {
        "room", "kitchen", "bedroom", "bathroom", "corridor",
        "office", "classroom", "indoor"
    }

    label = scene_label.lower()

    if any(k in label for k in outdoor_keywords):
        return "outdoor"

    if any(k in label for k in indoor_keywords):
        return "indoor"

    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="/home/jovyan/projects/data/test.jpg",
    )
    parser.add_argument(
        "--scene-architecture",
        choices=["resnet50", "densenet161"],
        default="resnet50",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.35,
    )

    args = parser.parse_args()

    base = Path("/home/jovyan/projects")
    image_path = Path(args.image)

    detector = GroundingDINOAdapter(
        config_path=base / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path=base / "models/groundingdino_swint_ogc.pth",
    )

    scene_model = Places365Adapter(
        architecture=args.scene_architecture,
        categories_path=base / "places365/categories_places365.txt",
        checkpoint_path=base
        / "models"
        / "places365"
        / f"{args.scene_architecture}_places365.pth.tar",
    )
    
    scene = scene_model.predict(
        image_path=image_path,
        topk=5,
    )

    raw_detections = []

    for batch in iter_grounding_prompt_batches():
        batch_detections = detector.predict(
            image_path=image_path,
            prompt=batch["prompt"],
            box_threshold=batch["box_threshold"],
            text_threshold=batch["text_threshold"],
        )

        print(f"\nRAW DETECTIONS [{batch['name']}]")
        for d in batch_detections:
            print(
                d.label,
                round(d.confidence, 3),
                [round(x, 1) for x in d.bbox],
            )

        raw_detections.extend(batch_detections)

    detections = filter_detections(
        raw_detections,
        min_confidence=args.min_confidence,
        nms_iou_threshold=0.85,
        semantic_iou_threshold=0.30,
        semantic_containment_threshold=0.65,
    )

    print("FILTERED DETECTIONS")
    for d in detections:
        print(d.label, round(d.confidence, 3), [round(x, 1) for x in d.bbox])

    with Image.open(image_path) as img:
        width, height = img.size

    core, extended = build_from_modules(
        image_id=image_path.stem,
        width=width,
        height=height,
        detections=detections,
        scene_label=scene["label"],
        scene_conf=scene["confidence"],
        hoi=[],
    )

    # Override indoor/outdoor because the current fusion layer does not infer it yet.
    core.scene.indoor_outdoor = infer_indoor_outdoor_from_scene(scene["label"])

    output = {
        "core": core.model_dump(),
        "extended": extended.model_dump(),
        "metadata": {
            "detector": "grounding_dino",
            "scene_model": scene,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_confidence": args.min_confidence,
        },
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()