"""Run Workflow B with real Grounding DINO detections.

This script validates the first real end-to-end path:

image -> Grounding DINO -> geometry -> fusion -> CORE + EXTENDED JSON

Scene classification and HOI are still placeholders at this stage. This is
intentional: we validate object detection and fusion before adding more models.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.fusion import build_from_modules
from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter

from src.workflow_b.vocabularies import TRADITIONAL_ENVIRONMENT_VOCAB, build_prompt

from src.postprocessing import filter_detections


def main() -> None:
    base = Path("/home/jovyan/projects")
    image_path = base / "data/test.jpg"

    adapter = GroundingDINOAdapter(
        config_path=base / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        checkpoint_path=base / "models/groundingdino_swint_ogc.pth",
    )

    print(type(TRADITIONAL_ENVIRONMENT_VOCAB))
    prompt = build_prompt(TRADITIONAL_ENVIRONMENT_VOCAB)
    raw_detections  = adapter.predict(
        image_path=image_path,
        prompt=prompt,
        box_threshold=0.30,
        text_threshold=0.25,
    )

    detections = filter_detections(
        raw_detections,
        min_confidence=0.30,
        nms_iou_threshold=0.85,
        semantic_iou_threshold=0.30,
        semantic_containment_threshold=0.65,
    )

    with Image.open(image_path) as img:
        width, height = img.size

    # Temporary placeholders until Places365 and UPT are integrated.
    scene_label = "unknown scene"
    scene_confidence = 0.0
    hoi_triplets = []

    core, extended = build_from_modules(
        image_id=image_path.stem,
        width=width,
        height=height,
        detections=detections,
        scene_label=scene_label,
        scene_conf=scene_confidence,
        hoi=hoi_triplets,
    )

    print("\n=== CORE JSON ===")
    print(json.dumps(core.model_dump(), indent=2, ensure_ascii=False))

    print("\n=== EXTENDED JSON ===")
    print(json.dumps(extended.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
