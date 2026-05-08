"""Run Workflow B with real Grounding DINO detections.

Current path:

image -> Places365 -> Grounding DINO -> post-processing -> geometry/fusion
      -> optional HOI adapter -> CORE + EXTENDED JSON

HOI is adapter-based and disabled by default. The dummy mode validates the
interaction fusion contract before integrating an external HOI model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
from dotenv import load_dotenv

from PIL import Image

from src.fusion import build_from_modules
from src.postprocessing import filter_detections
from src.captioning import build_caption
from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter
from src.workflow_b.hoi_adapter import DummyHOIAdapter, RawHOITriplet
from src.workflow_b.hoi_fusion import build_observed_interactions
from src.workflow_b.places365_adapter import Places365Adapter
from src.workflow_b.vocabularies import (
    infer_indoor_outdoor_from_scene,
    iter_grounding_prompt_batches,
)
from src.workflow_b.upt_adapter import UPTAdapter



def make_demo_dummy_hoi() -> DummyHOIAdapter:
    """Create a handcrafted HOI adapter for smoke tests.

    The coordinates roughly target the test2 street image used during local
    development. For arbitrary images this may produce no interaction because
    IoU matching will reject non-overlapping boxes.
    """
    return DummyHOIAdapter(
        triplets=[
            RawHOITriplet(
                human_bbox=[728.0, 737.0, 883.0, 901.0],
                object_bbox=[704.0, 772.0, 841.0, 913.0],
                verb="sitting on",
                confidence=0.90,
            )
        ]
    )


def main() -> None:
    load_dotenv()
    
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
    parser.add_argument(
        "--hoi",
        choices=["none", "dummy", "upt"],
        default="upt",
        help="HOI backend to use. 'dummy' is only for fusion smoke tests.",
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

    for batch in iter_grounding_prompt_batches(scene["label"]):
        batch_detections = detector.predict(
            image_path=image_path,
            prompt=batch["prompt"],
            box_threshold=batch["box_threshold"],
            text_threshold=batch["text_threshold"],
        )
        '''
        print(f"\nRAW DETECTIONS [{batch['name']}]")
        for d in batch_detections:
            print(
                d.label,
                round(d.confidence, 3),
                [round(x, 1) for x in d.bbox],
            )
        '''

        raw_detections.extend(batch_detections)

    detections = filter_detections(
        raw_detections,
        min_confidence=args.min_confidence,
        nms_iou_threshold=0.85,
        semantic_iou_threshold=0.30,
        semantic_containment_threshold=0.65,
    )
    '''
    print("FILTERED DETECTIONS")
    for d in detections:
        print(d.label, round(d.confidence, 3), [round(x, 1) for x in d.bbox])
    '''
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

    core.scene.indoor_outdoor = infer_indoor_outdoor_from_scene(scene["label"])

    raw_hois = []
    if args.hoi == "dummy":
        hoi_adapter = make_demo_dummy_hoi()
        raw_hois = hoi_adapter.predict(image_path)
        interactions = build_observed_interactions(
            raw_hois=raw_hois,
            entities_extended=extended.model_dump()["entities_extended"],
        )
        core.observed_interactions = interactions
        core.environment.activity_level = "medium" if interactions else "low"
        core.caption = core.caption if not interactions else core.caption

    elif args.hoi == "upt":
        hoi_adapter = UPTAdapter(
            upt_root=base / "upt",
            checkpoint_path=base / "models/upt/upt-r50-hicodet.pt",
            data_root=base / "upt/hicodet",
            device="cuda",
            action_score_thresh=0.10,
        )
        raw_hois = hoi_adapter.predict(image_path)
        interactions = build_observed_interactions(
            raw_hois=raw_hois,
            entities_extended=extended.model_dump()["entities_extended"],
        )
        core.observed_interactions = interactions
        core.environment.activity_level = "medium" if interactions else "low"

    '''
    print("RAW HOIS")
    for h in raw_hois:
        print(
            h.verb,
            round(h.confidence, 4),
            "human_bbox=", [round(x, 1) for x in h.human_bbox],
            "object_bbox=", [round(x, 1) for x in h.object_bbox],
        )
    '''

    core.caption = build_caption(
        scene_label=core.scene.label,
        entities=core.entities,
        interactions=core.observed_interactions,
        indoor_outdoor=core.scene.indoor_outdoor,
    )

    output = {
        "core": core.model_dump(),
        "extended": extended.model_dump(),
        "metadata": {
            "detector": "grounding_dino",
            "scene_model": scene,
            "hoi_backend": args.hoi,
            "raw_hoi_count": len(raw_hois),
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_confidence": args.min_confidence,
        },
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
