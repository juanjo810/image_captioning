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
from src.workflow_b.audioset_projection import project_detections_to_audioset
from src.postprocessing import filter_detections
from src.captioning import build_caption
from src.workflow_b.grounding_dino_adapter import GroundingDINOAdapter
from src.workflow_b.owlv2_adapter import OWLv2Adapter
from src.workflow_b.places365_adapter import Places365Adapter
from src.workflow_b.hoi_adapter import DummyHOIAdapter, RawHOITriplet
from src.workflow_b.hoi_fusion import build_observed_interactions
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


def process_image(
    image_path: Path,
    args: argparse.Namespace,
    base: Path,
    detector,
    scene_model,
) -> str:
    """Run the full Workflow B pipeline for one image and return the output JSON text."""

    scene = scene_model.predict(
        image_path=image_path,
        topk=5,
    )

    effective_vocab_mode = args.vocab_mode if args.legacy_visual_core else "audioset"

    raw_detections = []

    for batch in iter_grounding_prompt_batches(
        scene_label=scene["label"],
        vocab_mode=effective_vocab_mode,
    ):
        batch_detections = detector.predict(
            image_path=image_path,
            prompt=batch["prompt"],
            box_threshold=batch["box_threshold"] if args.box_threshold is None else args.box_threshold,
            text_threshold=batch["text_threshold"] if args.text_threshold is None else args.text_threshold,
        )

        if args.verbose:
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
        nms_iou_threshold=args.nms_iou_threshold,
        semantic_iou_threshold=0.30,
        semantic_containment_threshold=0.65,
    )

    if args.verbose:
        print("FILTERED DETECTIONS")
        for d in detections:
            print(d.label, round(d.confidence, 3), [round(x, 1) for x in d.bbox])

    with Image.open(image_path) as img:
        width, height = img.size

    if(args.legacy_visual_core):
        core, extended = build_from_modules(
            image_id=image_path.stem,
            width=width,
            height=height,
            detections=detections,
            scene_label=scene["label"],
            scene_conf=scene["confidence"],
            hoi=[],
        )
    else:
        core, extended = project_detections_to_audioset(
            detections=detections,
            scene_label=scene["label"],
            scene_confidence=scene["confidence"],
            width=width,
            height=height,
            image_id=image_path.stem,
        )

    if(args.legacy_visual_core):
        core.scene.indoor_outdoor = infer_indoor_outdoor_from_scene(scene["label"])

        raw_hois = []
        if args.hoi == "dummy" and args.legacy_visual_core:
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

        if args.verbose:
            print("RAW HOIS")
            for h in raw_hois:
                print(
                    h.verb,
                    round(h.confidence, 4),
                    "human_bbox=", [round(x, 1) for x in h.human_bbox],
                    "object_bbox=", [round(x, 1) for x in h.object_bbox],
                )

        core.caption = build_caption(
            scene_label=core.scene.label,
            entities=core.entities,
            interactions=core.observed_interactions,
            spatial_relations=[r.model_dump() for r in core.spatial_relations],
            indoor_outdoor=core.scene.indoor_outdoor,
            entities_extended=extended.model_dump()["entities_extended"]
        )

    metadata = {
        "workflow": "B",
        "detector": args.detector,
        "vocab_mode": effective_vocab_mode,
        "scene_model": scene,
        "box_threshold": batch["box_threshold"] if args.box_threshold is None else args.box_threshold,
        "text_threshold": batch["text_threshold"] if args.text_threshold is None else args.text_threshold,
        "min_confidence": args.min_confidence,
        "nms_iou_threshold": args.nms_iou_threshold,
        "n_raw_detections": len(raw_detections),
        "n_filtered_detections": len(detections),
    }
    if args.legacy_visual_core:
        metadata["hoi_backend"]=args.hoi,
        metadata["raw_hoi_count"]=len(raw_hois)
    else:
        metadata["ontology_mode"]="dag"

    output = {
        "core": core.model_dump(),
        "extended": extended.model_dump(),
        "metadata": metadata
    }

    return json.dumps(output, indent=2, ensure_ascii=False)


def write_output(output_text: str, image_path: Path, args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        if(args.legacy_visual_core):
            output_dir = output_dir / "legacy"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_name = args.output_name or f"{image_path.stem}.json"
        output_path = output_dir / output_name

        output_path.write_text(output_text, encoding="utf-8")
        print(f"[OK] JSON saved to: {output_path}")
    else:
        print(output_text)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="/home/jovyan/projects/data/test.jpg",
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory with input images. Processes every *.jpg in it instead of --image.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only used with --image-dir: cap the number of images processed.",
    )
    parser.add_argument(
        "--detector",
        choices=["grounding_dino", "owlv2"],
        default="grounding_dino",
        help="Open-vocabulary detector backend.",
    )
    parser.add_argument(
        "--vocab-mode",
        choices=["legacy", "audioset", "hybrid"],
        default="legacy",
        help="Detection vocabulary mode.",
    )
    parser.add_argument(
        "--scene-architecture",
        choices=["resnet50", "densenet161"],
        default="resnet50",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--hoi",
        choices=["none", "dummy", "upt"],
        default="none",
        help="HOI backend to use. 'dummy' is only for fusion smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where the generated JSON will be saved. If omitted, only prints to stdout.",
    )

    parser.add_argument(
        "--output-name",
        default=None,
        help="Optional output JSON filename. Defaults to image stem + '.json'.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    parser.add_argument(
        "--legacy-visual-core",
        action="store_true",
        help="Uses the legacy visual core instead of the current audioset core"
    )
    
    args = parser.parse_args()

    base = Path("/home/jovyan/projects")

    if args.detector == "grounding_dino":
        detector = GroundingDINOAdapter(
            config_path=base / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            checkpoint_path=base / "models/groundingdino_swint_ogc.pth",
        )

    elif args.detector == "owlv2":
        detector = OWLv2Adapter(
            model_id="google/owlv2-base-patch16-ensemble",
            device="cuda",
        )

    else:
        raise ValueError(f"Unsupported detector: {args.detector}")

    scene_model = Places365Adapter(
        architecture=args.scene_architecture,
        categories_path=base / "places365/categories_places365.txt",
        checkpoint_path=base
        / "models"
        / "places365"
        / f"{args.scene_architecture}_places365.pth.tar",
    )

    if args.image_dir:
        if args.output_name:
            print("[WARN] --output-name is ignored with --image-dir; using each image's stem instead.")
            args.output_name = None

        image_paths = sorted(Path(args.image_dir).glob("*.jpg"))

        if args.limit is not None:
            image_paths = image_paths[:args.limit]

        for image_path in image_paths:
            print(f"Processing {image_path.name}")

            try:
                output_text = process_image(image_path, args, base, detector, scene_model)
                write_output(output_text, image_path, args)
            except Exception as exc:
                print(f"FAILED: {image_path.name} -> {exc}")

    else:
        image_path = Path(args.image)
        output_text = process_image(image_path, args, base, detector, scene_model)
        write_output(output_text, image_path, args)


if __name__ == "__main__":
    main()
