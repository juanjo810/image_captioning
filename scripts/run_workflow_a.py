from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow_a.adapters.llamacpp_server_adapter import LlamaCppServerAdapter
from src.workflow_a.pipeline import WorkflowAPipeline


SUPPORTED_MODELS = {
    "llamacpp": LlamaCppServerAdapter,
}


DEFAULT_MODEL_IDS = {
    "llamacpp": "local-vlm",
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        help="Input image path.",
    )

    parser.add_argument(
        "--image-dir",
        help="Directory with input images.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--model-id",
        default=None,
    )

    parser.add_argument(
        "--server-url",
        default="http://localhost:8889",
        help="llama.cpp OpenAI-compatible server URL.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/workflow_a",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--include-audioset-nodes",
        action="store_true",
        help=(
            "Ask the VLM to add a top-level acoustic_semantics section with "
            "visually inferred AudioSet ontology nodes." 
            "Only works if --legacy-visual-core is passed!"
        )
    )

    parser.add_argument(
        "--legacy-visual-core",
        action="store_true",
        help="Uses the legacy visual core instead of the current audioset core."
    )

    parser.add_argument(
        "--call-mode",
        choices=["single", "three", "two"],
        default="single",
        help=(
            "Only meaningful without --legacy-visual-core. 'single' (default) asks "
            "for scene/visual_terms/nodes/caption in one VLM call (two stages inside "
            "the same response). 'three' issues three independent VLM calls (scene+"
            "visual_terms, then nodes, then caption), losing the autoregressive "
            "conditioning between stages -- kept for experimentation/comparison. "
            "'two' issues two calls: phase 1 (image attached) asks for a free-form "
            "scene description + visual_terms/nodes/caption in one response, then "
            "phase 2 (text-only) maps that free description onto the Places365 "
            "allow-list -- meant to fix low scene-label diversity/invalid labels "
            "from giving the model the allow-list while it's still looking at the "
            "image."
        ),
    )

    args = parser.parse_args()

    model_id = args.model_id or "local-vlm"

    adapter = LlamaCppServerAdapter(
        model_id=model_id,
        base_url=args.server_url,
    )

    pipeline = WorkflowAPipeline(adapter)

    if args.image_dir:
        image_paths = sorted(Path(args.image_dir).glob("*.jpg"))

        if args.limit is not None:
            image_paths = image_paths[:args.limit]

        for image_path in image_paths:
            print(f"Processing {image_path.name}")

            try:
                pipeline.run(
                    image_path=image_path,
                    output_dir=args.output_dir,
                    max_new_tokens=args.max_new_tokens,
                    include_audioset_nodes=args.include_audioset_nodes,
                    use_legacy_core=args.legacy_visual_core,
                    call_mode=args.call_mode,
                )
            except Exception as exc:
                print(f"FAILED: {image_path.name} -> {exc}")

    else:
        pipeline.run(
            image_path=args.image,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            include_audioset_nodes=args.include_audioset_nodes,
            use_legacy_core=args.legacy_visual_core,
            call_mode=args.call_mode,
        )


if __name__ == "__main__":
    main()
