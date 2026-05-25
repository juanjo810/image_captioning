from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow_a.adapters.gemma4_adapter import Gemma4Adapter
from src.workflow_a.adapters.llamacpp_server_adapter import LlamaCppServerAdapter
from src.workflow_a.pipeline import WorkflowAPipeline


SUPPORTED_MODELS = {
    "gemma4": Gemma4Adapter,
    "llamacpp": LlamaCppServerAdapter,
}


DEFAULT_MODEL_IDS = {
    "gemma4": "google/gemma-4-E4B-it",
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
        "--model",
        choices=SUPPORTED_MODELS.keys(),
        default="llamacpp",
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
        "--temperature",
        type=float,
        default=0.0,
    )

    args = parser.parse_args()

    adapter_cls = SUPPORTED_MODELS[args.model]

    model_id = args.model_id or DEFAULT_MODEL_IDS[args.model]

    if args.model == "llamacpp":
        adapter = adapter_cls(
            model_id=model_id,
            base_url=args.server_url,
        )
    else:
        adapter = adapter_cls(
            model_id=model_id,
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
                    temperature=args.temperature,
                )
            except Exception as exc:
                print(f"FAILED: {image_path.name} -> {exc}")

    else:
        pipeline.run(
            image_path=args.image,
            output_dir=args.output_dir,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
