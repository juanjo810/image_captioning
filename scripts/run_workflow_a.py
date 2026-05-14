from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow_a.adapters.gemma4_adapter import Gemma4Adapter
from src.workflow_a.pipeline import WorkflowAPipeline


SUPPORTED_MODELS = {
    "gemma4": Gemma4Adapter,
}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Input image path.",
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS.keys(),
        default="gemma4",
    )

    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E4B-it",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/workflow_a",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    args = parser.parse_args()

    adapter_cls = SUPPORTED_MODELS[args.model]

    adapter = adapter_cls(
        model_id=args.model_id,
    )

    pipeline = WorkflowAPipeline(adapter)

    result = pipeline.run(
        image_path=args.image,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
