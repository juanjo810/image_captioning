from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow_b.places365_adapter import Places365Adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=["resnet50", "densenet161"],
        default="resnet50",
    )
    parser.add_argument(
        "--image",
        default="/home/jovyan/projects/data/test.jpg",
    )

    args = parser.parse_args()

    base = Path("/home/jovyan/projects")

    checkpoint_path = (
        base / "models" / "places365" / f"{args.architecture}_places365.pth.tar"
    )

    adapter = Places365Adapter(
        architecture=args.architecture,
        categories_path=base / "places365/categories_places365.txt",
        checkpoint_path=checkpoint_path,
    )

    result = adapter.predict(
        image_path=args.image,
        topk=5,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()