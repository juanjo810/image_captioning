from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_predictions(manifest_path: Path) -> list[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_image_path(image_dir: Path, image_id: str) -> Path | None:
    for suffix in [".jpg", ".jpeg", ".png"]:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


@torch.inference_mode()
def compute_clipscore(
    image_path: Path,
    caption: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> float:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=[caption],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(device)

    outputs = model(**inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    cosine = (image_features * text_features).sum(dim=-1)

    # TorchMetrics convention: max(100 * cosine, 0)
    score = torch.clamp(100.0 * cosine, min=0.0)

    return float(score.detach().cpu().item())


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model-name",
        default="openai/clip-vit-base-patch16",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading CLIP model: {args.model_name}")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    model.eval()

    predictions = load_predictions(Path(args.manifest))
    image_dir = Path(args.image_dir)

    grouped_scores = defaultdict(list)

    for row in predictions:
        image_id = row["image_id"]
        caption = row["caption"]

        image_path = find_image_path(image_dir, image_id)

        if image_path is None:
            print(f"[WARN] Image not found for image_id={image_id}")
            continue

        score = compute_clipscore(
            image_path=image_path,
            caption=caption,
            model=model,
            processor=processor,
            device=device,
        )

        condition = (
            row["detector"],
            row["scene_model"],
            row["captioner"],
        )

        grouped_scores[condition].append(score)

    rows = []

    for (detector, scene_model, captioner), scores in grouped_scores.items():
        rows.append(
            {
                "detector": detector,
                "scene_model": scene_model,
                "captioner": captioner,
                "CLIPScore": sum(scores) / len(scores),
                "n": len(scores),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "detector",
        "scene_model",
        "captioner",
        "CLIPScore",
        "n",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] CLIPScore saved to {output_path}")


if __name__ == "__main__":
    main()