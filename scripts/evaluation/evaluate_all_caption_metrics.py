from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from scripts.evaluation.vg_utils import canonicalize, load_alias_map


FIELDNAMES = [
    "detector",
    "scene_model",
    "captioner",
    "CIDEr",
    "SPICE",
    "CLIPScore",
    "CHAIRi_VG",
    "CHAIRi_JSON",
    "avg_caption_objects_mentioned",
    "avg_hallucinated_vg",
    "avg_hallucinated_json",
    "n_caption_refs",
    "n_clipscore",
    "n_chair",
]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_predictions(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_caption_references(path: Path) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]].append(row["reference_caption"])

    return dict(refs)


def load_vg_object_refs(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            refs[row["image_id"]] = {
                obj for obj in row["objects"].split("|") if obj
            }

    return refs


def load_json_entities(json_path: Path, object_alias: dict[str, str]) -> set[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    entities = data.get("core", {}).get("entities", [])

    return {
        canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("label")
    }


def build_object_vocabulary(
    vg_refs: dict[str, set[str]],
    object_alias: dict[str, str],
) -> set[str]:
    vocab = set()

    for objects in vg_refs.values():
        for obj in objects:
            canonical = canonicalize(obj, object_alias)
            if canonical:
                vocab.add(canonical)

    return vocab


def extract_caption_objects(
    caption: str,
    object_vocab: set[str],
    object_alias: dict[str, str],
) -> set[str]:
    caption_norm = normalize_text(caption)
    found = set()

    for obj in sorted(object_vocab, key=lambda x: len(x.split()), reverse=True):
        obj_norm = normalize_text(obj)

        if not obj_norm:
            continue

        pattern = r"\b" + re.escape(obj_norm) + r"s?\b"

        if re.search(pattern, caption_norm):
            found.add(canonicalize(obj_norm, object_alias))

    return found


def chair_scores(
    caption_objects: set[str],
    reference_objects: set[str],
) -> tuple[float, int, int]:
    if not caption_objects:
        return 0.0, 0, 0

    hallucinated = caption_objects - reference_objects
    chair_i = len(hallucinated) / len(caption_objects)

    return chair_i, len(hallucinated), len(caption_objects)


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
    score = torch.clamp(100.0 * cosine, min=0.0)

    return float(score.detach().cpu().item())


def evaluate_cider_spice(
    rows: list[dict[str, str]],
    refs_by_image: dict[str, list[str]],
) -> dict[str, float | int]:
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice

    gts = {}
    res = {}

    for idx, row in enumerate(rows):
        image_id = row["image_id"]

        if image_id not in refs_by_image:
            continue

        key = str(idx)
        gts[key] = refs_by_image[image_id]
        res[key] = [row["caption"]]

    if not gts:
        return {
            "CIDEr": 0.0,
            "SPICE": 0.0,
            "n_caption_refs": 0,
        }

    cider_score, _ = Cider().compute_score(gts, res)
    spice_score, _ = Spice().compute_score(gts, res)

    return {
        "CIDEr": float(cider_score),
        "SPICE": float(spice_score),
        "n_caption_refs": len(gts),
    }


def evaluate_clipscore_group(
    rows: list[dict[str, str]],
    image_dir: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str,
) -> dict[str, float | int]:
    scores = []

    for row in rows:
        image_path = find_image_path(image_dir, row["image_id"])

        if image_path is None:
            print(f"[WARN] Image not found for image_id={row['image_id']}")
            continue

        scores.append(
            compute_clipscore(
                image_path=image_path,
                caption=row["caption"],
                model=model,
                processor=processor,
                device=device,
            )
        )

    return {
        "CLIPScore": sum(scores) / len(scores) if scores else 0.0,
        "n_clipscore": len(scores),
    }


def evaluate_chair_group(
    rows: list[dict[str, str]],
    vg_refs: dict[str, set[str]],
    object_vocab: set[str],
    object_alias: dict[str, str],
) -> dict[str, float | int]:
    items = []

    for row in rows:
        image_id = row["image_id"]
        caption = row["caption"]
        json_path = Path(row["json_path"])

        caption_objects = extract_caption_objects(
            caption=caption,
            object_vocab=object_vocab,
            object_alias=object_alias,
        )

        vg_objects = vg_refs.get(image_id, set())
        json_objects = load_json_entities(json_path, object_alias)

        chair_i_vg, halluc_vg, mentioned = chair_scores(
            caption_objects=caption_objects,
            reference_objects=vg_objects,
        )

        chair_i_json, halluc_json, _ = chair_scores(
            caption_objects=caption_objects,
            reference_objects=json_objects,
        )

        items.append(
            {
                "CHAIRi_VG": chair_i_vg,
                "CHAIRi_JSON": chair_i_json,
                "caption_objects_mentioned": mentioned,
                "hallucinated_vg": halluc_vg,
                "hallucinated_json": halluc_json,
            }
        )

    if not items:
        return {
            "CHAIRi_VG": 0.0,
            "CHAIRi_JSON": 0.0,
            "avg_caption_objects_mentioned": 0.0,
            "avg_hallucinated_vg": 0.0,
            "avg_hallucinated_json": 0.0,
            "n_chair": 0,
        }

    n = len(items)

    return {
        "CHAIRi_VG": sum(x["CHAIRi_VG"] for x in items) / n,
        "CHAIRi_JSON": sum(x["CHAIRi_JSON"] for x in items) / n,
        "avg_caption_objects_mentioned": (
            sum(x["caption_objects_mentioned"] for x in items) / n
        ),
        "avg_hallucinated_vg": sum(x["hallucinated_vg"] for x in items) / n,
        "avg_hallucinated_json": sum(x["hallucinated_json"] for x in items) / n,
        "n_chair": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--caption-refs", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch16",
    )

    args = parser.parse_args()

    predictions = load_predictions(Path(args.manifest))
    refs_by_image = load_caption_references(Path(args.caption_refs))
    vg_refs = load_vg_object_refs(Path(args.vg_object_refs))
    object_alias = load_alias_map(args.object_alias)
    object_vocab = build_object_vocabulary(vg_refs, object_alias)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading CLIP model: {args.clip_model_name}")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model_name)
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    clip_model.eval()

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)

    for row in predictions:
        condition = (
            row["detector"],
            row["scene_model"],
            row["captioner"],
        )
        grouped[condition].append(row)

    out_rows = []

    for (detector, scene_model, captioner), rows in grouped.items():
        caption_scores = evaluate_cider_spice(
            rows=rows,
            refs_by_image=refs_by_image,
        )
        clip_scores = evaluate_clipscore_group(
            rows=rows,
            image_dir=Path(args.image_dir),
            model=clip_model,
            processor=clip_processor,
            device=device,
        )
        chair = evaluate_chair_group(
            rows=rows,
            vg_refs=vg_refs,
            object_vocab=object_vocab,
            object_alias=object_alias,
        )

        out_rows.append(
            {
                "detector": detector,
                "scene_model": scene_model,
                "captioner": captioner,
                **caption_scores,
                **clip_scores,
                **chair,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDNAMES} for row in out_rows)

    print(f"[OK] Unified caption metrics saved to {output_path}")


if __name__ == "__main__":
    main()
