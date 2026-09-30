from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import CLIPModel, CLIPProcessor

from scripts.evaluation.vg_utils import canonicalize, is_audioset_core_json, load_alias_map


METRIC_CHOICES = ("cider", "spice", "clipscore", "chair")
GROUP_FIELDS = ("detector", "scene_model", "captioner")

# The VG object vocabulary is built from free-text annotations, so it contains
# function words that some annotator used as an object name (e.g. "an",
# "there", "two"). Matched against a caption, those turn every article into a
# "hallucinated object". Stopwords that are also plausible object names are
# kept, as are the words the caption template itself contributes ("image",
# "scene", "shows"), which are never object mentions.
OBJECT_LIKE_STOPWORDS = frozenset(
    {"back", "bottom", "can", "fire", "front", "side", "top", "bill", "mill"}
)
# "indoor"/"outdoor"/"mixed" are the indoor_outdoor values
# captioning._scene_phrase inserts ("an {indoor_outdoor} {label} scene").
# "show" is "shows" after singularize_phrase().
TEMPLATE_WORDS = frozenset(
    {"image", "scene", "shows", "show", "indoor", "outdoor", "mixed"}
)
NON_OBJECT_WORDS = (
    frozenset(ENGLISH_STOP_WORDS) - OBJECT_LIKE_STOPWORDS
) | TEMPLATE_WORDS

# Singular/plural folding for CHAIR. Caption text, the VG object vocabulary
# and the reference objects (VG annotations and visual_terms) all go through
# singularize_phrase() before being compared, so "buses" in a caption matches
# "bus" in the references and "person" in a caption matches "people" in VG.
# The same function runs on every side, so a word it folds imperfectly (e.g.
# "glasses" -> "glass") is still folded identically everywhere. This is
# morphology only, not synonymy: "man" still does not match "person".
#
# Plural -> singular forms the suffix rules in singularize_word() would get
# wrong. Every one of the 210 caption-visible visual terms (the only nouns a
# caption can name) was checked against an independent plural oracle; the
# rest come from singular/plural pairs in the VG object alias file or are
# common cases of the same patterns.
IRREGULAR_SINGULARS = {
    # Irregular plurals.
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "geese": "goose",
    "teeth": "tooth",
    "feet": "foot",
    "oxen": "ox",
    "cacti": "cactus",
    "fungi": "fungus",
    # -ves plurals of -f/-fe nouns. The default rule would give "knive"; it is
    # right for -ve nouns such as dove or wave, so those are not listed.
    "knives": "knife",
    "wives": "wife",
    "lives": "life",
    "leaves": "leaf",
    "loaves": "loaf",
    "halves": "half",
    "shelves": "shelf",
    "wolves": "wolf",
    "calves": "calf",
    "thieves": "thief",
    "scarves": "scarf",
    "hooves": "hoof",
    "elves": "elf",
    # -oes plurals of -o nouns. The default rule would give "mosquitoe"; it is
    # right for -oe nouns such as canoe or oboe, so those are not listed.
    "banjoes": "banjo",
    "mosquitoes": "mosquito",
    "volcanoes": "volcano",
    "velcroes": "velcro",
    "potatoes": "potato",
    "tomatoes": "tomato",
    "heroes": "hero",
    "echoes": "echo",
    "torpedoes": "torpedo",
    "buffaloes": "buffalo",
    "dominoes": "domino",
    "mangoes": "mango",
    "flamingoes": "flamingo",
    "zeroes": "zero",
    # -ses/-zes plurals of nouns ending in -s/-z.
    "buses": "bus",
    "busses": "bus",
    "gases": "gas",
    "lenses": "lens",
    "atlases": "atlas",
    "canvases": "canvas",
    "christmases": "christmas",
    "thermoses": "thermos",
    "platypuses": "platypus",
    "daises": "dais",
    "viruses": "virus",
    "cactuses": "cactus",
    "octopuses": "octopus",
    "walruses": "walrus",
    "circuses": "circus",
    "irises": "iris",
    "quizzes": "quiz",
    "waltzes": "waltz",
    # -ies plurals of -ie nouns (the default rule gives "-y").
    "movies": "movie",
    "cookies": "cookie",
    "pies": "pie",
    "ties": "tie",
    "lies": "lie",
    "brownies": "brownie",
    "zombies": "zombie",
    "hoodies": "hoodie",
    "selfies": "selfie",
    "calories": "calorie",
    "veggies": "veggie",
    "neckties": "necktie",
    "bowties": "bowtie",
    "crossties": "crosstie",
    "twinkies": "twinkie",
    "freebies": "freebie",
    "weenies": "weenie",
    # -ches/-sses plurals of -che/-sse nouns (the default rule drops "es").
    "aches": "ache",
    "quiches": "quiche",
    "crevasses": "crevasse",
    "headaches": "headache",
    "moustaches": "moustache",
    "mustaches": "mustache",
    "avalanches": "avalanche",
    "niches": "niche",
    "caches": "cache",
    # Plurals of nouns ending in -u, which the -us guard would keep.
    "menus": "menu",
    "emus": "emu",
    "gnus": "gnu",
    "tutus": "tutu",
    "plateaus": "plateau",
}
# Nouns ending in "s" that are singular (or the same in both numbers) and are
# not already protected by the -ss/-us guard. Words ending in -is are NOT
# guarded by default: in the VG alias file they are mostly plurals of -i
# nouns (taxis, skis, kiwis, paninis, bikinis...), so the few singular -is
# nouns are listed here instead.
INVARIANT_NOUNS = frozenset(
    {
        "lens",
        "gas",
        "atlas",
        "canvas",
        "christmas",
        "news",
        "series",
        "species",
        "bias",
        "chaos",
        "cosmos",
        "rhinoceros",
        "thermos",
        "tennis",
        "iris",
        "dais",
        "axis",
        "chassis",
        "trellis",
        "pelvis",
        "oasis",
    }
)


def singularize_word(word: str) -> str:
    if word in IRREGULAR_SINGULARS:
        return IRREGULAR_SINGULARS[word]
    if word in INVARIANT_NOUNS or len(word) <= 3 or not word.endswith("s"):
        return word
    # glass, bass and "-us" nouns (cactus, walrus) are singular.
    if word.endswith(("ss", "us")):
        return word
    # glasses, boxes, churches, dishes -> drop "es".
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    # babies, flies -> "-y".
    if word.endswith("ies"):
        return word[:-3] + "y"
    # Everything else drops the "s": cars, horses, sneezes, canoes, doves.
    return word[:-1]


def singularize_phrase(phrase: str) -> str:
    """Singularizes every word, not just the last one, so the result is the
    same no matter which side of the comparison a phrase comes from."""
    return " ".join(singularize_word(word) for word in phrase.split())

FIELDNAMES = [
    "detector",
    "scene_model",
    "captioner",
    "CIDEr",
    "SPICE",
    "SPICE_P",
    "SPICE_R",
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


def load_json_entities(json_path: Path, object_alias: dict[str, str]) -> set[str] | None:
    """Returns the canonicalized core.visual_terms set for AudioSetCoreJSON
    predictions, or None (not just an empty set) when that field is genuinely
    absent -- a prediction generated before visual_terms existed -- so callers
    can tell "no data in this schema" apart from "real run, zero visual
    terms". For legacy CoreJSON predictions, returns the canonicalized
    core.entities labels as before."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    core = data.get("core", {})

    if is_audioset_core_json(data):
        if "visual_terms" not in core:
            return None
        return {
            canonicalize(term, object_alias)
            for term in core["visual_terms"]
            if term
        }

    entities = core.get("entities", [])

    return {
        canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("label")
    }


def build_object_vocabulary(
    vg_refs: dict[str, set[str]],
    object_alias: dict[str, str],
) -> set[str]:
    """Returns the singularized VG object names a caption is searched for."""
    vocab = set()

    for objects in vg_refs.values():
        for obj in objects:
            canonical = canonicalize(obj, object_alias)
            singular = singularize_phrase(canonical)
            if not singular or {canonical, singular} & NON_OBJECT_WORDS:
                continue
            # Multi-word annotations built on template words ("outdoor
            # scene") would match the template itself, not an object.
            if TEMPLATE_WORDS & set(singular.split()):
                continue
            vocab.add(singular)

    return vocab


def strip_scene_label(caption_norm: str, scene_label: str) -> str:
    """Removes the scene label the caption template inserts ("The image shows
    an outdoor street scene with ...") so it is not scored as an object
    mention: CHAIR measures object hallucination, and the scene is evaluated
    separately by evaluate_scene_awareness.py. Places365 labels such as
    "church/outdoor" keep only the part before the slash. Expects a caption
    already passed through singularize_phrase(); the label is folded the same
    way before matching."""
    label_norm = singularize_phrase(
        normalize_text(scene_label.split("/")[0].replace("_", " "))
    )

    if not label_norm:
        return caption_norm

    # Replaced by a separator, not a space, so the words on either side
    # cannot join into a new multi-word match.
    pattern = r"\b" + re.escape(label_norm) + r"\b"
    return re.sub(pattern, " | ", caption_norm)


def normalize_caption_segments(caption: str) -> str:
    """normalize_text() drops punctuation, which would let a multi-word
    object match across a list separator ("a computer, keyboard" ->
    "computer keyboard"). Normalizing each punctuation-delimited segment
    separately and joining them with a non-word separator keeps matches
    inside a single segment."""
    segments = re.split(r"[,.;:!?()]", caption)
    return " | ".join(normalize_text(segment) for segment in segments)


def extract_caption_objects(
    caption: str,
    object_vocab: set[str],
    scene_label: str = "",
) -> set[str]:
    """Returns the vocabulary entries (singularized, see
    build_object_vocabulary) the caption mentions. The caption is singularized
    word by word first, so "two buses" is found as "bus"."""
    caption_norm = strip_scene_label(
        singularize_phrase(normalize_caption_segments(caption)), scene_label
    )
    found = set()

    # Longest entries first (then alphabetical, so ties are deterministic).
    for obj in sorted(object_vocab, key=lambda x: (-len(x.split()), x)):
        pattern = r"\b" + re.escape(obj) + r"\b"

        if re.search(pattern, caption_norm):
            found.add(obj)
            # Consume the matched span so a shorter entry nested inside it
            # ("keyboard" inside "computer keyboard") is not counted again.
            caption_norm = re.sub(pattern, " | ", caption_norm)

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
        truncation=True,
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
    metrics: set[str],
) -> dict[str, float | int]:
    """Computes whichever of CIDEr/SPICE is in `metrics`. References and
    candidates both go through PTBTokenizer first (the standard coco-caption
    pipeline): without it the candidate keeps its capitals and punctuation
    while the references do not, so "person," never matches "person"."""
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gts = {}
    res = {}

    for idx, row in enumerate(rows):
        image_id = row["image_id"]

        if image_id not in refs_by_image:
            continue

        key = str(idx)
        gts[key] = [{"caption": ref} for ref in refs_by_image[image_id]]
        res[key] = [{"caption": row["caption"]}]

    scores: dict[str, float | int] = {"n_caption_refs": len(gts)}

    if not gts:
        if "cider" in metrics:
            scores["CIDEr"] = 0.0
        if "spice" in metrics:
            scores["SPICE"] = 0.0
            scores["SPICE_P"] = ""
            scores["SPICE_R"] = ""
        return scores

    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    if "cider" in metrics:
        cider_score, _ = Cider().compute_score(gts, res)
        scores["CIDEr"] = float(cider_score)

    if "spice" in metrics:
        spice_score, per_image = Spice().compute_score(gts, res)
        scores["SPICE"] = float(spice_score)
        # SPICE is an F1 against the union of every reference's scene graph.
        # With dozens of region descriptions per image that union holds
        # hundreds of tuples, so recall stays near zero even when most of the
        # caption's tuples are correct -- precision and recall are reported
        # separately so the F1 can be read correctly.
        scores["SPICE_P"] = _mean_ignoring_nan(s["All"]["pr"] for s in per_image)
        scores["SPICE_R"] = _mean_ignoring_nan(s["All"]["re"] for s in per_image)

    return scores


def _mean_ignoring_nan(values) -> float | str:
    kept = [float(v) for v in values if not math.isnan(float(v))]
    return sum(kept) / len(kept) if kept else ""


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

    # Empty cell, not 0.0, when no image was found: a wrong --image-dir must
    # not look like a real (very bad) score.
    return {
        "CLIPScore": sum(scores) / len(scores) if scores else "",
        "n_clipscore": len(scores),
    }


def evaluate_chair_group(
    rows: list[dict[str, str]],
    vg_refs: dict[str, set[str]],
    object_vocab: set[str],
    object_alias: dict[str, str],
) -> dict[str, float | int | str]:
    """CHAIRi_VG is computed for every row (VG references are external ground
    truth, independent of the prediction's own schema). CHAIRi_JSON compares
    the caption against core.entities for legacy CoreJSON predictions, or
    against core.visual_terms for AudioSetCoreJSON predictions -- both are
    the JSON's own account of what's visually present, just under different
    field names. load_json_entities() returns None only for predictions that
    predate whichever field applies to their schema (no core.entities, or an
    AudioSetCoreJSON with no visual_terms key at all), and those rows are
    excluded from the JSON-hallucination average instead of being counted as
    100% hallucinated against a list that was never there to begin with.

    Captions that mention no object at all are excluded from both CHAIRi
    averages (0 hallucinated out of 0 mentioned is undefined, not a perfect
    0.0); n_chair counts the captions that did contribute. The avg_* columns
    still average over every row."""
    vg_items = []
    json_items = []

    for row in rows:
        image_id = row["image_id"]
        caption = row["caption"]
        json_path = Path(row["json_path"])

        caption_objects = extract_caption_objects(
            caption=caption,
            object_vocab=object_vocab,
            scene_label=row.get("scene_label", ""),
        )

        vg_objects = {singularize_phrase(o) for o in vg_refs.get(image_id, set())}
        chair_i_vg, halluc_vg, mentioned = chair_scores(
            caption_objects=caption_objects,
            reference_objects=vg_objects,
        )
        vg_items.append(
            {
                "CHAIRi_VG": chair_i_vg,
                "caption_objects_mentioned": mentioned,
                "hallucinated_vg": halluc_vg,
            }
        )

        json_objects = load_json_entities(json_path, object_alias)
        if json_objects is not None:
            chair_i_json, halluc_json, _ = chair_scores(
                caption_objects=caption_objects,
                reference_objects={singularize_phrase(o) for o in json_objects},
            )
            json_items.append(
                {
                    "CHAIRi_JSON": chair_i_json,
                    "caption_objects_mentioned": mentioned,
                    "hallucinated_json": halluc_json,
                }
            )

    if not vg_items:
        return {
            "CHAIRi_VG": "",
            "CHAIRi_JSON": "",
            "avg_caption_objects_mentioned": 0.0,
            "avg_hallucinated_vg": 0.0,
            "avg_hallucinated_json": "",
            "n_chair": 0,
        }

    n = len(vg_items)
    n_json = len(json_items)
    vg_scored = [x for x in vg_items if x["caption_objects_mentioned"] > 0]
    json_scored = [x for x in json_items if x["caption_objects_mentioned"] > 0]

    return {
        "CHAIRi_VG": (
            sum(x["CHAIRi_VG"] for x in vg_scored) / len(vg_scored) if vg_scored else ""
        ),
        "CHAIRi_JSON": (
            sum(x["CHAIRi_JSON"] for x in json_scored) / len(json_scored)
            if json_scored
            else ""
        ),
        "avg_caption_objects_mentioned": (
            sum(x["caption_objects_mentioned"] for x in vg_items) / n
        ),
        "avg_hallucinated_vg": sum(x["hallucinated_vg"] for x in vg_items) / n,
        "avg_hallucinated_json": (
            sum(x["hallucinated_json"] for x in json_items) / n_json if json_items else ""
        ),
        "n_chair": len(vg_scored),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRIC_CHOICES,
        default=list(METRIC_CHOICES),
        help="Metrics to compute (default: all). The output CSV only has the "
        "columns of the metrics requested.",
    )
    parser.add_argument("--caption-refs", help="Required by cider and spice.")
    parser.add_argument("--image-dir", help="Required by clipscore.")
    parser.add_argument("--vg-object-refs", help="Required by chair.")
    parser.add_argument("--object-alias", help="Required by chair.")

    parser.add_argument(
        "--clip-model-name",
        default="openai/clip-vit-base-patch16",
    )

    args = parser.parse_args()
    metrics = set(args.metrics)

    required_by_metric = {
        "cider": ["caption_refs"],
        "spice": ["caption_refs"],
        "clipscore": ["image_dir"],
        "chair": ["vg_object_refs", "object_alias"],
    }
    for metric in sorted(metrics):
        for attr in required_by_metric[metric]:
            if getattr(args, attr) is None:
                parser.error(f"--{attr.replace('_', '-')} is required by {metric}")

    predictions = load_predictions(Path(args.manifest))

    if metrics & {"cider", "spice"}:
        refs_by_image = load_caption_references(Path(args.caption_refs))

    if "chair" in metrics:
        vg_refs = load_vg_object_refs(Path(args.vg_object_refs))
        object_alias = load_alias_map(args.object_alias)
        object_vocab = build_object_vocabulary(vg_refs, object_alias)

    if "clipscore" in metrics:
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
        out_row: dict[str, Any] = {
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
        }

        if metrics & {"cider", "spice"}:
            out_row.update(
                evaluate_cider_spice(
                    rows=rows,
                    refs_by_image=refs_by_image,
                    metrics=metrics,
                )
            )

        if "clipscore" in metrics:
            out_row.update(
                evaluate_clipscore_group(
                    rows=rows,
                    image_dir=Path(args.image_dir),
                    model=clip_model,
                    processor=clip_processor,
                    device=device,
                )
            )

        if "chair" in metrics:
            out_row.update(
                evaluate_chair_group(
                    rows=rows,
                    vg_refs=vg_refs,
                    object_vocab=object_vocab,
                    object_alias=object_alias,
                )
            )

        out_rows.append(out_row)

    fieldnames = [
        key
        for key in FIELDNAMES
        if key in GROUP_FIELDS or any(key in row for row in out_rows)
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in out_rows)

    print(f"[OK] Unified caption metrics saved to {output_path}")


if __name__ == "__main__":
    main()
