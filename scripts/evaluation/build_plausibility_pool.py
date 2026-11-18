"""Build the annotation package for the human plausibility study.

The automatic evaluation scores predictions against pseudo-references derived
from Visual Genome with the same leaf rules the systems use, so it cannot say
whether a predicted sound is actually plausible in the image. This script
prepares what human annotators need to judge that, blind to which system
proposed what:

- for each selected image, the *pool* of candidate sounds: every AudioSet node
  predicted by any of the compared systems, plus the nodes of the
  pseudo-reference, plus a few random sounds from the study vocabulary (a
  chance baseline: how often a random pool sound is judged plausible);
- for each selected image, the visual terms any system cited as evidence, plus
  a random term from the study vocabulary, so annotators can say whether each
  one is really visible (evidence fidelity).

Outputs, under ``--output-dir``:

``images/``                    copies of the selected images
``annotate_main.html``         self-contained annotation page (open in a browser)
``annotate_pilot.html``        the same for a small pilot on *different* images
``study_main.json`` / ``study_pilot.json``
                               the items embedded in each page (no system names)
``key_main.json`` / ``key_pilot.json``
                               PRIVATE: which systems predicted each node, with
                               which evidence terms; which nodes are reference
                               or random. Needed for the analysis; never give it
                               to annotators.

Images are sampled with a fixed seed, keeping the share of images whose
pseudo-reference is empty. The page shuffles images and items per annotator, so
the order in the JSON carries no information.

Example::

    python -m scripts.evaluation.build_plausibility_pool \\
      --manifest outputs/manifests/3000/predictions_manifest_3000.csv \\
      --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \\
      --image-ids outputs/seeds_500/image_ids.txt \\
      --image-dir ../data/subsets/vg_5000_images \\
      --n-images 300 --n-pilot 30 --seed 0 \\
      --output-dir outputs/plausibility_study
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SYSTEMS = {
    "qwen_deterministic": "unknown/workflow_a_qwen36_discard_call4rules_deterministic",
    "gemma_deterministic": "unknown/workflow_a_gemma4_discard_call4rules_deterministic",
    "owlv2": "owlv2/densenet161",
    "grounding_dino": "grounding_dino/densenet161",
    "qwen_vlm_rules": "unknown/workflow_a_qwen36_discard_call4rules",
    "gemma_vlm_rules": "unknown/workflow_a_gemma4_discard_call4rules",
    "qwen_vlm_free": "unknown/workflow_a_qwen36_discard",
    "gemma_vlm_free": "unknown/workflow_a_gemma4_discard",
}
PLACEHOLDER = "__STUDY_JSON__"


def default_template_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "plausibility_annotation_template.html"


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ontology.json"


def parse_systems(values: list[str]) -> dict[str, tuple[str, str]]:
    """NAME=detector/scene_model -> {NAME: (detector, scene_model)}"""
    spec = dict(DEFAULT_SYSTEMS)
    if values:
        spec = {}
        for value in values:
            if "=" not in value or "/" not in value.split("=", 1)[1]:
                raise ValueError(f"--system expects NAME=detector/scene_model, got {value!r}")
            name, target = value.split("=", 1)
            spec[name] = target
    out = {}
    for name, target in spec.items():
        detector, scene_model = target.split("/", 1)
        out[name] = (detector, scene_model)
    return out


def stratified_sample(empty: list[str], nonempty: list[str], n: int, rng: random.Random) -> list[str]:
    total = len(empty) + len(nonempty)
    if n > total:
        raise ValueError(f"asked for {n} images but only {total} are available")
    n_empty = min(len(empty), round(n * len(empty) / total)) if total else 0
    n_empty = max(n_empty, n - len(nonempty))
    return sorted(rng.sample(empty, n_empty) + rng.sample(nonempty, n - n_empty))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, help="Prediction manifest CSV. Repeatable.")
    parser.add_argument("--per-image", required=True,
                        help="AudioSet per-image CSV (gives each image's pseudo-reference nodes).")
    parser.add_argument("--image-ids", default=None,
                        help="Text file with the candidate image ids, one per line. Default: every image in --per-image.")
    parser.add_argument("--image-dir", required=True, help="Directory holding <image_id>.jpg.")
    parser.add_argument("--n-images", type=int, default=300)
    parser.add_argument("--n-pilot", type=int, default=30, help="Pilot images, disjoint from the main ones.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--system", action="append", default=[],
                        help="NAME=detector/scene_model. Repeatable. Default: the eight systems of the paper.")
    parser.add_argument("--reference-system", default="owlv2/densenet161",
                        help="detector/scene_model whose rows are read for the pseudo-reference (it is the same for all).")
    parser.add_argument("--n-random-sounds", type=int, default=2)
    parser.add_argument("--n-random-objects", type=int, default=1)
    parser.add_argument("--template", default=None)
    parser.add_argument("--ontology", default=None)
    parser.add_argument("--study-id", default="plausibility_v1")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    systems = parse_systems(args.system)
    by_target = {target: name for name, target in systems.items()}
    template = Path(args.template) if args.template else default_template_path()
    template_text = template.read_text(encoding="utf-8")
    if PLACEHOLDER not in template_text:
        raise ValueError(f"{template} has no {PLACEHOLDER} placeholder")
    ontology = {
        node["id"]: node
        for node in json.loads((Path(args.ontology) if args.ontology else default_ontology_path()).read_text(encoding="utf-8"))
    }

    # ---- pseudo-reference of every candidate image --------------------------
    ref_detector, ref_scene_model = args.reference_system.split("/", 1)
    reference: dict[str, set[str]] = {}
    with open(args.per_image, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["detector"] == ref_detector and row["scene_model"] == ref_scene_model:
                reference[row["image_id"]] = {t for t in (row.get("gt_audioset_tags") or "").split("|") if t}
    if not reference:
        raise ValueError(f"no rows for {args.reference_system} in {args.per_image}")

    if args.image_ids:
        candidates = [line.strip() for line in Path(args.image_ids).read_text(encoding="utf-8").splitlines() if line.strip()]
        unknown = [i for i in candidates if i not in reference]
        if unknown:
            raise ValueError(f"{len(unknown)} ids of --image-ids have no pseudo-reference row, e.g. {unknown[:5]}")
    else:
        candidates = sorted(reference)
    image_dir = Path(args.image_dir)
    missing = [i for i in candidates if not (image_dir / f"{i}.jpg").is_file()]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} candidate images are not in {image_dir}, e.g. {missing[:5]}")

    # ---- selection: main and pilot, disjoint, same share of empty references -
    rng = random.Random(args.seed)
    empty = sorted(i for i in candidates if not reference[i])
    nonempty = sorted(i for i in candidates if reference[i])
    main_ids = stratified_sample(empty, nonempty, args.n_images, rng)
    taken = set(main_ids)
    pilot_ids = stratified_sample(
        [i for i in empty if i not in taken], [i for i in nonempty if i not in taken], args.n_pilot, rng
    ) if args.n_pilot else []
    selected = set(main_ids) | set(pilot_ids)

    # ---- predictions of every system on the selected images -----------------
    predictions: dict[str, dict[str, dict[str, list[str]]]] = {name: {} for name in systems}
    for manifest in args.manifest:
        with open(manifest, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = by_target.get((row["detector"], row["scene_model"]))
                if name is None or row["image_id"] not in selected:
                    continue
                core = json.loads(Path(row["json_path"]).read_text(encoding="utf-8")).get("core", {})
                nodes: dict[str, list[str]] = {}
                for node in core.get("nodes") or []:
                    node_id = node.get("audioset_id")
                    if node_id:
                        merged = set(nodes.get(node_id, [])) | set(node.get("visual_evidence_terms") or [])
                        nodes[node_id] = sorted(merged)
                predictions[name][row["image_id"]] = nodes
    for name in systems:
        if not predictions[name]:
            raise ValueError(f"system {name} ({'/'.join(systems[name])}) has no prediction on the selected images")

    # ---- pools ---------------------------------------------------------------
    def pool_of(image_id: str) -> tuple[set[str], set[str]]:
        sounds = set(reference[image_id])
        objects: set[str] = set()
        for name in systems:
            for node_id, evidence in predictions[name].get(image_id, {}).items():
                sounds.add(node_id)
                objects.update(evidence)
        return sounds, objects

    pools = {image_id: pool_of(image_id) for image_id in sorted(selected)}
    sound_vocab = sorted({s for sounds, _ in pools.values() for s in sounds})
    object_vocab = sorted({o for _, objects in pools.values() for o in objects})
    unknown_nodes = [s for s in sound_vocab if s not in ontology]
    if unknown_nodes:
        raise ValueError(f"nodes missing from the ontology: {unknown_nodes[:5]}")

    def build(ids: list[str], suffix: str) -> tuple[dict[str, Any], dict[str, Any]]:
        study_images, key_images = [], {}
        for image_id in ids:
            sounds, objects = pools[image_id]
            random_sounds = rng.sample([s for s in sound_vocab if s not in sounds],
                                       min(args.n_random_sounds, len(sound_vocab) - len(sounds)))
            random_objects = rng.sample([o for o in object_vocab if o not in objects],
                                        min(args.n_random_objects, len(object_vocab) - len(objects)))
            shown_sounds = sorted(sounds | set(random_sounds), key=lambda s: ontology[s]["name"])
            study_images.append({
                "image_id": image_id,
                "file": f"images/{image_id}.jpg",
                "objects": sorted(objects | set(random_objects)),
                "sounds": [
                    {"id": s, "name": ontology[s]["name"], "description": ontology[s].get("description", "")}
                    for s in shown_sounds
                ],
            })
            key_images[image_id] = {
                "reference_nodes": sorted(reference[image_id]),
                "random_sounds": sorted(random_sounds),
                "random_objects": sorted(random_objects),
                "systems": {
                    name: {
                        "has_prediction": image_id in predictions[name],
                        "nodes": predictions[name].get(image_id, {}),
                    }
                    for name in systems
                },
            }
        study = {
            "study_id": f"{args.study_id}_{suffix}",
            "title": "Estudio de plausibilidad sonora" + (" (piloto)" if suffix == "pilot" else ""),
            "images": study_images,
        }
        key = {
            "study_id": study["study_id"],
            "seed": args.seed,
            "systems": {name: "/".join(target) for name, target in systems.items()},
            "images": key_images,
        }
        return study, key

    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    for image_id in sorted(selected):
        target = output_dir / "images" / f"{image_id}.jpg"
        if not target.exists():
            shutil.copyfile((image_dir / f"{image_id}.jpg").resolve(), target)

    system_names = set(systems) | {"/".join(t) for t in systems.values()} | {t[1] for t in systems.values()}
    summary: list[str] = []
    for ids, suffix in ((main_ids, "main"), (pilot_ids, "pilot")):
        if not ids:
            continue
        study, key = build(ids, suffix)
        public = json.dumps(study, ensure_ascii=False)
        leaked = [name for name in system_names if len(name) > 6 and name in public]
        if leaked:
            raise AssertionError(f"system names leaked into the annotator-facing data: {leaked}")
        (output_dir / f"study_{suffix}.json").write_text(json.dumps(study, ensure_ascii=False, indent=1), encoding="utf-8")
        (output_dir / f"key_{suffix}.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
        html = template_text.replace(PLACEHOLDER, public.replace("</", "<\\/"))
        (output_dir / f"annotate_{suffix}.html").write_text(html, encoding="utf-8")

        n_sounds = [len(image["sounds"]) for image in study["images"]]
        n_objects = [len(image["objects"]) for image in study["images"]]
        n_empty = sum(1 for i in ids if not reference[i])
        summary.append(
            f"{suffix.upper()}: {len(ids)} images ({n_empty} with an empty pseudo-reference)\n"
            f"   sound judgements per annotator: {sum(n_sounds)} (mean {sum(n_sounds) / len(ids):.1f} per image, max {max(n_sounds)})\n"
            f"   object checks per annotator:    {sum(n_objects)} (mean {sum(n_objects) / len(ids):.1f} per image)\n"
            f"   page: {output_dir / f'annotate_{suffix}.html'}   private key: {output_dir / f'key_{suffix}.json'}"
        )

    most_common = Counter(s for image_id in main_ids for s in pools[image_id][0]).most_common(8)
    print("\n".join(summary))
    print(f"Distinct sounds in the study vocabulary: {len(sound_vocab)}; distinct visual terms: {len(object_vocab)}")
    print("Most frequent sounds in the main set: " + "; ".join(f"{ontology[s]['name']} ({c})" for s, c in most_common))
    print(f"Images copied to {output_dir / 'images'}: {len(selected)}")
    print("Give annotators the annotate_*.html page and the images/ folder, kept side by side. Never the key_*.json files.")


if __name__ == "__main__":
    main()
