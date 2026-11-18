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
  one is really visible (evidence fidelity);
- with ``--scene-per-image``, the scene labels predicted for the image by each
  scene source (the Places365 classifiers and the VLMs), so annotators can say
  whether each one describes the place. This is the human check of the scene
  step, which the automatic scene metric only approximates.

Images are drawn in one step from the evaluation set, at random with a fixed
seed, stratified with proportional allocation. The strata are whether the
pseudo-reference of the image is empty and, with ``--scene-per-image``, the
scene group predicted for the image by ``--strata-scene-system``. The pilot is
drawn the same way from the images left. The page shuffles images and items per
annotator, so the order in the JSON carries no information.

Outputs, under ``--output-dir``:

``images/``                    copies of the selected images
``annotate_main.html``         self-contained annotation page (open in a browser)
``annotate_pilot.html``        the same for a small pilot on *different* images
``study_main.json`` / ``study_pilot.json``
                               the items embedded in each page (no system names)
``key_main.json`` / ``key_pilot.json``
                               PRIVATE: which systems predicted each node, with
                               which evidence terms; which nodes are reference
                               or random; which scene source predicted each
                               label. Needed for the analysis; never give it to
                               annotators.
``sampling_strata.csv``        images per stratum in the evaluation set, in the
                               study and in the pilot

Example::

    python -m scripts.evaluation.build_plausibility_pool \\
      --manifest outputs/manifests/3000/predictions_manifest_3000.csv \\
      --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \\
      --scene-per-image outputs/final_metrics/stats_3000/scene_per_image.csv \\
      --image-dir ../data/subsets/vg_5000_images \\
      --n-images 300 --n-pilot 30 --seed 0 \\
      --output-dir outputs/plausibility_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable


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
# Who predicts the scene label: the two Places365 classifiers of Workflow B and
# the two VLMs of Workflow A. The detector does not matter for the scene label.
DEFAULT_SCENE_SOURCES = {
    "places365_densenet161": "owlv2/densenet161",
    "places365_resnet50": "owlv2/resnet50",
    "gemma4": "unknown/workflow_a_gemma4_discard",
    "qwen36": "unknown/workflow_a_qwen36_discard",
}
PLACEHOLDER = "__STUDY_JSON__"


def default_template_path() -> Path:
    return Path(__file__).resolve().parents[2] / "resources" / "plausibility_annotation_template.html"


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[2] / "ontology.json"


def parse_targets(values: list[str], defaults: dict[str, str], option: str) -> dict[str, tuple[str, str]]:
    """NAME=detector/scene_model -> {NAME: (detector, scene_model)}"""
    spec = dict(defaults)
    if values:
        spec = {}
        for value in values:
            if "=" not in value or "/" not in value.split("=", 1)[1]:
                raise ValueError(f"{option} expects NAME=detector/scene_model, got {value!r}")
            name, target = value.split("=", 1)
            spec[name] = target
    out = {}
    for name, target in spec.items():
        detector, scene_model = target.split("/", 1)
        out[name] = (detector, scene_model)
    return out


def normalize_scene_label(label: str) -> str:
    """Same normalisation as ``src.workflow_b.vocabularies.normalize_scene_label``."""
    return label.strip().lower().replace("_", " ").replace("/", " ")


def proportional_allocation(sizes: dict[Hashable, int], n: int) -> dict[Hashable, int]:
    """Share ``n`` units among strata in proportion to their sizes (largest
    remainders; ties go to the stratum that sorts first). A stratum never gets
    more units than it has."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"asked for {n} images but only {total} are available")
    quotas = {k: n * size / total for k, size in sizes.items()}
    allocation = {k: int(q) for k, q in quotas.items()}
    order = sorted(sizes, key=lambda k: (-(quotas[k] - allocation[k]), str(k)))
    for k in order[: n - sum(allocation.values())]:
        allocation[k] += 1
    return allocation


def stratified_sample(strata: dict[Hashable, list[str]], n: int, rng: random.Random) -> list[str]:
    allocation = proportional_allocation({k: len(v) for k, v in strata.items()}, n)
    chosen: list[str] = []
    for k in sorted(strata, key=str):
        chosen.extend(rng.sample(sorted(strata[k]), allocation[k]))
    return sorted(chosen)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", action="append", required=True, help="Prediction manifest CSV. Repeatable.")
    parser.add_argument("--per-image", required=True,
                        help="AudioSet per-image CSV (gives each image's pseudo-reference nodes).")
    parser.add_argument("--scene-per-image", default=None,
                        help="Scene per-image CSV. Adds the scene group to the sampling strata and the predicted "
                             "scene labels to the items to judge.")
    parser.add_argument("--strata-scene-system", default="owlv2/densenet161",
                        help="detector/scene_model whose predicted scene group defines the strata.")
    parser.add_argument("--scene-source", action="append", default=[],
                        help="NAME=detector/scene_model whose scene label is judged. Repeatable. "
                             "Default: the two Places365 classifiers and the two VLMs.")
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
    parser.add_argument("--study-id", default="plausibility_v2")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    systems = parse_targets(args.system, DEFAULT_SYSTEMS, "--system")
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

    # ---- scene labels and groups (optional) ---------------------------------
    scene_group: dict[str, str] = {}                                     # image -> group of the strata system
    scene_sources: dict[str, tuple[str, str]] = {}
    scene_labels: dict[str, dict[str, str]] = {}                         # source -> image -> normalised label
    notes: list[str] = []
    if args.scene_per_image:
        strata_target = tuple(args.strata_scene_system.split("/", 1))
        wanted = parse_targets(args.scene_source, DEFAULT_SCENE_SOURCES, "--scene-source")
        source_of = {target: name for name, target in wanted.items()}
        scene_labels = {name: {} for name in wanted}
        seen_targets: set[tuple[str, str]] = set()
        with open(args.scene_per_image, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                target = (row["detector"], row["scene_model"])
                seen_targets.add(target)
                if target == strata_target:
                    scene_group[row["image_id"]] = row.get("scene_group") or "unknown"
                name = source_of.get(target)
                label = normalize_scene_label(row.get("scene_label") or "")
                if name is not None and label:
                    scene_labels[name][row["image_id"]] = label
        available = ", ".join(sorted("/".join(t) for t in seen_targets))
        if not scene_group:
            raise SystemExit(f"STOP: no rows for {args.strata_scene_system} in {args.scene_per_image}. Systems there: {available}")
        absent = [name for name in wanted if not scene_labels[name]]
        if absent and args.scene_source:
            raise SystemExit(f"STOP: no scene rows for {absent} in {args.scene_per_image}. Systems there: {available}")
        for name in absent:                                              # a default source the CSV does not have
            notes.append(f"NOTE: scene source {name} ({'/'.join(wanted[name])}) is not in {args.scene_per_image} and is left out. "
                         f"Systems there: {available}")
        scene_sources = {name: target for name, target in wanted.items() if scene_labels[name]}
        scene_labels = {name: scene_labels[name] for name in scene_sources}

    # ---- selection: one stratified draw for the study, another for the pilot -
    def stratum(image_id: str) -> tuple[str, str]:
        group = scene_group.get(image_id, "unknown") if args.scene_per_image else "all"
        return group, "empty" if not reference[image_id] else "non-empty"

    def strata_of(ids: list[str]) -> dict[Hashable, list[str]]:
        out: dict[Hashable, list[str]] = defaultdict(list)
        for image_id in ids:
            out[stratum(image_id)].append(image_id)
        return dict(out)

    rng = random.Random(args.seed)
    main_ids = stratified_sample(strata_of(candidates), args.n_images, rng)
    taken = set(main_ids)
    pilot_ids = stratified_sample(strata_of([i for i in candidates if i not in taken]), args.n_pilot, rng) if args.n_pilot else []
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
            labels = {name: scene_labels[name].get(image_id) for name in scene_sources}
            study_image = {
                "image_id": image_id,
                "file": f"images/{image_id}.jpg",
                "objects": sorted(objects | set(random_objects)),
                "sounds": [
                    {"id": s, "name": ontology[s]["name"], "description": ontology[s].get("description", "")}
                    for s in shown_sounds
                ],
            }
            group, emptiness = stratum(image_id)
            key_image = {
                "strata": {"scene_group": group, "empty_reference": emptiness == "empty"},
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
            if scene_sources:
                study_image["scenes"] = sorted({label for label in labels.values() if label})
                key_image["scene_labels"] = labels
            study_images.append(study_image)
            key_images[image_id] = key_image
        study = {
            "study_id": f"{args.study_id}_{suffix}",
            "title": "Estudio de plausibilidad sonora" + (" (piloto)" if suffix == "pilot" else ""),
            "images": study_images,
        }
        key = {
            "study_id": study["study_id"],
            "seed": args.seed,
            "systems": {name: "/".join(target) for name, target in systems.items()},
            "scene_sources": {name: "/".join(target) for name, target in scene_sources.items()},
            "strata_scene_system": args.strata_scene_system if args.scene_per_image else None,
            "images": key_images,
        }
        return study, key

    output_dir = Path(args.output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    stale = sorted(p.name for p in (output_dir / "images").glob("*.jpg") if p.stem not in selected)
    if stale:
        raise SystemExit(f"STOP: {output_dir / 'images'} holds {len(stale)} images that are not in this selection "
                         f"(e.g. {stale[:3]}). They come from an earlier package; use a new --output-dir.")
    for image_id in sorted(selected):
        target = output_dir / "images" / f"{image_id}.jpg"
        if not target.exists():
            shutil.copyfile((image_dir / f"{image_id}.jpg").resolve(), target)

    # ---- sampling table -------------------------------------------------------
    population = Counter(stratum(i) for i in candidates)
    in_main, in_pilot = Counter(stratum(i) for i in main_ids), Counter(stratum(i) for i in pilot_ids)
    with open(output_dir / "sampling_strata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scene_group", "pseudo_reference", "evaluation_set", "study", "pilot"])
        for key_ in sorted(population):
            writer.writerow([key_[0], key_[1], population[key_], in_main[key_], in_pilot[key_]])

    system_names = (set(systems) | {"/".join(t) for t in systems.values()} | {t[1] for t in systems.values()}
                    | set(scene_sources) | {"/".join(t) for t in scene_sources.values()})
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
        n_scenes = [len(image.get("scenes", [])) for image in study["images"]]
        n_empty = sum(1 for i in ids if not reference[i])
        summary.append(
            f"{suffix.upper()}: {len(ids)} images ({n_empty} with an empty pseudo-reference)\n"
            f"   sound judgements per annotator: {sum(n_sounds)} (mean {sum(n_sounds) / len(ids):.1f} per image, max {max(n_sounds)})\n"
            f"   object checks per annotator:    {sum(n_objects)} (mean {sum(n_objects) / len(ids):.1f} per image)\n"
            + (f"   scene labels per annotator:     {sum(n_scenes)} (mean {sum(n_scenes) / len(ids):.1f} per image)\n" if scene_sources else "")
            + f"   page: {output_dir / f'annotate_{suffix}.html'}   private key: {output_dir / f'key_{suffix}.json'}"
        )

    print(f"Evaluation set: {len(candidates)} candidate images. Strata: "
          + (f"scene group of {args.strata_scene_system} x empty / non-empty pseudo-reference" if args.scene_per_image
             else "empty / non-empty pseudo-reference") + f" ({len(population)} non-empty strata). Seed {args.seed}.")
    groups = sorted({key_[0] for key_ in population})
    print(f"{'scene group':<26}{'evaluation set':>15}{'share':>8}{'study':>7}{'pilot':>7}")
    for group in groups:
        n_pop = sum(v for k, v in population.items() if k[0] == group)
        n_main = sum(v for k, v in in_main.items() if k[0] == group)
        n_pil = sum(v for k, v in in_pilot.items() if k[0] == group)
        print(f"{group:<26}{n_pop:>15}{n_pop / len(candidates):>8.1%}{n_main:>7}{n_pil:>7}")
    if scene_sources:
        print("Scene sources: " + "; ".join(f"{name} = {'/'.join(target)}" for name, target in scene_sources.items()))
    for note in notes:
        print(note)
    print("\n".join(summary))
    most_common = sorted(Counter(s for image_id in main_ids for s in pools[image_id][0]).items(),
                         key=lambda item: (-item[1], ontology[item[0]]["name"]))[:8]
    print(f"Distinct sounds in the study vocabulary: {len(sound_vocab)}; distinct visual terms: {len(object_vocab)}")
    print("Most frequent sounds in the main set: " + "; ".join(f"{ontology[s]['name']} ({c})" for s, c in most_common))
    print(f"Images copied to {output_dir / 'images'}: {len(selected)}. Strata table: {output_dir / 'sampling_strata.csv'}")
    print("Give annotators the annotate_*.html page and the images/ folder, kept side by side. Never the key_*.json files.")


if __name__ == "__main__":
    main()
