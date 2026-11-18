"""Check that a sample of images is typical of the set it was drawn from.

The human plausibility study annotates a few hundred images out of the
evaluation set. This script compares the sample with that set on what the
evaluation already knows about every image, and says whether each difference
is larger than drawing the same number of images at random would give:

- the pseudo-reference: share of empty ones, nodes per image, share of images
  whose reference touches each AudioSet top-level category;
- each system: automatic exact-node F1, predicted nodes per image, share of
  images with a prediction;
- optionally the scene group predicted for the image.

For every characteristic it reports the mean over the population, the mean over
the sample and a Monte Carlo p-value: the share of random subsets of the same
size (drawn without replacement from the population) whose mean is at least as
far from the population mean as the sample's. With many characteristics a few
p-values under 0.05 are expected by chance; the summary counts them.

It reads the per-image CSV of the AudioSet evaluation and runs nothing else.

Example::

    python -m scripts.evaluation.sample_representativeness \\
      --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \\
      --scene-per-image outputs/final_metrics/stats_3000/scene_per_image.csv \\
      --sample subset_500=outputs/seeds_500/image_ids.txt \\
      --sample study_300=outputs/plausibility_study/study_main.json \\
      --sample pilot_30=outputs/plausibility_study/study_pilot.json \\
      --output-dir outputs/plausibility_study/representativeness
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.bootstrap_stats import _fmt, write_csv
from scripts.evaluation.build_plausibility_pool import DEFAULT_SYSTEMS


def read_ids(path: Path) -> list[str]:
    """Image ids from a text file (one per line) or from a study_*.json /
    key_*.json written by build_plausibility_pool.py."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        images = json.loads(text)["images"]
        return [str(i) for i in images] if isinstance(images, dict) else [str(im["image_id"]) for im in images]
    return [line.strip() for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-image", required=True, help="AudioSet per-image CSV of the whole evaluation set.")
    parser.add_argument("--sample", action="append", required=True, metavar="NAME=PATH",
                        help="A sample to check: a text file of image ids or a study_*.json. Repeatable.")
    parser.add_argument("--population-ids", default=None,
                        help="Text file with the ids of the population. Default: every image of --reference-system.")
    parser.add_argument("--reference-system", default="owlv2/densenet161",
                        help="detector/scene_model whose rows give the pseudo-reference (it is the same for all).")
    parser.add_argument("--systems", nargs="*", default=None,
                        help="detector/scene_model labels to compare. Default: the systems of the plausibility study.")
    parser.add_argument("--metric", default="audioset_exact_node_f1")
    parser.add_argument("--scene-per-image", default=None, help="Optional scene per-image CSV (scene group of each image).")
    parser.add_argument("--scene-system", default="owlv2/densenet161",
                        help="detector/scene_model whose scene group is used with --scene-per-image.")
    parser.add_argument("--n-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)       # system -> image -> row
    with open(args.per_image, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows[f"{row['detector']}/{row['scene_model']}"][row["image_id"]] = row
    if args.reference_system not in rows:
        raise SystemExit(f"no rows for {args.reference_system} in {args.per_image}. Systems: {', '.join(sorted(rows))}")
    reference = rows[args.reference_system]

    population = read_ids(Path(args.population_ids)) if args.population_ids else sorted(reference)
    unknown = [i for i in population if i not in reference]
    if unknown:
        raise SystemExit(f"{len(unknown)} population ids have no row of {args.reference_system}, e.g. {unknown[:5]}")
    position = {image_id: k for k, image_id in enumerate(population)}
    n_pop = len(population)

    if args.systems:
        systems = list(args.systems)
        missing = [s for s in systems if s not in rows]
        if missing:
            raise SystemExit(f"systems not in {args.per_image}: {missing}. Available: {', '.join(sorted(rows))}")
    else:
        systems = [s for s in DEFAULT_SYSTEMS.values() if s in rows] or sorted(rows)

    # ---- one value per image and characteristic (NaN = not applicable) ------
    features: list[tuple[str, str, np.ndarray]] = []                      # (block, name, values)

    def add(block: str, name: str, values: list[float]) -> None:
        features.append((block, name, np.asarray(values, dtype=float)))

    def nodes(image_id: str) -> list[str]:
        return [t for t in (reference[image_id].get("gt_audioset_tags") or "").split("|") if t]

    add("pseudo-reference", "share of images with an empty reference", [float(not nodes(i)) for i in population])
    add("pseudo-reference", "reference nodes per image", [float(len(nodes(i))) for i in population])
    top_levels = sorted({t for i in population for t in (reference[i].get("gt_audioset_top_level_names") or "").split("|") if t})
    for top in top_levels:
        add("reference touches the category",
            top, [float(top in (reference[i].get("gt_audioset_top_level_names") or "").split("|")) for i in population])
    for s in systems:
        own = rows[s]
        add(f"automatic {args.metric}", s, [float(own[i][args.metric]) if i in own and own[i].get(args.metric) not in (None, "") else np.nan
                                           for i in population])
        add("predicted nodes per image", s, [float(own[i].get("n_pred_audioset_tags") or 0) if i in own else np.nan for i in population])
        add("share of images with a prediction", s, [float(i in own) for i in population])
    if args.scene_per_image:
        group: dict[str, str] = {}
        with open(args.scene_per_image, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if f"{row['detector']}/{row['scene_model']}" == args.scene_system:
                    group[row["image_id"]] = row.get("scene_group") or "unknown"
        if not group:
            raise SystemExit(f"no rows for {args.scene_system} in {args.scene_per_image}")
        for g in sorted(set(group.values())):
            add(f"scene group ({args.scene_system})", g, [float(group[i] == g) if i in group else np.nan for i in population])

    matrix = np.stack([values for _, _, values in features], axis=1)      # images x characteristics
    with np.errstate(invalid="ignore"):
        pop_mean = np.nanmean(matrix, axis=0)
    rng = np.random.default_rng(args.seed)

    out_rows: list[dict[str, Any]] = []
    lines = ["# Is the sample typical of the evaluation set?\n"]
    lines.append(f"Population: {n_pop} images ({args.per_image}). For each characteristic, *p* is the share of {args.n_draws} "
                 "random subsets of the same size whose mean is at least as far from the population mean as the "
                 "sample's. A small *p* means the sample is unusual on that characteristic. With many characteristics, "
                 "about 5% fall under 0.05 by chance.\n")
    for spec in args.sample:
        if "=" not in spec:
            raise SystemExit(f"--sample expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        ids = read_ids(Path(path))
        outside = [i for i in ids if i not in position]
        if outside:
            raise SystemExit(f"sample {name}: {len(outside)} ids are not in the population, e.g. {outside[:5]}")
        if len(set(ids)) != len(ids):
            raise SystemExit(f"sample {name}: repeated image ids")
        index = np.array([position[i] for i in ids])
        n = len(index)
        with np.errstate(invalid="ignore"):
            sample_mean = np.nanmean(matrix[index], axis=0)
            # random subsets without replacement, in blocks to bound memory
            extreme = np.zeros(matrix.shape[1])
            valid = np.zeros(matrix.shape[1])
            done = 0
            while done < args.n_draws:
                block = min(500, args.n_draws - done)
                draws = np.argsort(rng.random((block, n_pop)), axis=1)[:, :n]
                means = np.nanmean(matrix[draws], axis=1)                 # block x characteristics
                ok = ~np.isnan(means)
                extreme += (ok & (np.abs(means - pop_mean) >= np.abs(sample_mean - pop_mean) - 1e-12)).sum(axis=0)
                valid += ok.sum(axis=0)
                done += block
        p_values = np.where(valid > 0, (extreme + 1) / (valid + 1), np.nan)

        tested = [k for k in range(matrix.shape[1]) if not np.isnan(sample_mean[k]) and not np.isnan(p_values[k])]
        low = [k for k in tested if p_values[k] < 0.05]
        lines.append(f"## {name}: {n} images\n")
        lines.append(f"{len(low)} of {len(tested)} characteristics with p < 0.05 (expected by chance: about "
                     f"{0.05 * len(tested):.1f}).\n")
        lines.append("| characteristic | | population | sample | difference | p |")
        lines.append("|---|---|---|---|---|---|")
        for k, (block, label, _) in enumerate(features):
            row = {"sample": name, "n_images": n, "block": block, "characteristic": label,
                   "population_mean": float(pop_mean[k]), "sample_mean": float(sample_mean[k]),
                   "difference": float(sample_mean[k] - pop_mean[k]), "p_monte_carlo": float(p_values[k]),
                   "n_sample_with_value": int((~np.isnan(matrix[index, k])).sum())}
            out_rows.append(row)
            flag = " *" if k in low else ""
            lines.append(f"| {block} | {label} | {_fmt(row['population_mean'])} | {_fmt(row['sample_mean'])} | "
                         f"{row['difference']:+.3f} | {_fmt(row['p_monte_carlo'])}{flag} |")
        lines.append("")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out_rows, out / "representativeness.csv")
    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[OK] written to {out}")


if __name__ == "__main__":
    main()
