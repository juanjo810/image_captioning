"""Run-to-run variance of stochastic runs (seeds), and arm-vs-arm comparisons.

Workflow A samples from a VLM, so a single run is one draw. Given several runs
of the same configuration (an *arm*: same model and sampling settings, different
seeds) over the same images, this script reports, from the per-image AudioSet
CSV written by ``evaluate_audioset_semantics.py``:

1. per run: the metric over all images, over images whose pseudo-reference is
   non-empty (something to find) and over images whose pseudo-reference is
   empty (there every metric is 1.0 if the run predicted nothing, so the mean
   is the abstention rate);
2. per arm: mean, standard deviation, minimum and maximum of those run means
   across seeds, i.e. how much a reported number moves between runs;
3. per arm: how stable the *predicted node sets* are between seeds for the same
   image (mean pairwise Jaccard, share of images with identical sets), which
   doubles as an explanation-consistency check;
4. per pair of arms: the difference of the per-image seed-averaged metric, with
   a paired-bootstrap CI over images and a sign-flip permutation p-value, plus
   the range of that difference over every seed-vs-seed pairing, to show whether
   its sign depends on which seeds are compared.

All figures use the images that every run of every arm has (generation failures
drop out), so they are comparable; each run's own image count is reported too.

Arms are declared with ``--arm NAME=LABELS``, where a label is a system label of
the per-image CSV (``detector/scene_model/captioner``). ``LABELS`` is either a
comma-separated list or a single pattern containing ``{seed}``, expanded with
``--seeds``. An arm may have a single run (e.g. a deterministic detector): it
then has no across-seed spread but can still be compared.

With ``--manifest`` (the CSV from ``build_prediction_manifest.py``) the script
also reads, from each prediction's metadata, how many visual terms the model
named freely and how many survived the mapping to the closed vocabulary.

Example::

    python -m scripts.evaluation.seed_variance \\
      --per-image outputs/seeds_500/audioset_per_image.csv \\
      --manifest outputs/seeds_500/manifest.csv \\
      --seeds 1 2 3 \\
      --arm "gemma_paper=unknown/gemma4_paper_seed{seed}_deterministic/template" \\
      --arm "qwen_paper=unknown/qwen36_paper_seed{seed}_deterministic/template" \\
      --compare qwen_paper:gemma_paper \\
      --n-images 500 --output-dir outputs/seeds_500/stats_deterministic
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.bootstrap_stats import (
    _fmt,
    _to_float,
    bootstrap_means,
    load_systems,
    percentile_ci,
    sign_flip_pvalue,
    write_csv,
)


SUBSETS = ("all", "nonempty_ref", "empty_ref")
SUBSET_TITLES = {
    "all": "all images",
    "nonempty_ref": "non-empty reference",
    "empty_ref": "empty reference (abstention rate)",
}


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def parse_arms(arm_args: list[str], seeds: list[str]) -> dict[str, list[str]]:
    arms: dict[str, list[str]] = {}
    for value in arm_args:
        if "=" not in value:
            raise ValueError(f"--arm expects NAME=LABELS, got {value!r}")
        name, spec = value.split("=", 1)
        if "{seed}" in spec:
            if not seeds:
                raise ValueError(f"--arm {name} uses {{seed}} but --seeds was not given")
            labels = [spec.replace("{seed}", str(seed)) for seed in seeds]
        else:
            labels = [label for label in spec.split(",") if label]
        if not labels:
            raise ValueError(f"--arm {name} has no runs")
        if name in arms:
            raise ValueError(f"--arm {name} declared twice")
        arms[name] = labels
    return arms


def node_set(row: dict[str, str]) -> frozenset[str]:
    return frozenset(tag for tag in (row.get("pred_audioset_tags") or "").split("|") if tag)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def subset_ids(common: list[str], ref_empty: dict[str, bool], subset: str) -> list[str]:
    if subset == "all":
        return common
    want_empty = subset == "empty_ref"
    return [image_id for image_id in common if ref_empty[image_id] == want_empty]


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Term counts from prediction metadata (optional)
# ---------------------------------------------------------------------------


def term_counts(manifest_path: Path, labels: set[str], common: set[str]) -> dict[str, dict[str, float]]:
    """{label: {mean_free_terms, mean_mapped_terms, share_no_mapped_term}} read
    from each prediction JSON's metadata, over the common images."""
    free: dict[str, list[float]] = {label: [] for label in labels}
    mapped: dict[str, list[float]] = {label: [] for label in labels}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            label = "/".join(
                part for part in (row.get(col, "") for col in ("detector", "scene_model", "captioner")) if part
            )
            if label not in labels or row["image_id"] not in common:
                continue
            metadata = json.loads(Path(row["json_path"]).read_text(encoding="utf-8")).get("metadata", {})
            if "n_free_visual_terms" in metadata and "n_mapped_visual_terms" in metadata:
                free[label].append(float(metadata["n_free_visual_terms"]))
                mapped[label].append(float(metadata["n_mapped_visual_terms"]))
    out: dict[str, dict[str, float]] = {}
    for label in labels:
        if free[label]:
            out[label] = {
                "mean_free_terms": float(np.mean(free[label])),
                "mean_mapped_terms": float(np.mean(mapped[label])),
                "share_no_mapped_term": float(np.mean([m == 0 for m in mapped[label]])),
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-image", action="append", required=True,
                        help="AudioSet per-image CSV. Repeatable.")
    parser.add_argument("--arm", action="append", required=True,
                        help="NAME=LABEL1,LABEL2,... or NAME=PATTERN with {seed}. Repeatable.")
    parser.add_argument("--seeds", nargs="+", default=[],
                        help="Values substituted for {seed} in --arm patterns.")
    parser.add_argument("--compare", action="append", default=[],
                        help="ARM_A:ARM_B. Repeatable.")
    parser.add_argument("--metrics", nargs="+", default=["audioset_exact_node_f1"])
    parser.add_argument("--manifest", default=None,
                        help="Optional prediction manifest, to report named vs mapped visual terms.")
    parser.add_argument("--n-images", type=int, default=None,
                        help="Images each run was given; if set, the failure rate per run is reported.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    arms = parse_arms(args.arm, args.seeds)
    systems = load_systems(args.per_image)

    all_labels = [label for labels in arms.values() for label in labels]
    missing = [label for label in all_labels if label not in systems]
    if missing:
        raise ValueError(
            f"Runs not found in the per-image CSV: {missing}.\nKnown systems: {sorted(systems)}"
        )
    for needed in ("n_gt_audioset_tags", "pred_audioset_tags"):
        if needed not in next(iter(systems[all_labels[0]].values())):
            raise ValueError(f"The per-image CSV has no {needed!r} column (need the AudioSet per-image CSV).")

    pairs: list[tuple[str, str]] = []
    for value in args.compare:
        a, _, b = value.partition(":")
        if a not in arms or b not in arms:
            raise ValueError(f"--compare {value!r}: unknown arm. Known: {sorted(arms)}")
        pairs.append((a, b))

    common = sorted(set.intersection(*(set(systems[label]) for label in all_labels)))
    if not common:
        raise ValueError("The runs have no image in common.")
    first = systems[all_labels[0]]
    ref_empty = {image_id: (_to_float(first[image_id]["n_gt_audioset_tags"]) or 0.0) == 0 for image_id in common}
    rng = np.random.default_rng(args.seed)

    def values(label: str, metric: str, ids: list[str]) -> np.ndarray:
        return np.asarray([_to_float(systems[label][i].get(metric)) or 0.0 for i in ids], dtype=float)

    terms = term_counts(Path(args.manifest), set(all_labels), set(common)) if args.manifest else {}

    # ---- 1. per run -------------------------------------------------------
    run_rows: list[dict[str, Any]] = []
    run_means: dict[tuple[str, str, str], float | None] = {}   # (label, metric, subset)
    for arm, labels in arms.items():
        for label in labels:
            row: dict[str, Any] = {"arm": arm, "run": label, "n_own_images": len(systems[label])}
            if args.n_images:
                # Only meaningful for runs that were given exactly --n-images images;
                # a run brought in from a larger evaluation (more images) has none.
                n_own = len(systems[label])
                row["failure_rate"] = 1.0 - n_own / args.n_images if n_own <= args.n_images else None
            row["n_common_images"] = len(common)
            row["n_empty_pred"] = sum(1 for i in common if not node_set(systems[label][i]))
            for metric in args.metrics:
                for subset in SUBSETS:
                    ids = subset_ids(common, ref_empty, subset)
                    mean = mean_or_none(list(values(label, metric, ids))) if ids else None
                    run_means[(label, metric, subset)] = mean
                    row[f"{metric}__{subset}"] = mean
            row.update(terms.get(label, {}))
            run_rows.append(row)

    # ---- 2. per arm -------------------------------------------------------
    arm_rows: list[dict[str, Any]] = []
    for arm, labels in arms.items():
        for metric in args.metrics:
            for subset in SUBSETS:
                means = [run_means[(label, metric, subset)] for label in labels]
                means = [m for m in means if m is not None]
                if not means:
                    continue
                arm_rows.append({
                    "arm": arm, "metric": metric, "subset": subset, "n_runs": len(means),
                    "n_images": len(subset_ids(common, ref_empty, subset)),
                    "mean": statistics.fmean(means),
                    "sd_across_runs": statistics.stdev(means) if len(means) > 1 else None,
                    "min": min(means), "max": max(means),
                })

    # ---- 3. stability of node sets between seeds --------------------------
    stability_rows: list[dict[str, Any]] = []
    for arm, labels in arms.items():
        if len(labels) < 2:
            continue
        sets = {label: {i: node_set(systems[label][i]) for i in common} for label in labels}
        pair_jaccard, pair_jaccard_nontrivial = [], []
        identical, identical_nontrivial, nontrivial = 0, 0, 0
        for image_id in common:
            image_sets = [sets[label][image_id] for label in labels]
            js = [jaccard(a, b) for a, b in itertools.combinations(image_sets, 2)]
            pair_jaccard.append(float(np.mean(js)))
            same = all(s == image_sets[0] for s in image_sets)
            identical += same
            if any(image_sets):
                nontrivial += 1
                pair_jaccard_nontrivial.append(float(np.mean(js)))
                identical_nontrivial += same
        metric = args.metrics[0]
        per_image = np.stack([values(label, metric, common) for label in labels])
        stability_rows.append({
            "arm": arm, "n_runs": len(labels), "n_images": len(common),
            "mean_pairwise_jaccard": float(np.mean(pair_jaccard)),
            "share_identical_node_sets": identical / len(common),
            "n_images_some_prediction": nontrivial,
            "mean_pairwise_jaccard_some_prediction": mean_or_none(pair_jaccard_nontrivial),
            "share_identical_some_prediction": identical_nontrivial / nontrivial if nontrivial else None,
            f"mean_within_image_sd__{metric}": float(np.mean(per_image.std(axis=0, ddof=1))),
        })

    # ---- 4. arm vs arm ----------------------------------------------------
    cmp_rows: list[dict[str, Any]] = []
    for metric in args.metrics:
        for a, b in pairs:
            for subset in SUBSETS:
                ids = subset_ids(common, ref_empty, subset)
                if not ids:
                    continue
                mat_a = np.stack([values(label, metric, ids) for label in arms[a]])
                mat_b = np.stack([values(label, metric, ids) for label in arms[b]])
                diff = mat_a.mean(axis=0) - mat_b.mean(axis=0)
                boot = bootstrap_means(diff, args.n_boot, rng)
                low, high = percentile_ci(boot, args.ci)
                seed_pair_diffs = [float(ra.mean() - rb.mean()) for ra in mat_a for rb in mat_b]
                cmp_rows.append({
                    "metric": metric, "subset": subset, "arm_a": a, "arm_b": b, "n_images": len(ids),
                    "mean_a": float(mat_a.mean()), "mean_b": float(mat_b.mean()),
                    "diff_a_minus_b": float(diff.mean()), "diff_ci_low": low, "diff_ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "p_permutation": sign_flip_pvalue(diff, args.n_boot, rng),
                    "seed_pair_diff_min": min(seed_pair_diffs),
                    "seed_pair_diff_max": max(seed_pair_diffs),
                    "n_seed_pairs": len(seed_pair_diffs),
                    "sign_consistent_across_seed_pairs": bool(
                        all(d > 0 for d in seed_pair_diffs) or all(d < 0 for d in seed_pair_diffs)
                    ),
                })

    # ---- write ------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_rows, output_dir / "runs.csv")
    write_csv(arm_rows, output_dir / "arms.csv")
    if stability_rows:
        write_csv(stability_rows, output_dir / "stability.csv")
    if cmp_rows:
        write_csv(cmp_rows, output_dir / "comparisons.csv")

    pct = int(round(args.ci * 100))
    n_empty = sum(ref_empty.values())
    lines = ["# Seed variance\n"]
    lines.append(
        f"Images common to every run: {len(common)} ({len(common) - n_empty} with a non-empty "
        f"pseudo-reference, {n_empty} with an empty one). Bootstrap B = {args.n_boot}.\n"
    )

    lines.append("## Per run\n")
    header = ["arm", "run", "own images"] + (["failure rate"] if args.n_images else []) + ["no prediction"]
    header += [f"{m.replace('audioset_', '')}: {SUBSET_TITLES[s]}" for m in args.metrics for s in SUBSETS]
    if terms:
        header += ["terms named", "terms mapped", "no mapped term"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in run_rows:
        cells = [r["arm"], r["run"], str(r["n_own_images"])]
        if args.n_images:
            cells.append(f"{r['failure_rate']:.1%}" if r["failure_rate"] is not None else "—")
        cells.append(f"{r['n_empty_pred']} ({r['n_empty_pred'] / len(common):.0%})")
        cells += [_fmt(r[f"{m}__{s}"]) for m in args.metrics for s in SUBSETS]
        if terms:
            cells += [_fmt(r.get("mean_free_terms"), 2), _fmt(r.get("mean_mapped_terms"), 2),
                      f"{r['share_no_mapped_term']:.1%}" if "share_no_mapped_term" in r else "—"]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Per arm: spread across runs\n")
    lines.append("| arm | metric | subset | runs | images | mean | SD across runs | min | max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in arm_rows:
        lines.append("| " + " | ".join([
            r["arm"], r["metric"], SUBSET_TITLES[r["subset"]], str(r["n_runs"]), str(r["n_images"]),
            _fmt(r["mean"]), _fmt(r["sd_across_runs"], 4), _fmt(r["min"]), _fmt(r["max"]),
        ]) + " |")

    if stability_rows:
        metric = args.metrics[0]
        lines.append("\n## Stability of the predicted node sets between runs of the same arm\n")
        lines.append("| arm | runs | mean pairwise Jaccard | identical in every run | "
                     "images with some prediction | Jaccard (some prediction) | identical (some prediction) | "
                     f"within-image SD of {metric.replace('audioset_', '')} |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in stability_rows:
            lines.append("| " + " | ".join([
                r["arm"], str(r["n_runs"]), _fmt(r["mean_pairwise_jaccard"]),
                f"{r['share_identical_node_sets']:.1%}", str(r["n_images_some_prediction"]),
                _fmt(r["mean_pairwise_jaccard_some_prediction"]),
                f"{r['share_identical_some_prediction']:.1%}" if r["share_identical_some_prediction"] is not None else "—",
                _fmt(r[f"mean_within_image_sd__{metric}"]),
            ]) + " |")

    if cmp_rows:
        lines.append("\n## Arm vs arm (per-image scores averaged over the arm's runs)\n")
        lines.append(f"| metric | subset | A | B | images | mean A | mean B | A−B | {pct}% CI | p perm | "
                     "range over seed pairings | same sign in every pairing |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in cmp_rows:
            lines.append("| " + " | ".join([
                r["metric"], SUBSET_TITLES[r["subset"]], r["arm_a"], r["arm_b"], str(r["n_images"]),
                _fmt(r["mean_a"]), _fmt(r["mean_b"]), _fmt(r["diff_a_minus_b"]),
                f"[{_fmt(r['diff_ci_low'])}, {_fmt(r['diff_ci_high'])}]", _fmt(r["p_permutation"], 4),
                f"[{_fmt(r['seed_pair_diff_min'])}, {_fmt(r['seed_pair_diff_max'])}] ({r['n_seed_pairs']})",
                "yes" if r["sign_consistent_across_seed_pairs"] else "no",
            ]) + " |")

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[OK] written to {output_dir}")


if __name__ == "__main__":
    main()
