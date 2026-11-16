"""Bootstrap confidence intervals and paired significance tests over per-image
metric CSVs.

Works on the per-image CSVs written by ``evaluate_audioset_semantics.py``
(``--per-image-output``) and ``evaluate_scene_awareness.py`` (same flag), or on
any CSV with an ``image_id`` column plus numeric metric columns. No model is
re-run: everything is computed from the per-image scores that already exist.

What it produces (all under ``--output-dir``):

``ci_summary.csv``
    One row per (system, metric): n, mean, percentile-bootstrap CI. For the
    AudioSet per-image CSVs it also reports how the empty-reference images
    behave, because ``prf`` scores an image 1.0 when both the prediction and the
    pseudo-reference are empty (abstention is rewarded):

    - ``n_empty_ref`` / ``n_empty_pred`` / ``n_empty_both``
    - ``mean_nonempty_ref``: mean over images whose pseudo-reference is non-empty
    - ``mean_excl_empty_both``: mean excluding the images where both sets are empty

``paired_comparisons.csv``
    One row per (metric, system_a, system_b): paired difference of means with a
    paired-bootstrap CI, a sign-flip permutation p-value, a Wilcoxon
    signed-rank p-value (when scipy is importable), Holm-corrected p-values,
    win/tie rates and the matched-pairs rank-biserial correlation as effect
    size.

``summary.md``
    The same numbers as Markdown tables, ready to paste into a draft.

Systems are identified by a label. A label can be given explicitly with
``--per-image PATH:LABEL``; without it, the label is built from the
``detector``/``scene_model``/``captioner`` columns of each row, so a single
CSV holding several configurations works too.

Example::

    python -m scripts.evaluation.bootstrap_stats \
      --per-image outputs/metrics/audioset_per_image_gemma_five.csv:A-gemma-five \
      --per-image outputs/metrics/audioset_per_image_gemma_det.csv:A-gemma-det \
      --per-image outputs/metrics/audioset_per_image_owlv2.csv:B-owlv2 \
      --per-image outputs/metrics/audioset_per_image_gdino.csv:B-gdino \
      --compare A-gemma-det:B-owlv2 --compare A-gemma-five:A-gemma-det \
      --compare B-owlv2:B-gdino \
      --n-boot 10000 --seed 0 \
      --output-dir outputs/metrics/stats/audioset

Reproducibility: the random generator is seeded (``--seed``); the same inputs
and seed give the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # scipy is optional: only the Wilcoxon test needs it.
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except Exception:  # pragma: no cover - depends on the environment
    _scipy_wilcoxon = None


AUDIOSET_DEFAULT_METRICS = [
    "audioset_exact_node_f1",
    "audioset_parent_f1",
    "audioset_top_level_f1",
    "audioset_lca_similarity",
    "audioset_tree_distance_similarity",
]

SCENE_DEFAULT_METRICS = [
    "scene_group_known",
    "pred_entity_scene_consistency",
]

SYSTEM_COLUMNS = ("detector", "scene_model", "captioner")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _parse_per_image_arg(value: str) -> tuple[Path, str | None]:
    """``PATH`` or ``PATH:LABEL``. A Windows drive letter (``C:\\...``) is not a
    label separator: only the last ``:`` is considered, and only when what
    follows it contains no path separator."""
    if ":" in value:
        path_part, _, label = value.rpartition(":")
        if label and "/" not in label and "\\" not in label and path_part:
            return Path(path_part), label
    return Path(value), None


def _to_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_systems(
    per_image_args: list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    """Return {label: {image_id: row}}."""
    systems: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

    for arg in per_image_args:
        path, forced_label = _parse_per_image_arg(arg)
        if not path.exists():
            raise FileNotFoundError(f"Per-image CSV not found: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "image_id" not in reader.fieldnames:
                raise ValueError(f"{path}: expected an 'image_id' column")

            for row in reader:
                if forced_label is not None:
                    label = forced_label
                else:
                    parts = [row.get(col, "") for col in SYSTEM_COLUMNS]
                    label = "/".join(p for p in parts if p) or path.stem

                image_id = row["image_id"]
                if image_id in systems[label]:
                    raise ValueError(
                        f"Duplicate image_id {image_id!r} for system {label!r} "
                        f"(file {path}). Give each file its own label with PATH:LABEL."
                    )
                systems[label][image_id] = row

    if not systems:
        raise ValueError("No rows loaded.")
    return dict(systems)


def load_strata(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    strata: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError(f"{path}: expected columns image_id,stratum")
        stratum_col = "stratum" if "stratum" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            strata[row["image_id"]] = row[stratum_col]
    return strata


def detect_default_metrics(systems: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    any_row = next(iter(next(iter(systems.values())).values()))
    columns = set(any_row.keys())
    if any(m in columns for m in AUDIOSET_DEFAULT_METRICS):
        return [m for m in AUDIOSET_DEFAULT_METRICS if m in columns]
    if any(m in columns for m in SCENE_DEFAULT_METRICS):
        return [m for m in SCENE_DEFAULT_METRICS if m in columns]
    raise ValueError(
        "Could not infer the metric columns; pass them explicitly with --metrics."
    )


# ---------------------------------------------------------------------------
# Bootstrap machinery
# ---------------------------------------------------------------------------


_CHUNK = 1000  # resamples per block, keeps memory at ~n * _CHUNK * 8 bytes


def bootstrap_indices(
    n: int,
    n_boot: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """(n_boot, n) array of resampled row indices. With ``strata`` (an int
    array of length n), resampling is done within each stratum, keeping the
    per-stratum counts fixed."""
    if strata is None:
        return rng.integers(0, n, size=(n_boot, n))

    out = np.empty((n_boot, n), dtype=np.int64)
    col = 0
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        k = len(members)
        out[:, col:col + k] = members[rng.integers(0, k, size=(n_boot, k))]
        col += k
    return out


def bootstrap_means(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Means of ``n_boot`` bootstrap resamples of ``values``, computed in blocks."""
    means = np.empty(n_boot, dtype=float)
    done = 0
    while done < n_boot:
        block = min(_CHUNK, n_boot - done)
        idx = bootstrap_indices(len(values), block, rng, strata)
        means[done:done + block] = values[idx].mean(axis=1)
        done += block
    return means


def percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def sign_flip_pvalue(diff: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    """Paired permutation test: randomly flip the sign of each paired
    difference and compare |mean| with the observed |mean|. Two-sided; the
    smallest attainable p is 1 / (n_perm + 1)."""
    observed = abs(diff.mean())
    count = 0
    done = 0
    while done < n_perm:
        block = min(_CHUNK, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(block, len(diff)))
        perm_means = np.abs((signs * diff).mean(axis=1))
        count += int(np.sum(perm_means >= observed - 1e-15))
        done += block
    return float((count + 1) / (n_perm + 1))


def rank_biserial(diff: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby, 2014): ranks of the
    non-zero |differences|, r = (sum of ranks with d>0 - sum with d<0) / total."""
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = _average_ranks(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = ranks.sum()
    return float((positive - negative) / total) if total > 0 else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # 1-based average rank
        i = j + 1
    return ranks


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm–Bonferroni step-down adjustment. ``None`` entries are left as is
    and do not count towards the family size."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(pvalues)
    if m == 0:
        return adjusted
    indexed.sort()
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def _metric_vector(
    rows: dict[str, dict[str, str]], image_ids: list[str], metric: str
) -> tuple[np.ndarray, list[str]]:
    values, kept = [], []
    for image_id in image_ids:
        v = _to_float(rows[image_id].get(metric))
        if v is None:
            continue
        values.append(v)
        kept.append(image_id)
    return np.asarray(values, dtype=float), kept


def _strata_array(image_ids: list[str], strata: dict[str, str] | None) -> np.ndarray | None:
    if strata is None:
        return None
    labels = [strata.get(i, "__missing__") for i in image_ids]
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes


def ci_summary(
    systems: dict[str, dict[str, dict[str, str]]],
    image_ids_by_system: dict[str, list[str]],
    metrics: list[str],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
    strata: dict[str, str] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for label, rows in systems.items():
        image_ids = image_ids_by_system[label]
        has_counts = all(
            col in next(iter(rows.values()))
            for col in ("n_gt_audioset_tags", "n_pred_audioset_tags")
        )

        for metric in metrics:
            values, kept = _metric_vector(rows, image_ids, metric)
            n = len(values)
            if n == 0:
                out.append({"system": label, "metric": metric, "n": 0})
                continue

            boot_means = bootstrap_means(values, n_boot, rng, _strata_array(kept, strata))
            low, high = percentile_ci(boot_means, ci)

            row: dict[str, Any] = {
                "system": label,
                "metric": metric,
                "n": n,
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
                "std": float(values.std(ddof=1)) if n > 1 else 0.0,
            }

            if has_counts:
                n_ref = np.asarray(
                    [_to_float(rows[i]["n_gt_audioset_tags"]) or 0.0 for i in kept]
                )
                n_pred = np.asarray(
                    [_to_float(rows[i]["n_pred_audioset_tags"]) or 0.0 for i in kept]
                )
                empty_ref = n_ref == 0
                empty_pred = n_pred == 0
                empty_both = empty_ref & empty_pred
                row.update(
                    {
                        "n_empty_ref": int(empty_ref.sum()),
                        "n_empty_pred": int(empty_pred.sum()),
                        "n_empty_both": int(empty_both.sum()),
                        "mean_nonempty_ref": (
                            float(values[~empty_ref].mean()) if (~empty_ref).any() else ""
                        ),
                        "mean_excl_empty_both": (
                            float(values[~empty_both].mean()) if (~empty_both).any() else ""
                        ),
                    }
                )
            out.append(row)
    return out


def paired_comparisons(
    systems: dict[str, dict[str, dict[str, str]]],
    pairs: list[tuple[str, str]],
    metrics: list[str],
    n_boot: int,
    ci: float,
    rng: np.random.Generator,
    strata: dict[str, str] | None,
    holm_family: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for metric in metrics:
        for a, b in pairs:
            common = sorted(set(systems[a]) & set(systems[b]))
            va, kept_a = _metric_vector(systems[a], common, metric)
            vb, kept_b = _metric_vector(systems[b], common, metric)
            kept = sorted(set(kept_a) & set(kept_b))
            if not kept:
                out.append({"metric": metric, "system_a": a, "system_b": b, "n": 0})
                continue
            va, _ = _metric_vector(systems[a], kept, metric)
            vb, _ = _metric_vector(systems[b], kept, metric)
            diff = va - vb
            n = len(diff)

            boot_diff = bootstrap_means(diff, n_boot, rng, _strata_array(kept, strata))
            low, high = percentile_ci(boot_diff, ci)

            p_perm = sign_flip_pvalue(diff, n_boot, rng)
            p_wilcoxon: float | None = None
            if _scipy_wilcoxon is not None and np.any(diff != 0):
                try:
                    p_wilcoxon = float(_scipy_wilcoxon(va, vb, zero_method="wilcox").pvalue)
                except Exception:  # pragma: no cover - scipy edge cases
                    p_wilcoxon = None

            out.append(
                {
                    "metric": metric,
                    "system_a": a,
                    "system_b": b,
                    "n": n,
                    "mean_a": float(va.mean()),
                    "mean_b": float(vb.mean()),
                    "diff_a_minus_b": float(diff.mean()),
                    "diff_ci_low": low,
                    "diff_ci_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "p_permutation": p_perm,
                    "p_wilcoxon": p_wilcoxon,
                    "win_rate_a": float((diff > 0).mean()),
                    "win_rate_b": float((diff < 0).mean()),
                    "tie_rate": float((diff == 0).mean()),
                    "rank_biserial_r": rank_biserial(diff),
                }
            )

    # Holm correction. family = "all": every (metric, pair) test is one
    # family; "metric": one family per metric.
    def _apply(key: str) -> None:
        if holm_family == "all":
            groups: dict[Any, list[int]] = {"all": [i for i, r in enumerate(out) if r.get("n")]}
        else:
            groups = defaultdict(list)
            for i, r in enumerate(out):
                if r.get("n"):
                    groups[r["metric"]].append(i)
        for indices in groups.values():
            adjusted = holm([out[i].get(key) for i in indices])
            for i, adj in zip(indices, adjusted):
                out[i][f"{key}_holm"] = adj

    _apply("p_permutation")
    _apply("p_wilcoxon")
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if abs(value) < 1e-3 and value != 0:
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def markdown_summary(
    ci_rows: list[dict[str, Any]],
    cmp_rows: list[dict[str, Any]],
    ci: float,
    n_boot: int,
    holm_family: str,
    n_common: int | None,
) -> str:
    lines: list[str] = []
    pct = int(round(ci * 100))
    lines.append(f"# Bootstrap summary ({pct}% percentile CI, B = {n_boot})\n")
    if n_common is not None:
        lines.append(f"All systems restricted to the {n_common} images they have in common.\n")

    lines.append("## Per-system means with CI\n")
    has_empty = any("n_empty_ref" in r for r in ci_rows)
    header = ["system", "metric", "n", "mean", f"{pct}% CI"]
    if has_empty:
        header += ["n empty ref", "n empty both", "mean (non-empty ref)", "mean (excl. empty-both)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in ci_rows:
        if not r.get("n"):
            continue
        cells = [
            r["system"], r["metric"], str(r["n"]), _fmt(r["mean"]),
            f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]",
        ]
        if has_empty:
            cells += [
                _fmt(r.get("n_empty_ref")), _fmt(r.get("n_empty_both")),
                _fmt(r.get("mean_nonempty_ref")), _fmt(r.get("mean_excl_empty_both")),
            ]
        lines.append("| " + " | ".join(cells) + " |")

    if cmp_rows:
        lines.append(f"\n## Paired comparisons (Holm family: {holm_family})\n")
        header = [
            "metric", "A", "B", "n", "mean A", "mean B", "A−B", f"{pct}% CI of A−B",
            "p perm (Holm)", "p Wilcoxon (Holm)", "win A / win B / tie", "rank-biserial r",
        ]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for r in cmp_rows:
            if not r.get("n"):
                continue
            lines.append(
                "| " + " | ".join([
                    r["metric"], r["system_a"], r["system_b"], str(r["n"]),
                    _fmt(r["mean_a"]), _fmt(r["mean_b"]), _fmt(r["diff_a_minus_b"]),
                    f"[{_fmt(r['diff_ci_low'])}, {_fmt(r['diff_ci_high'])}]",
                    f"{_fmt(r['p_permutation'], 4)} ({_fmt(r.get('p_permutation_holm'), 4)})",
                    f"{_fmt(r['p_wilcoxon'], 4)} ({_fmt(r.get('p_wilcoxon_holm'), 4)})",
                    f"{_fmt(r['win_rate_a'], 2)} / {_fmt(r['win_rate_b'], 2)} / {_fmt(r['tie_rate'], 2)}",
                    _fmt(r["rank_biserial_r"]),
                ]) + " |"
            )
        if _scipy_wilcoxon is None:
            lines.append(
                "\n_scipy was not importable, so the Wilcoxon column is empty; the "
                "sign-flip permutation test is reported instead._"
            )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_pairs(values: Iterable[str], labels: set[str]) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"--compare expects LABEL_A:LABEL_B, got {value!r}")
        a, b = value.split(":", 1)
        for label in (a, b):
            if label not in labels:
                raise ValueError(
                    f"--compare refers to unknown system {label!r}. Known: {sorted(labels)}"
                )
        pairs.append((a, b))
    return pairs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--per-image", action="append", required=True,
        help="Per-image CSV, optionally as PATH:LABEL. Repeatable.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        help="Metric columns to analyse. Default: the 5 official AudioSet metrics "
             "or the scene metrics, whichever the CSV contains.",
    )
    parser.add_argument(
        "--compare", action="append", default=[],
        help="Paired comparison LABEL_A:LABEL_B. Repeatable. Keep this list short "
             "and fixed in advance; every pair enters the Holm correction.",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strata-csv", default=None,
        help="Optional CSV image_id,stratum (e.g. the scene group used to build "
             "the evaluation subset). Bootstrap resamples within strata.",
    )
    parser.add_argument(
        "--align", choices=["common", "own"], default="common",
        help="'common' (default): every system is evaluated on the images all "
             "systems share, so means are comparable. 'own': each system uses all "
             "its images (comparisons still use the pairwise intersection).",
    )
    parser.add_argument(
        "--holm-family", choices=["all", "metric"], default="all",
        help="Family for the Holm correction: all (metric, pair) tests together "
             "(default, conservative) or one family per metric.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    systems = load_systems(args.per_image)
    metrics = args.metrics or detect_default_metrics(systems)
    strata = load_strata(Path(args.strata_csv)) if args.strata_csv else None
    rng = np.random.default_rng(args.seed)

    n_common: int | None = None
    if args.align == "common":
        common = set.intersection(*(set(rows) for rows in systems.values()))
        n_common = len(common)
        dropped = {label: len(rows) - n_common for label, rows in systems.items()}
        if any(dropped.values()):
            print(
                "[warn] restricting to the common image set; dropped per system: "
                + ", ".join(f"{k}={v}" for k, v in dropped.items()),
                file=sys.stderr,
            )
        image_ids_by_system = {label: sorted(common) for label in systems}
    else:
        image_ids_by_system = {label: sorted(rows) for label, rows in systems.items()}

    pairs = _parse_pairs(args.compare, set(systems))

    ci_rows = ci_summary(systems, image_ids_by_system, metrics, args.n_boot, args.ci, rng, strata)
    cmp_rows = paired_comparisons(
        systems, pairs, metrics, args.n_boot, args.ci, rng, strata, args.holm_family
    ) if pairs else []

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(ci_rows, output_dir / "ci_summary.csv")
    if cmp_rows:
        write_csv(cmp_rows, output_dir / "paired_comparisons.csv")
    summary = markdown_summary(ci_rows, cmp_rows, args.ci, args.n_boot, args.holm_family, n_common)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[OK] written to {output_dir}")


if __name__ == "__main__":
    main()
