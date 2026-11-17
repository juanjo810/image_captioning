"""Sensitivity of the scene-consistency metrics to the hand-set family weights
``FAMILY_WEIGHTS`` in ``evaluate_scene_awareness.py``.

Two metrics share those weights (``--target``):

- ``gt`` (default): ``gt_entity_scene_consistency``. Weighted share of the
  Visual Genome *reference* objects whose family is expected in the scene
  group the system predicted. It depends only on the predicted scene label,
  so it scores the scene step (Places365 in Workflow B, the VLM scene calls
  in Workflow A), not the detected entities or the AudioSet nodes.
- ``pred``: ``pred_entity_scene_consistency``. Same score over the system's
  own evidence terms; an image with no predicted node scores 0.

The weights are author-chosen values in [0.35, 1.00]. This script asks whether
the *ranking of the compared systems* depends on them. Systems are restricted
to the images they all have, so means are comparable (``--systems`` selects
which systems enter; under ``gt``, systems that share a scene prediction tie
exactly, so pass one per distinct scene source). It recomputes the metric under:

- ``default``: the weights as published (sanity check: must reproduce
  ``family_evidence_score`` exactly);
- ``uniform``: every family weighted 1.0 (plain count-based consistency);
- ``shuffle_<k>``: the default weight values permuted among families
  (does the specific assignment matter, or just the spread?);
- ``drop_<family>``: leave-one-family-out (that family's evidence ignored);
- ``random_<k>``: ``--n-draws`` weight vectors drawn uniformly in
  [``--weight-min``, ``--weight-max``] per family.

Nothing is re-run: the prediction JSONs listed in the manifest(s) are read
once, family counts are cached, and only the weighted score is recomputed.

Outputs under ``--output-dir``: ``conditions.csv`` (condition × system mean
score and rank), ``draws.csv`` (one row per random draw × system) and
``summary.md`` (default ranking, fraction of draws that preserve it, pairwise
order stability, Kendall tau-a distribution, leave-one-out table).

Example::

    python -m scripts.evaluation.scene_weight_sensitivity \
      --manifest outputs/manifests/predictions_all.csv \
      --vg-object-refs data/vg_refs/vg_object_references.csv \
      --object-alias data/aliases/object_alias.csv \
      --n-draws 1000 --seed 0 \
      --output-dir outputs/metrics/stats/scene_sensitivity

Systems are labelled from the manifest's ``detector``/``scene_model``/
``captioner`` columns, or with ``--manifest PATH:LABEL``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.evaluation import evaluate_scene_awareness as esa
from scripts.evaluation.vg_utils import load_alias_map, load_json


SYSTEM_COLUMNS = ("detector", "scene_model", "captioner")
UNLISTED_FAMILY_WEIGHT = 0.20  # the fallback evaluate_scene_awareness uses


# ---------------------------------------------------------------------------
# Scoring (mirrors evaluate_scene_awareness.family_evidence_score, but with
# the weight table passed in instead of read from the module global)
# ---------------------------------------------------------------------------


def weighted_score(
    observed_counts: dict[str, int],
    expected_families: set[str],
    weights: dict[str, float],
) -> float:
    if not observed_counts:
        return 0.0
    total = 0.0
    supported = 0.0
    for family, count in observed_counts.items():
        weight = weights.get(family, UNLISTED_FAMILY_WEIGHT) * max(count, 1)
        total += weight
        if family in expected_families:
            supported += weight
    return supported / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _parse_manifest_arg(value: str) -> tuple[Path, str | None]:
    if ":" in value:
        path_part, _, label = value.rpartition(":")
        if label and "/" not in label and "\\" not in label and path_part:
            return Path(path_part), label
    return Path(value), None


def load_items(
    manifest_args: list[str],
    object_refs: dict[str, set[str]],
    object_alias: dict[str, str],
    only: set[str] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    """Return ({system_label: [ {image_id, pred_counts, gt_counts, expected} ]},
    every label seen in the manifests). With ``only``, prediction JSONs of
    other systems are not even opened."""
    items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_labels: set[str] = set()

    for arg in manifest_args:
        path, forced_label = _parse_manifest_arg(arg)
        for row in esa.load_predictions(path):
            if forced_label is not None:
                label = forced_label
            else:
                parts = [row.get(col, "") for col in SYSTEM_COLUMNS]
                label = "/".join(p for p in parts if p) or path.stem

            seen_labels.add(label)
            if only is not None and label not in only:
                continue

            pred_json = load_json(row["json_path"])
            scene_label = pred_json.get("core", {}).get("scene", {}).get("label", "")
            group = esa.scene_group(scene_label)
            expected = esa.SCENE_EXPECTED_FAMILIES.get(group or "", set())
            image_id = row["image_id"]

            items[label].append(
                {
                    "image_id": image_id,
                    "pred_counts": esa.prediction_family_counts(pred_json, object_alias),
                    "gt_counts": esa.labels_to_family_counts(
                        object_refs.get(image_id, set()), object_alias
                    ),
                    "expected": expected,
                }
            )

    if not seen_labels:
        raise ValueError("No manifest rows loaded.")
    return dict(items), seen_labels


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def mean_scores(
    items: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    counts_key: str = "pred_counts",
) -> dict[str, float]:
    return {
        label: statistics.fmean(
            weighted_score(item[counts_key], item["expected"], weights) for item in rows
        )
        for label, rows in items.items()
    }


def ranking(scores: dict[str, float]) -> list[str]:
    """Systems from best to worst (ties broken by label for determinism)."""
    return [label for label, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def kendall_tau_a(order_a: list[str], order_b: list[str]) -> float:
    """Kendall's tau-a between two rankings of the same labels."""
    pos_a = {label: i for i, label in enumerate(order_a)}
    pos_b = {label: i for i, label in enumerate(order_b)}
    labels = list(order_a)
    concordant = discordant = 0
    for x, y in itertools.combinations(labels, 2):
        sa = pos_a[x] - pos_a[y]
        sb = pos_b[x] - pos_b[y]
        if sa * sb > 0:
            concordant += 1
        elif sa * sb < 0:
            discordant += 1
    n_pairs = len(labels) * (len(labels) - 1) / 2
    return (concordant - discordant) / n_pairs if n_pairs else 1.0


def build_conditions(
    families: list[str],
    default_weights: dict[str, float],
    n_shuffles: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, float]]]:
    conditions: list[tuple[str, dict[str, float]]] = [
        ("default", dict(default_weights)),
        ("uniform", {f: 1.0 for f in families}),
    ]
    values = [default_weights[f] for f in families]
    for k in range(n_shuffles):
        shuffled = list(values)
        rng.shuffle(shuffled)
        conditions.append((f"shuffle_{k}", dict(zip(families, shuffled))))
    for family in families:
        weights = dict(default_weights)
        weights[family] = 0.0
        conditions.append((f"drop_{family}", weights))
    return conditions


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
        writer.writerows(rows)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", action="append", required=True,
                        help="Prediction manifest CSV, optionally PATH:LABEL. Repeatable.")
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--n-shuffles", type=int, default=20)
    parser.add_argument("--weight-min", type=float, default=0.35)
    parser.add_argument("--weight-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--target", choices=["gt", "pred"], default="gt",
        help="Which metric drives rankings: 'gt' = gt_entity_scene_consistency "
             "(reference objects vs predicted scene; default), 'pred' = "
             "pred_entity_scene_consistency (the system's own evidence terms).",
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Labels of the systems to include (default: all in the manifest).",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    object_refs = esa.load_object_refs(Path(args.vg_object_refs))
    object_alias = load_alias_map(args.object_alias)
    items, seen_labels = load_items(
        args.manifest, object_refs, object_alias,
        only=set(args.systems) if args.systems else None,
    )

    if args.systems:
        unknown = [label for label in args.systems if label not in seen_labels]
        if unknown:
            raise ValueError(
                f"--systems refers to unknown systems {unknown}. Known: {sorted(seen_labels)}"
            )
        items = {label: items[label] for label in args.systems}

    # Same images for every system, so the means are comparable.
    common_ids = set.intersection(*({item["image_id"] for item in rows} for rows in items.values()))
    dropped = {label: len(rows) - len(common_ids) for label, rows in items.items()}
    items = {
        label: [item for item in rows if item["image_id"] in common_ids]
        for label, rows in items.items()
    }
    if not common_ids:
        raise ValueError("The selected systems have no image in common.")

    counts_key = f"{args.target}_counts"
    metric_name = f"{args.target}_entity_scene_consistency"

    families = sorted(esa.ACOUSTIC_ENTITY_FAMILIES)
    default_weights = {f: esa.FAMILY_WEIGHTS.get(f, UNLISTED_FAMILY_WEIGHT) for f in families}

    # Sanity check: our scorer must reproduce the published one exactly.
    for rows in items.values():
        for item in rows:
            for key in ("pred_counts", "gt_counts"):
                ours = weighted_score(item[key], item["expected"], default_weights)
                theirs = esa.family_evidence_score(item[key], item["expected"])
                if abs(ours - theirs) > 1e-12:
                    raise AssertionError(
                        f"weighted_score diverges from family_evidence_score on "
                        f"{item['image_id']} ({key}): {ours} vs {theirs}"
                    )

    # Fixed conditions -------------------------------------------------------
    condition_rows: list[dict[str, Any]] = []
    default_scores = mean_scores(items, default_weights, counts_key=counts_key)
    default_order = ranking(default_scores)

    for name, weights in build_conditions(families, default_weights, args.n_shuffles, rng):
        pred_scores = mean_scores(items, weights, counts_key="pred_counts")
        gt_scores = mean_scores(items, weights, counts_key="gt_counts")
        scores = gt_scores if args.target == "gt" else pred_scores
        order = ranking(scores)
        for label in items:
            condition_rows.append(
                {
                    "condition": name,
                    "system": label,
                    "target": args.target,
                    "pred_entity_scene_consistency": pred_scores[label],
                    "gt_entity_scene_consistency": gt_scores[label],
                    "rank": order.index(label) + 1,
                    "kendall_tau_vs_default": kendall_tau_a(default_order, order),
                    **{f"w_{f}": weights[f] for f in families},
                }
            )

    # Random draws -----------------------------------------------------------
    draw_rows: list[dict[str, Any]] = []
    taus: list[float] = []
    same_ranking = 0
    pair_agreement: dict[tuple[str, str], int] = defaultdict(int)
    per_system_draws: dict[str, list[float]] = defaultdict(list)

    for k in range(args.n_draws):
        weights = {f: rng.uniform(args.weight_min, args.weight_max) for f in families}
        scores = mean_scores(items, weights, counts_key=counts_key)
        order = ranking(scores)
        tau = kendall_tau_a(default_order, order)
        taus.append(tau)
        same_ranking += int(order == default_order)
        for i in range(len(default_order) - 1):
            a, b = default_order[i], default_order[i + 1]
            pair_agreement[(a, b)] += int(scores[a] >= scores[b])
        for label in items:
            per_system_draws[label].append(scores[label])
            draw_rows.append(
                {
                    "draw": k,
                    "system": label,
                    metric_name: scores[label],
                    "rank": order.index(label) + 1,
                    **{f"w_{f}": weights[f] for f in families},
                }
            )

    # Write --------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(condition_rows, output_dir / "conditions.csv")
    if draw_rows:
        write_csv(draw_rows, output_dir / "draws.csv")

    lines = [f"# Scene-consistency weight sensitivity: `{metric_name}`\n"]
    lines.append(f"Systems: {len(items)}, all restricted to the {len(common_ids)} images they have in common"
                 + (" (dropped per system: " + ", ".join(f"{k}={v}" for k, v in dropped.items() if v) + ")"
                    if any(dropped.values()) else "") + ".\n")
    lines.append("## Default weights\n")
    lines.append(f"| rank | system | {metric_name} |")
    lines.append("|---|---|---|")
    for i, label in enumerate(default_order, 1):
        lines.append(f"| {i} | {label} | {default_scores[label]:.4f} |")

    if args.n_draws:
        lines.append(f"\n## Random weights, {args.n_draws} draws in "
                     f"[{args.weight_min}, {args.weight_max}] per family\n")
        lines.append(f"- Full default ranking preserved in {same_ranking}/{args.n_draws} draws "
                     f"({same_ranking / args.n_draws:.1%}).")
        lines.append(f"- Kendall tau-a vs default: mean {statistics.fmean(taus):.3f}, "
                     f"min {min(taus):.3f}, 5th pct {_quantile(taus, 0.05):.3f}.")
        lines.append("- Adjacent-pair order preserved (A >= B):")
        for (a, b), count in pair_agreement.items():
            lines.append(f"    - {a} >= {b}: {count}/{args.n_draws} ({count / args.n_draws:.1%})")
        lines.append("\n| system | mean over draws | 2.5th pct | 97.5th pct | min | max |")
        lines.append("|---|---|---|---|---|---|")
        for label in default_order:
            vals = per_system_draws[label]
            lines.append(
                f"| {label} | {statistics.fmean(vals):.4f} | {_quantile(vals, 0.025):.4f} | "
                f"{_quantile(vals, 0.975):.4f} | {min(vals):.4f} | {max(vals):.4f} |"
            )

    lines.append("\n## Fixed conditions (uniform, shuffled, leave-one-family-out)\n")
    header = ["condition"] + default_order + ["tau vs default"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in condition_rows:
        by_condition[row["condition"]][row["system"]] = row
    for name, systems in by_condition.items():
        cells = [name] + [f"{systems[l][metric_name]:.4f} (#{systems[l]['rank']})"
                          for l in default_order]
        cells.append(f"{next(iter(systems.values()))['kendall_tau_vs_default']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")

    summary = "\n".join(lines) + "\n"
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[OK] written to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
