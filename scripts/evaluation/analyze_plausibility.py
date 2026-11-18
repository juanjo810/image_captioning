"""Analyse the human plausibility study.

Inputs are the CSV files exported by the annotation page (one per annotator)
and the private key written by ``build_plausibility_pool.py``. The script
reports:

- per annotator: images completed, time, distribution of answers;
- inter-annotator agreement: Krippendorff's alpha (ordinal on the four-level
  plausibility scale, nominal on its two binary readings and on object
  presence), with a bootstrap CI over images;
- a consensus label per item by strict majority of the annotators who rated it;
- per system, against that consensus: F1, human-validated precision, recall
  over the pool, evidence fidelity and the reasons given for implausible
  sounds, with bootstrap CIs over images. The pseudo-reference and the random
  sounds are scored the same way, as "pseudo_reference" and "random_baseline";
- paired differences between systems on the images both have a prediction for;
- for sounds with cited evidence, whether the evidence is in the image and
  whether the sound is plausible, which separates perception errors from rules
  that over-generate;
- per AudioSet node: how often it is judged plausible when it is in the pool;
- optionally, how the automatic score against pseudo-references relates to the
  score against human judgement, per image and across systems;
- every item the annotators disagree on, for the pilot review.

Two readings of "plausible" are always reported: *strict* (answer "Sí") and
*lenient* ("Sí" or "Posible").

Recall is computed over the pool: every sound some system or the
pseudo-reference proposed for the image, plus the random ones. Plausible sounds
nobody proposed are not in the pool, so this recall is an upper bound of the
true one.

Example::

    python -m scripts.evaluation.analyze_plausibility \\
      --annotations outputs/plausibility_study/results/plausibility_v1_main_*.csv \\
      --key outputs/plausibility_study/key_main.json \\
      --per-image outputs/final_metrics/stats_3000/audioset_per_image.csv \\
      --output-dir outputs/plausibility_study/analysis_main
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluation.bootstrap_stats import _fmt, holm, percentile_ci, sign_flip_pvalue, write_csv


PLAUS_LEVELS = [1, 2, 3, 4]
SEEN_LEVELS = ["yes", "no", "unsure"]
REASON_LABELS = {"absent": "no source", "inactive": "source inactive", "place": "does not fit the place", "other": "other"}
EXTRA_SYSTEMS = ("pseudo_reference", "random_baseline")

# Every metric is a ratio of two per-image sums, so one bootstrap over images
# serves them all. The two F1 are means over images (denominator: images the
# system has a prediction for), with the convention of the automatic metric.
METRICS = {
    "f1_human_lenient": ("f1_len", "has"),
    "f1_human_strict": ("f1_str", "has"),
    "precision_lenient": ("tp_len", "n_pred"),
    "precision_strict": ("tp_str", "n_pred"),
    "recall_pool_lenient": ("tp_len", "pool_len"),
    "recall_pool_strict": ("tp_str", "pool_str"),
    "evidence_fidelity": ("faithful", "with_evidence"),
    "cited_terms_present": ("terms_present", "terms"),
}
COMPARE_METRICS = ("f1_human_lenient", "precision_lenient", "precision_strict", "recall_pool_lenient", "evidence_fidelity")
# The comparisons of the automatic evaluation; those naming a system that is
# not in the key are skipped.
DEFAULT_COMPARISONS = [
    ("qwen_deterministic", "owlv2"), ("gemma_deterministic", "owlv2"),
    ("qwen_deterministic", "gemma_deterministic"), ("owlv2", "grounding_dino"),
    ("qwen_vlm_rules", "qwen_deterministic"), ("gemma_vlm_rules", "gemma_deterministic"),
    ("qwen_vlm_free", "qwen_vlm_rules"), ("gemma_vlm_free", "gemma_vlm_rules"),
]


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------


def coincidence_matrix(units: list[list[Any]], categories: list[Any]) -> np.ndarray:
    """Coincidence matrix of Krippendorff's alpha. Units with fewer than two
    ratings contribute nothing."""
    index = {c: i for i, c in enumerate(categories)}
    o = np.zeros((len(categories), len(categories)))
    for ratings in units:
        m = len(ratings)
        if m < 2:
            continue
        counts = Counter(ratings)
        for c, nc in counts.items():
            for k, nk in counts.items():
                pairs = nc * (nc - 1) if c == k else nc * nk
                o[index[c], index[k]] += pairs / (m - 1)
    return o


def alpha_from_coincidences(o: np.ndarray, level: str) -> float | None:
    """alpha = 1 - D_o / D_e, with the nominal or ordinal difference function
    (Krippendorff, 2011, "Computing Krippendorff's Alpha-Reliability")."""
    n_c = o.sum(axis=1)
    n = n_c.sum()
    k = len(n_c)
    if n <= 1:
        return None
    delta = np.zeros((k, k))
    for c in range(k):
        for g in range(k):
            if c == g:
                continue
            if level == "nominal":
                delta[c, g] = 1.0
            else:  # ordinal
                lo, hi = min(c, g), max(c, g)
                delta[c, g] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0) ** 2
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))
    if d_e == 0:
        return None      # every rating in the same category: alpha is undefined
    return float(1.0 - d_o / d_e)


def alpha_with_ci(
    units_by_image: dict[str, list[list[Any]]], categories: list[Any], level: str,
    n_boot: int, ci: float, rng: np.random.Generator,
) -> dict[str, Any]:
    images = sorted(units_by_image)
    per_image = np.stack([coincidence_matrix(units_by_image[i], categories) for i in images]) if images else np.zeros((0, 1, 1))
    n_units = sum(1 for i in images for u in units_by_image[i] if len(u) >= 2)
    full = sum(1 for i in images for u in units_by_image[i] if len(u) >= 2 and len(set(u)) == 1)
    out: dict[str, Any] = {"n_units": n_units, "share_unanimous": full / n_units if n_units else None, "alpha": None}
    if not n_units:
        return out
    out["alpha"] = alpha_from_coincidences(per_image.sum(axis=0), level)
    boots = []
    for _ in range(n_boot):
        value = alpha_from_coincidences(per_image[rng.integers(0, len(images), len(images))].sum(axis=0), level)
        if value is not None:
            boots.append(value)
    if boots:
        out["ci_low"], out["ci_high"] = percentile_ci(np.asarray(boots), ci)
    return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def majority(votes: list[bool]) -> tuple[bool, bool]:
    """(strict-majority outcome, is_tie). A tie counts as False."""
    yes = sum(votes)
    return yes * 2 > len(votes), yes * 2 == len(votes)


def set_f1(pred: set[str], ref: set[str]) -> float:
    """Same convention as the automatic metric: both empty -> 1, one empty -> 0."""
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    tp = len(pred & ref)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(ref)
    return 2 * precision * recall / (precision + recall)


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    def ranks(values: list[float]) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        r = np.empty(len(values))
        sorted_values = np.asarray(values)[order]
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", nargs="+", required=True, help="CSV files exported by the annotation page.")
    parser.add_argument("--key", required=True, help="key_main.json or key_pilot.json from build_plausibility_pool.py.")
    parser.add_argument("--per-image", default=None,
                        help="Optional AudioSet per-image CSV, to relate the automatic score to the human one.")
    parser.add_argument("--min-annotators", type=int, default=2,
                        help="An image enters the consensus when at least this many annotators completed it.")
    parser.add_argument("--compare", action="append", default=[], metavar="A:B",
                        help="Paired comparison A - B between two systems of the key (or pseudo_reference, "
                             "random_baseline). Repeatable. Default: the comparisons of the automatic evaluation.")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    csv.field_size_limit(sys.maxsize)
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key_images: dict[str, Any] = key["images"]
    systems = list(key["systems"]) + list(EXTRA_SYSTEMS)
    rng = np.random.default_rng(args.seed)

    # ---- load annotations ---------------------------------------------------
    # ratings[kind][image][item][annotator] = answer ; reasons[image][item][annotator] = reason
    ratings: dict[str, dict[str, dict[str, dict[str, Any]]]] = {"sound": defaultdict(lambda: defaultdict(dict)),
                                                                   "object": defaultdict(lambda: defaultdict(dict))}
    reasons: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    names: dict[str, str] = {}
    done: dict[str, set[str]] = defaultdict(set)            # annotator -> images completed
    seconds: dict[str, dict[str, float]] = defaultdict(dict)
    skipped_other_study = 0
    for path in args.annotations:
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["study_id"] != key["study_id"]:
                    skipped_other_study += 1
                    continue
                annotator, image_id = row["annotator"], row["image_id"]
                if image_id not in key_images or row.get("image_done") != "1":
                    continue                                  # only images the annotator finished
                done[annotator].add(image_id)
                seconds[annotator][image_id] = float(row.get("image_seconds") or 0)
                names[row["item_id"]] = row["item_name"]
                if row["kind"] == "sound":
                    answer = int(row["answer"])
                    ratings["sound"][image_id][row["item_id"]][annotator] = answer
                    # A reason only counts with "Improbable"/"No": the page keeps the old
                    # one if the annotator later changes the answer to "Sí"/"Posible".
                    if row.get("reason") and answer >= 3:
                        reasons[image_id][row["item_id"]][annotator] = row["reason"]
                else:
                    ratings["object"][image_id][row["item_id"]][annotator] = row["answer"]
    if not done:
        raise SystemExit(f"No annotation row matches study {key['study_id']!r} "
                         f"({skipped_other_study} rows belong to another study).")
    annotators = sorted(done)

    # ---- per annotator ------------------------------------------------------
    annotator_rows = []
    for a in annotators:
        sound_answers = [r[a] for img in ratings["sound"].values() for r in img.values() if a in r]
        object_answers = [r[a] for img in ratings["object"].values() for r in img.values() if a in r]
        secs = list(seconds[a].values())
        annotator_rows.append({
            "annotator": a, "images_done": len(done[a]), "sound_judgements": len(sound_answers),
            "object_checks": len(object_answers), "hours": sum(secs) / 3600,
            "median_seconds_per_image": statistics.median(secs) if secs else None,
            **{f"share_answer_{lvl}": sound_answers.count(lvl) / len(sound_answers) if sound_answers else None for lvl in PLAUS_LEVELS},
            "share_object_yes": object_answers.count("yes") / len(object_answers) if object_answers else None,
        })

    # ---- agreement ----------------------------------------------------------
    def units(kind: str, transform) -> dict[str, list[list[Any]]]:
        return {img: [[transform(v) for v in item.values()] for item in items.values()]
                for img, items in ratings[kind].items()}

    agreement_rows = []
    for label, kind, transform, cats, level in (
        ("sounds: 4-level scale (ordinal)", "sound", lambda v: v, PLAUS_LEVELS, "ordinal"),
        ("sounds: lenient (Sí/Posible vs rest)", "sound", lambda v: v <= 2, [True, False], "nominal"),
        ("sounds: strict (Sí vs rest)", "sound", lambda v: v == 1, [True, False], "nominal"),
        ("objects: yes / no / unsure", "object", lambda v: v, SEEN_LEVELS, "nominal"),
        ("objects: yes vs rest", "object", lambda v: v == "yes", [True, False], "nominal"),
    ):
        result = alpha_with_ci(units(kind, transform), cats, level, min(args.n_boot, 2000), args.ci, rng)
        agreement_rows.append({"measure": label, "level": level, **result})
    if len(annotators) > 2:                                   # one row per pair, to spot an annotator who reads the scale differently
        for i, first in enumerate(annotators):
            for second in annotators[i + 1:]:
                pair_units = {img: [[item[x] for x in (first, second) if x in item] for item in items.values()]
                              for img, items in ratings["sound"].items()}
                result = alpha_with_ci(pair_units, PLAUS_LEVELS, "ordinal", min(args.n_boot, 1000), args.ci, rng)
                agreement_rows.append({"measure": f"sounds: 4-level scale, {first} vs {second}", "level": "ordinal", **result})

    # ---- consensus ----------------------------------------------------------
    images = sorted(i for i in key_images if sum(1 for a in annotators if i in done[a]) >= args.min_annotators)
    if not images:
        raise SystemExit(f"No image was completed by at least {args.min_annotators} annotators yet. "
                         "For a single annotator use --min-annotators 1.")
    lenient: dict[str, dict[str, bool]] = {}
    strict: dict[str, dict[str, bool]] = {}
    present: dict[str, dict[str, bool]] = {}
    consensus_rows, disagreement_rows = [], []
    ties = Counter()
    for image_id in images:
        lenient[image_id], strict[image_id], present[image_id] = {}, {}, {}
        for item_id, by_annotator in ratings["sound"][image_id].items():
            values = list(by_annotator.values())
            lenient[image_id][item_id], tie_l = majority([v <= 2 for v in values])
            strict[image_id][item_id], tie_s = majority([v == 1 for v in values])
            ties["sound_lenient"] += tie_l
            ties["sound_strict"] += tie_s
            given = [reasons[image_id][item_id][a] for a in by_annotator if a in reasons[image_id][item_id]]
            top = Counter(given).most_common(2)
            reason = "" if not top else ("mixed" if len(top) > 1 and top[0][1] == top[1][1] else top[0][0])
            consensus_rows.append({
                "image_id": image_id, "kind": "sound", "item_id": item_id, "item_name": names.get(item_id, item_id),
                "n_ratings": len(values), "mean_rating": float(np.mean(values)),
                "plausible_lenient": int(lenient[image_id][item_id]), "plausible_strict": int(strict[image_id][item_id]),
                "tie_lenient": int(tie_l), "main_reason": reason,
                **{f"rating_{a}": by_annotator.get(a, "") for a in annotators},
            })
            if len({v <= 2 for v in values}) > 1:
                disagreement_rows.append({"image_id": image_id, "kind": "sound", "item": names.get(item_id, item_id),
                                          **{a: by_annotator.get(a, "") for a in annotators}})
        for term, by_annotator in ratings["object"][image_id].items():
            values = list(by_annotator.values())
            present[image_id][term], tie_o = majority([v == "yes" for v in values])
            ties["object"] += tie_o
            consensus_rows.append({
                "image_id": image_id, "kind": "object", "item_id": term, "item_name": term, "n_ratings": len(values),
                "present": int(present[image_id][term]), "tie_present": int(tie_o),
                **{f"rating_{a}": by_annotator.get(a, "") for a in annotators},
            })
            if len({v == "yes" for v in values}) > 1:
                disagreement_rows.append({"image_id": image_id, "kind": "object", "item": term,
                                          **{a: by_annotator.get(a, "") for a in annotators}})

    # ---- per system ---------------------------------------------------------
    def predicted(image_id: str, system: str) -> dict[str, list[str]]:
        info = key_images[image_id]
        if system == "pseudo_reference":
            return {n: [] for n in info["reference_nodes"]}
        if system == "random_baseline":
            return {n: [] for n in info["random_sounds"]}
        return info["systems"][system]["nodes"]

    n_img = len(images)
    cols = ("has", "n_pred", "tp_len", "tp_str", "pool_len", "pool_str", "f1_len", "f1_str",
            "with_evidence", "faithful", "terms", "terms_present", "unjudged",
            "pl_seen", "pl_unseen", "impl_seen", "impl_unseen")
    arrays = {s: {c: np.zeros(n_img) for c in cols} for s in systems}
    reason_counts: dict[str, Counter] = {s: Counter() for s in systems}
    pool_size = np.zeros(n_img)                               # plausible sounds in the pool (lenient)
    for j, image_id in enumerate(images):
        pool_len = {n for n, ok in lenient[image_id].items() if ok}
        pool_str = {n for n, ok in strict[image_id].items() if ok}
        pool_size[j] = len(pool_len)
        for s in systems:
            if s in key["systems"] and not key_images[image_id]["systems"][s]["has_prediction"]:
                continue                                      # failed run: the image does not count for this system
            nodes = predicted(image_id, s)
            judged = {n: ev for n, ev in nodes.items() if n in lenient[image_id]}
            a = arrays[s]
            a["has"][j] = 1
            a["unjudged"][j] = len(nodes) - len(judged)
            a["n_pred"][j] = len(judged)
            a["tp_len"][j] = sum(1 for n in judged if lenient[image_id][n])
            a["tp_str"][j] = sum(1 for n in judged if strict[image_id][n])
            a["pool_len"][j], a["pool_str"][j] = len(pool_len), len(pool_str)
            a["f1_len"][j] = set_f1(set(judged), pool_len)
            a["f1_str"][j] = set_f1(set(judged), pool_str)
            cited: set[str] = set()
            for n, evidence in judged.items():
                known = [t for t in evidence if t in present[image_id]]
                if known:
                    seen = all(present[image_id][t] for t in known)
                    a["with_evidence"][j] += 1
                    a["faithful"][j] += seen
                    a[("pl_" if lenient[image_id][n] else "impl_") + ("seen" if seen else "unseen")][j] += 1
                    cited.update(known)
                if not lenient[image_id][n]:
                    given = [reasons[image_id][n][x] for x in reasons[image_id][n]]
                    top = Counter(given).most_common(2)
                    reason_counts[s]["mixed" if len(top) > 1 and top[0][1] == top[1][1] else (top[0][0] if top else "none")] += 1
            a["terms"][j] = len(cited)
            a["terms_present"][j] = sum(1 for t in cited if present[image_id][t])

    idx = rng.integers(0, n_img, size=(args.n_boot, n_img))

    def value(a: dict[str, np.ndarray], metric: str) -> float | None:
        num, den = METRICS[metric]
        return ratio(float(a[num].sum()), float(a[den].sum()))

    def samples(a: dict[str, np.ndarray], metric: str) -> np.ndarray:
        """Bootstrap replicates of a ratio of sums over images (NaN when the
        resample leaves the denominator empty)."""
        num, den = METRICS[metric]
        d = a[den][idx].sum(axis=1)
        return np.where(d > 0, a[num][idx].sum(axis=1) / np.where(d > 0, d, 1.0), np.nan)

    def ci_of(values: np.ndarray) -> tuple[float | None, float | None]:
        values = values[~np.isnan(values)]
        return percentile_ci(values, args.ci) if len(values) else (None, None)

    system_rows = []
    for s in systems:
        a = arrays[s]
        row: dict[str, Any] = {"system": s, "n_images": int(a["has"].sum()),
                               "images_without_prediction_file": int(n_img - a["has"].sum()),
                               "n_predicted_sounds": int(a["n_pred"].sum())}
        for name in METRICS:
            row[name] = value(a, name)
            row[name + "_ci_low"], row[name + "_ci_high"] = ci_of(samples(a, name))
        nothing = (pool_size == 0) & (a["has"] == 1)
        row["images_nothing_plausible"] = int(nothing.sum())
        row["abstention_when_nothing_plausible"] = (None if s == "random_baseline" else
                                                    ratio(float((a["n_pred"][nothing] == 0).sum()), float(nothing.sum())))
        with_evidence = float(a["with_evidence"].sum())
        row["n_sounds_with_judged_evidence"] = int(with_evidence)
        for cell in ("pl_seen", "impl_seen", "impl_unseen", "pl_unseen"):
            row["share_" + cell] = ratio(float(a[cell].sum()), with_evidence)
        row["unjudged_predicted_sounds"] = int(a["unjudged"].sum())
        system_rows.append(row)

    # ---- paired comparisons -------------------------------------------------
    requested = [tuple(c.split(":", 1)) for c in args.compare] if args.compare else DEFAULT_COMPARISONS
    comparison_rows = []
    for pair in requested:
        if len(pair) != 2 or pair[0] not in arrays or pair[1] not in arrays:
            if args.compare:
                raise SystemExit(f"--compare {':'.join(pair)}: unknown system. Available: {', '.join(systems)}")
            continue
        sa, sb = pair
        both = (arrays[sa]["has"] == 1) & (arrays[sb]["has"] == 1)
        a = {c: v * both for c, v in arrays[sa].items()}
        b = {c: v * both for c, v in arrays[sb].items()}
        for metric in COMPARE_METRICS:
            va, vb = value(a, metric), value(b, metric)
            if va is None or vb is None:
                continue
            low, high = ci_of(samples(a, metric) - samples(b, metric))
            row = {"system_a": sa, "system_b": sb, "metric": metric, "n_images": int(both.sum()),
                   "value_a": va, "value_b": vb, "diff": va - vb, "ci_low": low, "ci_high": high,
                   "ci_excludes_zero": None if low is None else bool(low > 0 or high < 0), "p_sign_flip": None}
            if metric == "f1_human_lenient":                  # a mean over images: paired permutation test
                row["p_sign_flip"] = sign_flip_pvalue((a["f1_len"] - b["f1_len"])[both], 10000, rng)
            comparison_rows.append(row)
    for row, adjusted in zip(comparison_rows, holm([r["p_sign_flip"] for r in comparison_rows])):
        row["p_holm"] = adjusted

    reason_rows = []
    for s in systems:
        total = sum(reason_counts[s].values())
        reason_rows.append({"system": s, "n_implausible_predicted": total,
                            **{k: (reason_counts[s][k] / total if total else None) for k in ("absent", "inactive", "place", "other", "mixed", "none")}})

    # ---- per node -----------------------------------------------------------
    by_node: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "len": 0, "str": 0, "ratings": [], "reasons": Counter()})
    for image_id in images:
        proposed = set(key_images[image_id]["reference_nodes"])
        for s in key["systems"]:
            proposed |= set(key_images[image_id]["systems"][s]["nodes"])
        for item_id, by_annotator in ratings["sound"][image_id].items():
            if item_id not in proposed:
                continue                                      # random sounds are not rule outputs
            entry = by_node[item_id]
            entry["n"] += 1
            entry["len"] += lenient[image_id][item_id]
            entry["str"] += strict[image_id][item_id]
            entry["ratings"].extend(by_annotator.values())
            entry["reasons"].update(reasons[image_id][item_id].values())
    node_rows = sorted(({
        "node_id": node_id, "name": names.get(node_id, node_id), "n_images_in_pool": e["n"],
        "share_plausible_lenient": e["len"] / e["n"], "share_plausible_strict": e["str"] / e["n"],
        "mean_rating": float(np.mean(e["ratings"])),
        "most_common_reason": (e["reasons"].most_common(1)[0][0] if e["reasons"] else ""),
    } for node_id, e in by_node.items()), key=lambda r: (-r["n_images_in_pool"], r["name"]))

    # ---- automatic score vs human score (optional) --------------------------
    relation_rows = []
    if args.per_image:
        auto: dict[str, dict[str, float]] = defaultdict(dict)
        targets = {target: name for name, target in key["systems"].items()}
        wanted = set(images)
        with open(args.per_image, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = targets.get(f"{row['detector']}/{row['scene_model']}")
                if name and row["image_id"] in wanted:
                    auto[name][row["image_id"]] = float(row["audioset_exact_node_f1"])
        for s in key["systems"]:
            pairs = [(auto[s][i], float(arrays[s]["f1_len"][j])) for j, i in enumerate(images)
                     if i in auto[s] and arrays[s]["has"][j]]
            if not pairs:
                continue
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            relation_rows.append({"system": s, "n_images": len(pairs), "mean_f1_vs_pseudo_reference": float(np.mean(xs)),
                                  "mean_f1_vs_human_lenient": float(np.mean(ys)), "spearman_per_image": spearman(xs, ys)})

    # ---- write --------------------------------------------------------------
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(annotator_rows, out / "annotators.csv")
    write_csv(agreement_rows, out / "agreement.csv")
    write_csv(consensus_rows, out / "items_consensus.csv")
    write_csv(system_rows, out / "systems.csv")
    write_csv(reason_rows, out / "reasons.csv")
    write_csv(node_rows, out / "nodes.csv")
    if comparison_rows:
        write_csv(comparison_rows, out / "comparisons.csv")
    if disagreement_rows:
        write_csv(disagreement_rows, out / "disagreements.csv")
    if relation_rows:
        write_csv(relation_rows, out / "automatic_vs_human.csv")

    pct = int(round(args.ci * 100))
    def with_ci(row: dict[str, Any], name: str) -> str:
        if row.get(name) is None:
            return "—"
        if row.get(name + "_ci_low") is None:
            return _fmt(row[name])
        return f"{_fmt(row[name])} [{_fmt(row[name + '_ci_low'])}, {_fmt(row[name + '_ci_high'])}]"

    lines = [f"# Human plausibility study: {key['study_id']}\n"]
    lines.append(f"Annotators: {len(annotators)} ({', '.join(annotators)}). Images with at least {args.min_annotators} "
                 f"complete annotations: {n_img} of {len(key_images)}. Consensus by strict majority; ties count as "
                 f"not plausible / not present (ties: {ties['sound_lenient']} sounds lenient, {ties['sound_strict']} strict, "
                 f"{ties['object']} objects). CIs: {pct}% bootstrap over images, B = {args.n_boot}.\n")

    lines.append("## Annotators\n")
    lines.append("| annotator | images | sound judgements | object checks | hours | median s/image | Sí | Posible | Improbable | No | objects seen |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in annotator_rows:
        lines.append("| " + " | ".join([
            r["annotator"], str(r["images_done"]), str(r["sound_judgements"]), str(r["object_checks"]),
            _fmt(r["hours"], 1), _fmt(r["median_seconds_per_image"], 0),
            *[f"{r[f'share_answer_{lvl}']:.0%}" if r[f"share_answer_{lvl}"] is not None else "—" for lvl in PLAUS_LEVELS],
            f"{r['share_object_yes']:.0%}" if r["share_object_yes"] is not None else "—",
        ]) + " |")

    lines.append("\n## Agreement (Krippendorff's alpha)\n")
    lines.append(f"| measure | items rated by 2+ | unanimous | alpha | {pct}% CI |")
    lines.append("|---|---|---|---|---|")
    for r in agreement_rows:
        ci_text = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]" if r.get("ci_low") is not None else "—"
        lines.append("| " + " | ".join([
            r["measure"], str(r["n_units"]), f"{r['share_unanimous']:.0%}" if r["share_unanimous"] is not None else "—",
            _fmt(r["alpha"]), ci_text]) + " |")

    lines.append("\n## Systems against the human consensus\n")
    lines.append("F1 is the mean over images of the set F1 between the predicted sounds and the sounds of the pool judged "
                 "plausible (lenient reading), with the convention of the automatic metric: both empty counts 1. Precision, "
                 "recall and evidence fidelity pool all sounds of all images. Recall is over the pool, not over every "
                 "plausible sound. `pseudo_reference` scores the Visual Genome pseudo-reference as if it were a system; "
                 "`random_baseline` scores the two random sounds added to each image.\n")
    lines.append("| system | images | sounds predicted | F1 vs human | precision (lenient) | precision (strict) | "
                 "recall over pool (lenient) | evidence fidelity | abstains when nothing is plausible |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in system_rows:
        abst = (f"{r['abstention_when_nothing_plausible']:.0%} of {r['images_nothing_plausible']}"
                if r["abstention_when_nothing_plausible"] is not None else "—")
        lines.append("| " + " | ".join([
            r["system"], str(r["n_images"]), str(r["n_predicted_sounds"]), with_ci(r, "f1_human_lenient"),
            with_ci(r, "precision_lenient"), with_ci(r, "precision_strict"), with_ci(r, "recall_pool_lenient"),
            with_ci(r, "evidence_fidelity"), abst]) + " |")
    unjudged = sum(r["unjudged_predicted_sounds"] for r in system_rows)
    if unjudged:
        lines.append(f"\n_{unjudged} predicted sounds had no judgement and were left out._")

    if comparison_rows:
        lines.append("\n## Paired comparisons (A − B)\n")
        lines.append(f"Images where both systems have a prediction. Difference with its {pct}% CI; `*` marks a CI that "
                     "excludes zero. The p-value is a paired sign-flip test on F1, Holm-adjusted over the comparisons.\n")
        lines.append("| A − B | images | F1 vs human | p (Holm) | precision (lenient) | precision (strict) | "
                     "recall over pool | evidence fidelity |")
        lines.append("|---|---|---|---|---|---|---|---|")
        by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in comparison_rows:
            by_pair[(r["system_a"], r["system_b"])][r["metric"]] = r
        def diff_cell(r: dict[str, Any] | None) -> str:
            if r is None:
                return "—"
            if r["ci_low"] is None:
                return f"{r['diff']:+.3f}"
            return f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" + (" *" if r["ci_excludes_zero"] else "")
        for (sa, sb), cells in by_pair.items():
            f1 = cells.get("f1_human_lenient")
            lines.append("| " + " | ".join([
                f"{sa} − {sb}", str(next(iter(cells.values()))["n_images"]), diff_cell(f1),
                _fmt(f1["p_holm"]) if f1 else "—", diff_cell(cells.get("precision_lenient")),
                diff_cell(cells.get("precision_strict")), diff_cell(cells.get("recall_pool_lenient")),
                diff_cell(cells.get("evidence_fidelity"))]) + " |")

    lines.append("\n## Sounds with cited evidence: is the evidence in the image, is the sound plausible\n")
    lines.append("Only sounds whose cited visual terms were judged. *Implausible, evidence seen*: the object is there but "
                 "the sound is not plausible (the rule over-generates). *Implausible, evidence not seen*: the object is "
                 "not in the image (perception error). *Plausible, evidence not seen*: a plausible sound for a wrong "
                 "reason.\n")
    lines.append("| system | sounds with judged evidence | plausible, evidence seen | implausible, evidence seen | "
                 "implausible, evidence not seen | plausible, evidence not seen |")
    lines.append("|---|---|---|---|---|---|")
    for r in system_rows:
        if not r["n_sounds_with_judged_evidence"]:
            continue
        lines.append("| " + " | ".join([r["system"], str(r["n_sounds_with_judged_evidence"])] + [
            f"{r['share_' + cell]:.0%}" for cell in ("pl_seen", "impl_seen", "impl_unseen", "pl_unseen")]) + " |")

    lines.append("\n## Why predicted sounds were judged implausible (lenient reading)\n")
    lines.append("| system | implausible predicted sounds | no source | source inactive | does not fit the place | other | mixed |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in reason_rows:
        lines.append("| " + " | ".join([r["system"], str(r["n_implausible_predicted"])] + [
            f"{r[k]:.0%}" if r[k] is not None else "—" for k in ("absent", "inactive", "place", "other", "mixed")]) + " |")

    lines.append("\n## Most frequent sounds in the pool\n")
    lines.append("| sound | images | plausible (lenient) | plausible (strict) | mean rating (1 Sí – 4 No) | most common reason |")
    lines.append("|---|---|---|---|---|---|")
    for r in node_rows[:30]:
        lines.append("| " + " | ".join([
            r["name"], str(r["n_images_in_pool"]), f"{r['share_plausible_lenient']:.0%}", f"{r['share_plausible_strict']:.0%}",
            _fmt(r["mean_rating"], 2), REASON_LABELS.get(r["most_common_reason"], r["most_common_reason"] or "—")]) + " |")

    if relation_rows:
        lines.append("\n## Automatic score (pseudo-reference) vs human score, same images\n")
        lines.append("| system | images | mean F1 vs pseudo-reference | mean F1 vs human (lenient) | Spearman per image |")
        lines.append("|---|---|---|---|---|")
        for r in relation_rows:
            lines.append("| " + " | ".join([r["system"], str(r["n_images"]), _fmt(r["mean_f1_vs_pseudo_reference"]),
                                           _fmt(r["mean_f1_vs_human_lenient"]), _fmt(r["spearman_per_image"])]) + " |")
        if len(relation_rows) >= 3:
            rho = spearman([r["mean_f1_vs_pseudo_reference"] for r in relation_rows],
                           [r["mean_f1_vs_human_lenient"] for r in relation_rows])
            lines.append(f"\nSpearman between the two system rankings ({len(relation_rows)} systems): {_fmt(rho)}.")

    if disagreement_rows:
        lines.append(f"\n{len(disagreement_rows)} items where annotators disagree on the lenient reading or on object "
                     "presence are listed in `disagreements.csv`.")

    summary = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[OK] written to {out}")


if __name__ == "__main__":
    main()
