from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from scripts.evaluation.vg_utils import load_json


def detection_counts(pred_json: dict[str, Any]) -> tuple[int, int]:
    """Return (n_raw_detections, n_filtered_detections) for one prediction JSON.

    Prefers the counts logged in metadata (paso 7 instrumentation). Falls back
    to len(extended.grounding) for older JSONs that predate that metadata, in
    which case n_raw_detections is unavailable and reported as -1.
    """

    metadata = pred_json.get("metadata", {})

    if "n_filtered_detections" in metadata:
        return (
            metadata.get("n_raw_detections", -1),
            metadata["n_filtered_detections"],
        )

    grounding = pred_json.get("extended", {}).get("grounding", [])
    return -1, len(grounding)


def compare_directories(before_dir: Path, after_dir: Path) -> list[dict[str, Any]]:
    before_files = {p.stem: p for p in before_dir.rglob("*.json")}
    after_files = {p.stem: p for p in after_dir.rglob("*.json")}

    common_ids = sorted(set(before_files) & set(after_files))
    missing_before = sorted(set(after_files) - set(before_files))
    missing_after = sorted(set(before_files) - set(after_files))

    if missing_before:
        print(f"[WARN] {len(missing_before)} image(s) only in --after-dir, skipped: {missing_before[:5]}...")
    if missing_after:
        print(f"[WARN] {len(missing_after)} image(s) only in --before-dir, skipped: {missing_after[:5]}...")

    rows = []
    for image_id in common_ids:
        raw_before, filtered_before = detection_counts(load_json(before_files[image_id]))
        raw_after, filtered_after = detection_counts(load_json(after_files[image_id]))

        rows.append({
            "image_id": image_id,
            "n_raw_before": raw_before,
            "n_raw_after": raw_after,
            "n_filtered_before": filtered_before,
            "n_filtered_after": filtered_after,
            "filtered_delta": filtered_after - filtered_before,
        })

    return rows


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("[WARN] No common images between --before-dir and --after-dir.")
        return

    n = len(rows)
    mean_before = sum(r["n_filtered_before"] for r in rows) / n
    mean_after = sum(r["n_filtered_after"] for r in rows) / n

    print(f"[OK] Compared {n} images.")
    print(f"     mean filtered detections before: {mean_before:.2f}")
    print(f"     mean filtered detections after:  {mean_after:.2f}")

    biggest_drops = sorted(rows, key=lambda r: r["filtered_delta"])[:5]
    print("     largest drops (image_id: before -> after):")
    for r in biggest_drops:
        print(f"       {r['image_id']}: {r['n_filtered_before']} -> {r['n_filtered_after']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-image detection counts between two Workflow B prediction "
            "runs (e.g. before/after paso 7 threshold and NMS changes)."
        )
    )
    parser.add_argument("--before-dir", required=True, help="Directory with the older run's JSON predictions.")
    parser.add_argument("--after-dir", required=True, help="Directory with the newer run's JSON predictions.")
    parser.add_argument("--output", required=True, help="Output CSV path for per-image counts.")

    args = parser.parse_args()

    rows = compare_directories(Path(args.before_dir), Path(args.after_dir))
    print_summary(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[OK] Per-image comparison saved to {output_path}")


if __name__ == "__main__":
    main()
