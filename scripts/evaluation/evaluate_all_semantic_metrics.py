from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from metrics.audioset_semantics import compute_audioset_metrics
from scripts.evaluation.evaluate_scene_awareness import (
    compute_image_metrics as compute_scene_awareness_metrics,
    load_object_refs as load_scene_object_refs,
)
from scripts.evaluation.evaluate_soundscape_semantics import (
    configure_label_space,
    compute_image_metrics as compute_soundscape_metrics,
    load_object_refs,
    load_predictions,
    load_relationship_refs,
    write_csv,
)
from scripts.evaluation.evaluate_structured_vg import prf
from scripts.evaluation.vg_utils import (
    get_prediction_entities,
    get_prediction_interactions,
    load_alias_map,
    load_json,
)


SEMANTIC_CORE_FIELDS = {
    "image_id",
    "detector",
    "scene_model",
    "captioner",
    "scene_label",
    "scene_family",
    "background_family",
    "indoor_outdoor",
}

SCENE_AWARENESS_META_FIELDS = {
    "image_id",
    "detector",
    "scene_model",
    "captioner",
    "scene_label",
    "scene_group",
}

# These counts are useful for debugging individual images, but they make the
# summary CSV noisy and partly redundant with precision/recall/F1 metrics.
# They are therefore kept in --per-image-output and omitted from --output.
SUMMARY_DIAGNOSTIC_EXCLUDE_FIELDS = {
    "n_gt_entities",
    "n_pred_entities_norm",
    "n_gt_interactions",
    "n_pred_interactions_norm",
    "n_pred_families",
    "n_gt_families",
    "n_pred_discrete_families",
    "n_gt_discrete_families",
    "n_pred_interaction_families",
    "n_gt_interaction_families",
    "n_pred_audioset_tags",
    "n_gt_audioset_tags",
    "n_pred_audioset_categories",
    "n_gt_audioset_categories",
    "pred_audioset_tags",
    "gt_audioset_tags",
    "pred_audioset_categories",
    "gt_audioset_categories",
    "pred_audioset_tag_counts",
    "gt_audioset_tag_counts",
}


def compute_structured_vg_metrics(
    row: dict[str, str],
    object_refs: dict[str, set[str]],
    relationship_refs: dict[str, set[tuple[str, str, str]]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
    pred_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_id = row["image_id"]
    if pred_json is None:
        pred_json = load_json(row["json_path"])

    pred_entities = get_prediction_entities(pred_json, object_alias)
    gt_entities = object_refs.get(image_id, set())
    ep, er, ef1 = prf(pred_entities, gt_entities)

    pred_interactions = get_prediction_interactions(
        pred_json,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )
    gt_interactions = relationship_refs.get(image_id, set())
    ip, ir, if1 = prf(pred_interactions, gt_interactions)

    return {
        "vg_entity_precision": ep,
        "vg_entity_recall": er,
        "vg_entity_f1": ef1,
        "vg_interaction_precision": ip,
        "vg_interaction_recall": ir,
        "vg_interaction_f1": if1,
        "n_gt_entities": len(gt_entities),
        "n_pred_entities_norm": len(pred_entities),
        "n_gt_interactions": len(gt_interactions),
        "n_pred_interactions_norm": len(pred_interactions),
    }


def compute_image_all_semantic_metrics(
    row: dict[str, str],
    object_refs: dict[str, set[str]],
    relationship_refs: dict[str, set[tuple[str, str, str]]],
    scene_object_refs: dict[str, set[str]],
    object_alias: dict[str, str],
    relationship_alias: dict[str, str],
) -> dict[str, Any]:
    pred_json = load_json(row["json_path"])
    image_id = row["image_id"]
    gt_objects = object_refs.get(image_id, set())
    gt_relationships = relationship_refs.get(image_id, set())

    soundscape = compute_soundscape_metrics(
        row=row,
        object_refs=object_refs,
        relationship_refs=relationship_refs,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )

    audioset = compute_audioset_metrics(
        pred_json=pred_json,
        gt_objects=gt_objects,
        gt_relationships=gt_relationships,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
    )

    structured = compute_structured_vg_metrics(
        row=row,
        object_refs=object_refs,
        relationship_refs=relationship_refs,
        object_alias=object_alias,
        relationship_alias=relationship_alias,
        pred_json=pred_json,
    )

    scene_awareness = compute_scene_awareness_metrics(
        row=row,
        object_refs=scene_object_refs,
        object_alias=object_alias,
    )

    scene_awareness_metrics = {
        key: value
        for key, value in scene_awareness.items()
        if key not in SCENE_AWARENESS_META_FIELDS
    }

    return {
        **soundscape,
        **audioset,
        **structured,
        **scene_awareness_metrics,
    }


def aggregate_all_semantic_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[(row["detector"], row["scene_model"], row["captioner"])].append(row)

    metric_keys = [
        key for key in rows[0].keys()
        if key not in SEMANTIC_CORE_FIELDS
        and key not in SUMMARY_DIAGNOSTIC_EXCLUDE_FIELDS
    ]

    output = []

    for (detector, scene_model, captioner), items in grouped.items():
        result = {
            "detector": detector,
            "scene_model": scene_model,
            "captioner": captioner,
            "n": len(items),
        }

        for key in metric_keys:
            result[key] = sum(float(item[key]) for item in items) / len(items)

        output.append(result)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--vg-object-refs", required=True)
    parser.add_argument("--vg-relationship-refs", required=True)
    parser.add_argument("--object-alias", required=True)
    parser.add_argument("--relationship-alias", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--label-space",
        choices=["workflow_b", "open"],
        default="workflow_b",
        help=(
            "workflow_b evaluates only labels reachable by current detector prompts; "
            "open keeps extra future/VLM labels."
        ),
    )
    parser.add_argument(
        "--per-image-output",
        default=None,
        help="Optional path for per-image unified semantic metrics.",
    )

    args = parser.parse_args()

    configure_label_space(args.label_space)

    predictions = load_predictions(Path(args.manifest))
    object_refs = load_object_refs(Path(args.vg_object_refs))
    scene_object_refs = load_scene_object_refs(Path(args.vg_object_refs))
    relationship_refs = load_relationship_refs(Path(args.vg_relationship_refs))
    object_alias = load_alias_map(args.object_alias)
    relationship_alias = load_alias_map(args.relationship_alias)

    per_image_rows = [
        compute_image_all_semantic_metrics(
            row=row,
            object_refs=object_refs,
            relationship_refs=relationship_refs,
            scene_object_refs=scene_object_refs,
            object_alias=object_alias,
            relationship_alias=relationship_alias,
        )
        for row in predictions
    ]

    summary_rows = aggregate_all_semantic_metrics(per_image_rows)
    write_csv(summary_rows, Path(args.output))

    if args.per_image_output:
        write_csv(per_image_rows, Path(args.per_image_output))

    print(f"[OK] Unified semantic metrics saved to {args.output}")
    if args.per_image_output:
        print(f"[OK] Per-image unified semantic metrics saved to {args.per_image_output}")


if __name__ == "__main__":
    main()
