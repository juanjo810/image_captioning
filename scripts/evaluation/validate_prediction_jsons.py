from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from metrics.audioset_leaf_vocab import audioset_detectable_terms
from metrics.audioset_ontology import AudioSetOntology, load_audioset_ontology
from scripts.evaluation.vg_utils import is_audioset_core_json


def safe_load_json(path: Path) -> tuple[bool, dict | None, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return True, data, ""
    except Exception as exc:
        return False, None, str(exc)


def compute_audioset_json_metrics(
    json_path: Path,
    data: dict,
    ontology: AudioSetOntology,
) -> dict:
    core = data.get("core", {})
    extended = data.get("extended", {})

    nodes = core.get("nodes", [])
    grounding = extended.get("grounding", [])

    node_ids = {n.get("node_id") for n in nodes if n.get("node_id")}
    grounding_node_ids = {g.get("node_id") for g in grounding if g.get("node_id")}

    allowed_visual_terms = set(audioset_detectable_terms())
    declared_visual_terms = set(core.get("visual_terms") or [])

    valid_audioset_id_count = 0
    nodes_with_visual_evidence_terms = 0
    valid_visual_evidence_terms_count = 0
    subset_of_visual_terms_count = 0

    for node in nodes:
        audioset_id = str(node.get("audioset_id") or "").strip()
        if audioset_id and audioset_id in ontology.node_by_id:
            valid_audioset_id_count += 1

        terms = node.get("visual_evidence_terms")
        if terms:
            nodes_with_visual_evidence_terms += 1
            if set(terms) <= allowed_visual_terms:
                valid_visual_evidence_terms_count += 1
            if set(terms) <= declared_visual_terms:
                subset_of_visual_terms_count += 1

    n_nodes = len(nodes)

    return {
        "json_path": str(json_path),
        "image_id": core.get("image_id", json_path.stem),
        "json_valid": True,
        "json_error": "",
        "n_nodes": n_nodes,
        "n_grounding": len(grounding),
        "grounding_node_id_consistency": (
            grounding_node_ids.issubset(node_ids) if grounding_node_ids else True
        ),
        "audioset_id_valid_ratio": (
            valid_audioset_id_count / n_nodes if n_nodes > 0 else 1.0
        ),
        "visual_evidence_terms_valid_ratio": (
            valid_visual_evidence_terms_count / nodes_with_visual_evidence_terms
            if nodes_with_visual_evidence_terms > 0
            else 1.0
        ),
        # Should be 1.0 for any node whose evidence terms were filtered
        # against the declared visual_terms at generation time -- a ratio
        # below 1.0 means some node cites a term that was never declared as
        # visible, or the prediction predates visual_terms being recorded.
        "visual_evidence_terms_subset_of_visual_terms_ratio": (
            subset_of_visual_terms_count / nodes_with_visual_evidence_terms
            if nodes_with_visual_evidence_terms > 0
            else 1.0
        ),
        "n_visual_terms": len(core.get("visual_terms") or []),
        "has_caption": bool(core.get("caption")),
        "scene_label": core.get("scene", {}).get("label", ""),
        "scene_confidence": core.get("scene", {}).get("confidence", ""),
    }


def compute_legacy_json_metrics(json_path: Path, data: dict) -> dict:
    core = data.get("core", {})
    extended = data.get("extended", {})

    entities = core.get("entities", [])
    interactions = core.get("observed_interactions", [])
    entities_extended = extended.get("entities_extended", [])

    entity_ids = {e.get("id") for e in entities}
    extended_ids = {e.get("id") for e in entities_extended}

    interaction_valid_count = 0

    for inter in interactions:
        if (
            inter.get("subject_id") in entity_ids
            and inter.get("object_id") in entity_ids
            and inter.get("verb")
        ):
            interaction_valid_count += 1

    n_interactions = len(interactions)

    return {
        "json_path": str(json_path),
        "image_id": core.get("image_id", json_path.stem),
        "json_valid": True,
        "json_error": "",
        "n_entities": len(entities),
        "n_entities_extended": len(entities_extended),
        "n_interactions": n_interactions,
        "interaction_valid_ratio": (
            interaction_valid_count / n_interactions
            if n_interactions > 0
            else 1.0
        ),
        "entity_id_consistency": entity_ids.issubset(extended_ids) if extended_ids else True,
        "has_caption": bool(core.get("caption")),
        "scene_label": core.get("scene", {}).get("label", ""),
        "scene_confidence": core.get("scene", {}).get("confidence", ""),
    }


def compute_json_metrics(json_path: Path, ontology: AudioSetOntology) -> dict:
    valid, data, error = safe_load_json(json_path)

    if not valid or data is None:
        return {
            "json_path": str(json_path),
            "json_valid": False,
            "json_error": error,
        }

    if is_audioset_core_json(data):
        return compute_audioset_json_metrics(json_path, data, ontology)

    return compute_legacy_json_metrics(json_path, data)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Directory with prediction JSON files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    ontology = load_audioset_ontology()

    rows = [
        compute_json_metrics(path, ontology)
        for path in sorted(input_path.rglob("*.json"))
    ]

    if not rows:
        raise RuntimeError(f"No JSON files found in {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Structured metrics saved to {output_path}")
    print(f"[OK] Rows: {len(rows)}")


if __name__ == "__main__":
    main()