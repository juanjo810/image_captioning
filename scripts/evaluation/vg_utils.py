from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_alias_map(path: str | Path) -> dict[str, str]:
    """Load Visual Genome alias file.

    Each line:
        canonical,alias1,alias2,...

    Returns:
        alias -> canonical
    """

    alias_map: dict[str, str] = {}

    path = Path(path)

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        parts = [normalize_text(p) for p in line.split(",") if normalize_text(p)]

        if not parts:
            continue

        canonical = parts[0]

        for alias in parts:
            alias_map[alias] = canonical

    return alias_map


def canonicalize(text: str, alias_map: dict[str, str] | None = None) -> str:
    text = normalize_text(text)

    if alias_map is None:
        return text

    return alias_map.get(text, text)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_prediction_entities(
    pred_json: dict[str, Any],
    object_alias: dict[str, str] | None = None,
) -> set[str]:
    entities = pred_json.get("core", {}).get("entities", [])

    return {
        canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("label")
    }


def get_prediction_interactions(
    pred_json: dict[str, Any],
    object_alias: dict[str, str] | None = None,
    relationship_alias: dict[str, str] | None = None,
) -> set[tuple[str, str, str]]:
    core = pred_json.get("core", {})
    entities = core.get("entities", [])
    interactions = core.get("observed_interactions", [])

    entity_by_id = {
        entity.get("id"): canonicalize(entity.get("label", ""), object_alias)
        for entity in entities
        if entity.get("id") and entity.get("label")
    }

    triples = set()

    for rel in interactions:
        subj = entity_by_id.get(rel.get("subject_id"))
        obj = entity_by_id.get(rel.get("object_id"))
        pred = canonicalize(rel.get("verb", ""), relationship_alias)

        if subj and pred and obj:
            triples.add((subj, pred, obj))

    return triples
