from __future__ import annotations

"""Utilities for deterministic AudioSet ontology traversal.

The AudioSet ontology is a sound taxonomy, not a Visual Genome label space.
Callers in this repository use it to project visual objects and relationships
onto acoustic-semantic pseudo-labels for analysis. No model inference happens
here; all mappings are deterministic and ontology based.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from scripts.evaluation.vg_utils import normalize_text


@dataclass(frozen=True)
class AudioSetNode:
    id: str
    name: str
    description: str
    child_ids: tuple[str, ...]
    restrictions: frozenset[str]

    @property
    def is_blacklisted(self) -> bool:
        return "blacklist" in self.restrictions

    @property
    def is_abstract(self) -> bool:
        return "abstract" in self.restrictions


class AudioSetOntology:
    """Indexed AudioSet tree with blacklist-aware label utilities."""

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.node_by_id: dict[str, AudioSetNode] = {}

        for raw in nodes:
            node = AudioSetNode(
                id=raw["id"],
                name=raw["name"],
                description=raw.get("description", ""),
                child_ids=tuple(raw.get("child_ids", [])),
                restrictions=frozenset(raw.get("restrictions", [])),
            )
            self.node_by_id[node.id] = node

        self.node_name_to_id: dict[str, str] = {}
        for node in self.node_by_id.values():
            self.node_name_to_id[normalize_text(node.name)] = node.id
            for alias in node.name.split(","):
                normalized_alias = normalize_text(alias)
                if normalized_alias:
                    self.node_name_to_id.setdefault(normalized_alias, node.id)

        self.children_map: dict[str, list[str]] = {
            node_id: [
                child_id
                for child_id in node.child_ids
                if child_id in self.node_by_id
            ]
            for node_id, node in self.node_by_id.items()
        }

        self.parent_map: dict[str, str] = {}
        for node_id, child_ids in self.children_map.items():
            for child_id in child_ids:
                self.parent_map[child_id] = node_id

        self.root_ids: list[str] = sorted(
            node_id
            for node_id in self.node_by_id
            if node_id not in self.parent_map
        )
        self.leaf_node_ids: list[str] = sorted(
            node_id
            for node_id, node in self.node_by_id.items()
            if not self.children_map.get(node_id)
            and not node.is_blacklisted
            and not node.is_abstract
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "AudioSetOntology":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def is_usable_label(self, node_id: str) -> bool:
        """Return whether a node can be used as a prediction/evaluation label."""

        node = self.node_by_id.get(node_id)
        return bool(node and not node.is_blacklisted)

    def resolve_name(self, name: str) -> str | None:
        node_id = self.node_name_to_id.get(normalize_text(name))
        if node_id and self.is_usable_label(node_id):
            return node_id
        return None

    def ancestors(self, node_id: str, *, include_self: bool = False) -> list[str]:
        path = [node_id] if include_self and node_id in self.node_by_id else []
        current = node_id

        while current in self.parent_map:
            current = self.parent_map[current]
            path.append(current)

        return path

    def path_to_root(self, node_id: str, *, include_self: bool = True) -> list[str]:
        return list(reversed(self.ancestors(node_id, include_self=include_self)))

    def path_names_to_root(self, node_id: str, *, include_self: bool = True) -> list[str]:
        return [
            self.node_by_id[path_node_id].name
            for path_node_id in self.path_to_root(node_id, include_self=include_self)
        ]

    def parent(self, node_id: str) -> str | None:
        return self.parent_map.get(node_id)

    def parent_or_self(self, node_id: str) -> str | None:
        if node_id not in self.node_by_id:
            return None
        return self.parent_map.get(node_id, node_id)

    def top_level(self, node_id: str) -> str | None:
        path = self.path_to_root(node_id)
        return path[0] if path else None

    def lowest_common_ancestor(self, left_id: str, right_id: str) -> str | None:
        left_path = self.path_to_root(left_id)
        right_path = self.path_to_root(right_id)
        lca = None

        for left_node_id, right_node_id in zip(left_path, right_path):
            if left_node_id != right_node_id:
                break
            lca = left_node_id

        return lca

    def depth(self, node_id: str) -> int:
        path = self.path_to_root(node_id)
        return max(0, len(path) - 1)

    def tree_distance(self, left_id: str, right_id: str) -> int | None:
        lca = self.lowest_common_ancestor(left_id, right_id)
        if lca is None:
            return None
        return self.depth(left_id) + self.depth(right_id) - (2 * self.depth(lca))

    def lca_similarity(self, left_id: str, right_id: str) -> float:
        lca = self.lowest_common_ancestor(left_id, right_id)
        if lca is None:
            return 0.0

        left_depth = self.depth(left_id)
        right_depth = self.depth(right_id)
        denominator = left_depth + right_depth
        if denominator == 0:
            return 1.0
        return (2 * self.depth(lca)) / denominator

    def tree_distance_similarity(self, left_id: str, right_id: str) -> float:
        distance = self.tree_distance(left_id, right_id)
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + distance)


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ontology.json"


@lru_cache(maxsize=4)
def load_audioset_ontology(path: str | Path | None = None) -> AudioSetOntology:
    return AudioSetOntology.from_path(path or default_ontology_path())
