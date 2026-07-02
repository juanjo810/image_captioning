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
from collections import deque
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
    """Indexed AudioSet DAG with blacklist-aware label utilities."""

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

        self.missing_child_refs: list[tuple[str, str]] = [
            (node_id, child_id)
            for node_id, node in self.node_by_id.items()
            for child_id in node.child_ids
            if child_id not in self.node_by_id
        ]
        self.children_map: dict[str, list[str]] = {
            node_id: [
                child_id
                for child_id in node.child_ids
                if child_id in self.node_by_id
            ]
            for node_id, node in self.node_by_id.items()
        }

        self.parents_map: dict[str, set[str]] = {
            node_id: set()
            for node_id in self.node_by_id
        }
        for node_id, child_ids in self.children_map.items():
            for child_id in child_ids:
                self.parents_map[child_id].add(node_id)

        self.root_ids: list[str] = sorted(
            node_id
            for node_id in self.node_by_id
            if not self.parents_map[node_id]
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

    def parents(self, node_id: str) -> set[str]:
        return set(self.parents_map.get(node_id, set()))

    def parents_or_self(self, node_id: str) -> set[str]:
        if node_id not in self.node_by_id:
            return set()
        parents = self.parents(node_id)
        return parents or {node_id}

    def ancestors(self, node_id: str, *, include_self: bool = False) -> set[str]:
        if node_id not in self.node_by_id:
            return set()

        ancestors: set[str] = {node_id} if include_self else set()
        stack = list(self.parents_map[node_id])

        while stack:
            current = stack.pop()
            if current in ancestors:
                continue
            ancestors.add(current)
            stack.extend(self.parents_map.get(current, set()) - ancestors)

        return ancestors

    def ancestor_distances(
        self,
        node_id: str,
        *,
        include_self: bool = True,
    ) -> dict[str, int]:
        if node_id not in self.node_by_id:
            return {}

        distances: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])

        while queue:
            current, distance = queue.popleft()
            previous_distance = distances.get(current)
            if previous_distance is not None and previous_distance <= distance:
                continue

            distances[current] = distance
            for parent_id in sorted(self.parents_map.get(current, set())):
                queue.append((parent_id, distance + 1))

        if not include_self:
            distances.pop(node_id, None)
        return distances

    def paths_to_roots(
        self,
        node_id: str,
        *,
        include_self: bool = True,
    ) -> list[list[str]]:
        if node_id not in self.node_by_id:
            return []

        paths: list[list[str]] = []
        stack: list[tuple[str, list[str]]] = [(node_id, [node_id])]

        while stack:
            current, path_from_node = stack.pop()
            parents = sorted(self.parents_map.get(current, set()))
            if not parents:
                path = list(reversed(path_from_node))
                if not include_self:
                    path = path[:-1]
                paths.append(path)
                continue

            for parent_id in reversed(parents):
                if parent_id in path_from_node:
                    continue
                stack.append((parent_id, path_from_node + [parent_id]))

        return sorted(paths, key=lambda path: [self.node_by_id[node_id].name for node_id in path])

    def path_names_to_roots(
        self,
        node_id: str,
        *,
        include_self: bool = True,
    ) -> list[list[str]]:
        return [
            [self.node_by_id[path_node_id].name for path_node_id in path]
            for path in self.paths_to_roots(node_id, include_self=include_self)
        ]

    def top_levels(self, node_id: str) -> set[str]:
        if node_id not in self.node_by_id:
            return set()
        return {
            path[0]
            for path in self.paths_to_roots(node_id)
            if path
        }

    def path_to_root(self, node_id: str, *, include_self: bool = True) -> list[str]:
        paths = self.paths_to_roots(node_id, include_self=include_self)
        return paths[0] if paths else []

    def path_names_to_root(self, node_id: str, *, include_self: bool = True) -> list[str]:
        paths = self.path_names_to_roots(node_id, include_self=include_self)
        return paths[0] if paths else []

    def parent(self, node_id: str) -> str | None:
        parents = sorted(self.parents_map.get(node_id, set()))
        return parents[0] if parents else None

    def parent_or_self(self, node_id: str) -> str | None:
        if node_id not in self.node_by_id:
            return None
        return self.parent(node_id) or node_id

    def top_level(self, node_id: str) -> str | None:
        top_levels = sorted(self.top_levels(node_id))
        return top_levels[0] if top_levels else None

    def lowest_common_ancestor(self, left_id: str, right_id: str) -> str | None:
        common_ancestors = (
            set(self.ancestor_distances(left_id))
            & set(self.ancestor_distances(right_id))
        )
        if not common_ancestors:
            return None
        return max(
            common_ancestors,
            key=lambda ancestor_id: (
                self.depth(ancestor_id),
                self.node_by_id[ancestor_id].name,
                ancestor_id,
            ),
        )

    def depth(self, node_id: str) -> int:
        paths = self.paths_to_roots(node_id)
        if not paths:
            return 0
        return max(0, max(len(path) for path in paths) - 1)

    def tree_distance(self, left_id: str, right_id: str) -> int | None:
        left_distances = self.ancestor_distances(left_id)
        right_distances = self.ancestor_distances(right_id)
        common_ancestors = set(left_distances) & set(right_distances)
        if not common_ancestors:
            return None
        return min(
            left_distances[ancestor_id] + right_distances[ancestor_id]
            for ancestor_id in common_ancestors
        )

    def lca_similarity(self, left_id: str, right_id: str) -> float:
        common_ancestors = (
            set(self.ancestor_distances(left_id))
            & set(self.ancestor_distances(right_id))
        )
        if not common_ancestors:
            return 0.0

        left_depth = self.depth(left_id)
        right_depth = self.depth(right_id)
        denominator = left_depth + right_depth
        if denominator == 0:
            return 1.0
        return max(
            (2 * self.depth(ancestor_id)) / denominator
            for ancestor_id in common_ancestors
        )

    def tree_distance_similarity(self, left_id: str, right_id: str) -> float:
        distance = self.tree_distance(left_id, right_id)
        if distance is None:
            return 0.0
        return 1.0 / (1.0 + distance)

    def validate_integrity(self) -> dict[str, Any]:
        cycle_nodes: set[str] = set()

        def visit(node_id: str, path: set[str]) -> None:
            if node_id in path:
                cycle_nodes.add(node_id)
                return
            next_path = path | {node_id}
            for child_id in self.children_map.get(node_id, []):
                visit(child_id, next_path)

        for node_id in self.node_by_id:
            visit(node_id, set())

        return {
            "total_nodes": len(self.node_by_id),
            "root_nodes": len(self.root_ids),
            "multi_parent_nodes": sum(
                1 for parent_ids in self.parents_map.values() if len(parent_ids) > 1
            ),
            "missing_child_refs": len(self.missing_child_refs),
            "cycle_nodes": len(cycle_nodes),
        }


def default_ontology_path() -> Path:
    return Path(__file__).resolve().parents[1] / "ontology.json"


@lru_cache(maxsize=4)
def load_audioset_ontology(path: str | Path | None = None) -> AudioSetOntology:
    return AudioSetOntology.from_path(path or default_ontology_path())
