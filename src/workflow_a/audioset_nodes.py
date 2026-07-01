from __future__ import annotations

"""Allowed AudioSet nodes for optional Workflow A acoustic semantics."""

from functools import lru_cache

from metrics.audioset_ontology import load_audioset_ontology
from metrics.audioset_semantics import AUDIOSET_CONCEPT_TO_NODE_NAME


@lru_cache(maxsize=1)
def default_allowed_audioset_nodes() -> tuple[dict[str, str], ...]:
    """Return a deterministic, prompt-sized allowed list of ontology nodes.

    The VLM must choose only from this list. It is intentionally conservative
    and aligned with the ontology-backed evaluation concepts used elsewhere in
    the repository.
    """

    ontology = load_audioset_ontology()
    node_ids = {
        node_id
        for node_name in AUDIOSET_CONCEPT_TO_NODE_NAME.values()
        if (node_id := ontology.resolve_name(node_name))
    }

    return tuple(
        {"id": node_id, "name": ontology.node_by_id[node_id].name}
        for node_id in sorted(node_ids, key=lambda item: ontology.node_by_id[item].name)
    )


def format_allowed_audioset_nodes_for_prompt(
    nodes: tuple[dict[str, str], ...] | list[dict[str, str]],
) -> str:
    return "\n".join(f"- {node['id']} | {node['name']}" for node in nodes)
