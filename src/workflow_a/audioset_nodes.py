from __future__ import annotations

"""Allowed AudioSet nodes for optional Workflow A acoustic semantics."""

from metrics.audioset_leaf_vocab import allowed_audioset_leaf_nodes, audioset_detectable_terms


def default_allowed_audioset_nodes() -> tuple[dict[str, str], ...]:
    """Return the shared ontology-backed AudioSet leaf-node label set.

    The VLM must choose only from this list. It is derived from the same
    detectable-evidence mapping used by Workflow B.
    """
    return allowed_audioset_leaf_nodes()


def format_allowed_audioset_nodes_for_prompt(
    nodes: tuple[dict[str, str], ...] | list[dict[str, str]],
) -> str:
    return "\n".join(f"- {node['id']} | {node['name']}" for node in nodes)


def default_allowed_visual_terms() -> tuple[str, ...]:
    """Return the shared 210-term visual vocabulary Workflow B's detectors also use.

    Stage 1 of the audioset-core prompt restricts the VLM's observed visual terms
    to this list, so its stage-2 AudioSet nodes stay traceable to a term it
    actually committed to.
    """
    return audioset_detectable_terms()


def format_allowed_visual_terms_for_prompt(terms: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {term}" for term in terms)
