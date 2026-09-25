from __future__ import annotations

"""Allowed AudioSet nodes for optional Workflow A acoustic semantics."""

from metrics.audioset_leaf_vocab import (
    AudioSetLeafRule,
    allowed_audioset_leaf_nodes,
    audioset_detectable_terms,
)


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


def format_audioset_leaf_rules_for_prompt(rules: tuple[AudioSetLeafRule, ...] | list[AudioSetLeafRule]) -> str:
    """Render each leaf rule as ``- id | name | requires: ...``.

    Uses the parsed ``all_of`` groups (already normalized and alias-expanded,
    the same terms Workflow B's projection matches against) rather than the
    raw evidence text, so every term shown is spelled exactly as it appears
    in the 210-term visual vocabulary.
    """

    def format_group(alternatives: tuple[str, ...]) -> str:
        if len(alternatives) == 1:
            return alternatives[0]
        return "(" + " OR ".join(alternatives) + ")"

    return "\n".join(
        f"- {rule.id} | {rule.name} | requires: "
        + " AND ".join(format_group(group) for group in rule.all_of)
        for rule in rules
    )


def default_allowed_visual_terms() -> tuple[str, ...]:
    """Return the shared 210-term visual vocabulary Workflow B's detectors also use.

    Stage 1 of the audioset-core prompt restricts the VLM's observed visual terms
    to this list, so its stage-2 AudioSet nodes stay traceable to a term it
    actually committed to.
    """
    return audioset_detectable_terms()


def format_allowed_visual_terms_for_prompt(terms: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {term}" for term in terms)
