from __future__ import annotations

"""Allowed AudioSet nodes for optional Workflow A acoustic semantics."""

from typing import Any

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


def deterministic_nodes_from_visual_terms(
    visual_terms: list[str],
    rules: tuple[AudioSetLeafRule, ...] | list[AudioSetLeafRule],
) -> list[dict[str, Any]]:
    """Apply Workflow B's leaf rules in Python over ``visual_terms``, no VLM.

    Same matching as ``_match_leaf_rules`` in Workflow B's audioset projection,
    over a list of terms instead of detections: a rule fires when every
    ``all_of`` group has at least one of its alternatives in ``visual_terms``.
    Each fired rule becomes one ``visible_source`` node whose
    ``visual_evidence_terms`` are the terms (a subset of ``visual_terms``) that
    satisfied its groups. Returns node dicts in the shape the VLM would emit,
    so they go through the same normalization/validation.
    """
    present = set(visual_terms)
    nodes: list[dict[str, Any]] = []
    for rule in rules:
        if not all(any(term in present for term in group) for group in rule.all_of):
            continue
        firing_terms = sorted(
            {term for group in rule.all_of for term in group if term in present}
        )
        nodes.append({
            "node_id": f"n{len(nodes) + 1}",
            "audioset_id": rule.id,
            "audioset_name": rule.name,
            "node_type": "visible_source",
            "evidence": f"Leaf rule satisfied by: {', '.join(firing_terms)}",
            "visual_evidence_terms": firing_terms,
        })
    return nodes


def default_allowed_visual_terms() -> tuple[str, ...]:
    """Return the shared 210-term visual vocabulary Workflow B's detectors also use.

    Stage 1 of the audioset-core prompt restricts the VLM's observed visual terms
    to this list, so its stage-2 AudioSet nodes stay traceable to a term it
    actually committed to.
    """
    return audioset_detectable_terms()


def format_allowed_visual_terms_for_prompt(terms: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"- {term}" for term in terms)
