from __future__ import annotations

"""AudioSet leaf-node vocabulary derived from detectable visual evidence."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from metrics.audioset_ontology import AudioSetOntology, load_audioset_ontology


@dataclass(frozen=True)
class AudioSetLeafRule:
    id: str
    name: str
    evidence: str
    all_of: tuple[tuple[str, ...], ...]


_IGNORED_BLOCK_START = "FROM HERE"
_IGNORED_BLOCK_END = "TO HERE"
_NO_EVIDENCE = "no"

_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "forg": ("frog",),
    "toilet fulsh": ("toilet flush", "toilet"),
    "volcan": ("volcano",),
    "cd": ("compact disc",),
}


def default_leaf_vocab_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "audioset_leaf_node_names.txt"


def _normalize_detectable_term(term: str) -> str:
    term = term.replace("_", " ")
    term = re.sub(r"\([^)]*\)", "", term)
    term = term.replace("...", " ")
    term = re.sub(r"\s+", " ", term)
    return term.strip(" .").lower()


def _parse_evidence_expression(evidence: str) -> tuple[tuple[str, ...], ...]:
    """Parse simple ``and``/``or`` evidence into conjunctive groups."""

    groups: list[tuple[str, ...]] = []

    for raw_and_group in re.split(r"\s+and\s+", evidence, flags=re.IGNORECASE):
        alternatives: list[str] = []

        for raw_term in re.split(r"\s+or\s+", raw_and_group, flags=re.IGNORECASE):
            term = _normalize_detectable_term(raw_term)
            if not term:
                continue

            alternatives.append(term)
            alternatives.extend(_TERM_ALIASES.get(term, ()))

        deduped = tuple(dict.fromkeys(alternatives))
        if deduped:
            groups.append(deduped)

    return tuple(groups)


def _iter_leaf_vocab_rows(path: Path):
    ignore_block = False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue

        if _IGNORED_BLOCK_START in line:
            ignore_block = True
            continue

        if _IGNORED_BLOCK_END in line:
            ignore_block = False
            continue

        if ignore_block:
            continue

        if "|" not in line:
            continue

        node_name, evidence = line.split("|", 1)
        node_name = node_name.strip()
        evidence = evidence.strip()

        if not node_name or evidence.split("|", 1)[0].strip().lower() == _NO_EVIDENCE:
            continue

        yield line_number, node_name, evidence


def _resolve_usable_node_id(ontology: AudioSetOntology, node_name: str) -> str | None:
    return ontology.resolve_name(node_name)


@lru_cache(maxsize=4)
def audioset_leaf_rules(
    path: str | Path | None = None,
) -> tuple[AudioSetLeafRule, ...]:
    """Return inferible, non-blacklisted AudioSet leaf rules."""

    vocab_path = Path(path) if path is not None else default_leaf_vocab_path()
    ontology = load_audioset_ontology()
    rules: list[AudioSetLeafRule] = []
    seen_ids: set[str] = set()

    for _line_number, node_name, evidence in _iter_leaf_vocab_rows(vocab_path):
        node_id = _resolve_usable_node_id(ontology, node_name)
        if node_id is None or node_id in seen_ids:
            continue

        all_of = _parse_evidence_expression(evidence)
        if not all_of:
            continue

        rules.append(
            AudioSetLeafRule(
                id=node_id,
                name=ontology.node_by_id[node_id].name,
                evidence=evidence,
                all_of=all_of,
            )
        )
        seen_ids.add(node_id)

    return tuple(sorted(rules, key=lambda rule: rule.name))


@lru_cache(maxsize=4)
def allowed_audioset_leaf_nodes(
    path: str | Path | None = None,
) -> tuple[dict[str, str], ...]:
    """Return the shared AudioSet label set for Workflow A and Workflow B."""

    return tuple(
        {"id": rule.id, "name": rule.name}
        for rule in audioset_leaf_rules(path)
    )


@lru_cache(maxsize=4)
def audioset_detectable_terms(
    path: str | Path | None = None,
) -> tuple[str, ...]:
    """Return every visual term needed to evaluate the shared leaf rules."""

    terms: set[str] = set()

    for rule in audioset_leaf_rules(path):
        for alternatives in rule.all_of:
            terms.update(alternatives)

    return tuple(sorted(terms))
