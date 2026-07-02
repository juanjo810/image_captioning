from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.audioset_ontology import load_audioset_ontology


def main() -> None:
    ontology = load_audioset_ontology()
    stats = ontology.validate_integrity()

    for key in (
        "total_nodes",
        "root_nodes",
        "multi_parent_nodes",
        "missing_child_refs",
        "cycle_nodes",
    ):
        print(f"{key}: {stats[key]}")

    if stats["missing_child_refs"] or stats["cycle_nodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
