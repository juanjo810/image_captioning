"""Re-run only call 4 (visual_terms -> AudioSet nodes) of existing
call_mode=five outputs, with the rule-guided nodes prompt.

Calls 1-3 and 5 are not re-run: scene, mapped visual_terms and caption are
reused verbatim from --source-dir, and only the nodes are regenerated. The
old acoustic_caption described the old nodes, so it is dropped rather than
regenerated. Everything is written to --output-dir (never --source-dir), in
the same raw/json/captions/manifests/failed layout WorkflowAPipeline uses, so
the evaluation scripts work on it unchanged.

With --with-deterministic, the same loop also writes a second output where
nodes come from applying Workflow B's leaf rules to the same visual_terms in
Python, with no VLM involved, to --deterministic-output-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from metrics.audioset_leaf_vocab import AudioSetLeafRule, audioset_leaf_rules
from src.workflow_a.adapters.llamacpp_server_adapter import LlamaCppServerAdapter
from src.workflow_a.audioset_nodes import default_allowed_audioset_nodes
from src.workflow_a.parser import extract_json_block
from src.workflow_a.pipeline import WorkflowAPipeline, _StagedOutput
from src.workflow_a.prompt_builder import build_workflow_a_audioset_core_nodes_from_rules_prompt
from src.workflow_b.places365_mapping import allowed_places365_labels


LLM_PROMPT_VERSION_SUFFIX = "+call4_leaf_rules_v1"
DETERMINISTIC_PROMPT_VERSION_SUFFIX = "+call4_leaf_rules_deterministic_v1"


def rule_is_satisfied(rule: AudioSetLeafRule, terms: set[str]) -> bool:
    return all(any(term in terms for term in group) for group in rule.all_of)


def deterministic_nodes_from_visual_terms(
    visual_terms: list[str],
    rules: tuple[AudioSetLeafRule, ...],
) -> list[dict[str, Any]]:
    """Same matching as Workflow B's _match_leaf_rules, over a list of terms
    instead of detections: a rule fires when every all_of group has at least
    one term present."""
    present = set(visual_terms)
    nodes = []
    for rule in rules:
        if not rule_is_satisfied(rule, present):
            continue
        firing_terms = sorted({term for group in rule.all_of for term in group if term in present})
        nodes.append({
            "node_id": f"n{len(nodes) + 1}",
            "audioset_id": rule.id,
            "audioset_name": rule.name,
            "node_type": "visible_source",
            "evidence": f"Leaf rule satisfied by: {', '.join(firing_terms)}",
            "visual_evidence_terms": firing_terms,
        })
    return nodes


def count_rule_violations(core, rules_by_id: dict[str, AudioSetLeafRule]) -> int:
    """Nodes whose own rule is not satisfied by their visual_evidence_terms.
    Diagnostic only -- nothing is filtered on it."""
    violations = 0
    for node in core.nodes:
        rule = rules_by_id.get(node.audioset_id)
        if rule is None or not rule_is_satisfied(rule, set(node.visual_evidence_terms or [])):
            violations += 1
    return violations


def check_dirs_are_disjoint(dirs: dict[str, Path]) -> None:
    resolved = {name: path.resolve() for name, path in dirs.items()}
    names = list(resolved)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = resolved[a], resolved[b]
            if pa == pb or pa.is_relative_to(pb) or pb.is_relative_to(pa):
                raise SystemExit(
                    f"--{a} ({pa}) and --{b} ({pb}) must be different directories, "
                    "neither inside the other"
                )


def load_source_image_paths(source_dir: Path) -> dict[str, str]:
    manifest = source_dir / "manifests" / "manifest.jsonl"
    image_paths: dict[str, str] = {}
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                image_paths[record["image_id"]] = record["image_path"]
    return image_paths


def image_sort_key(image_id: str):
    return (0, int(image_id), "") if image_id.isdigit() else (1, 0, image_id)


def write_output(
    *,
    output_dir: Path,
    image_id: str,
    image_path: Path,
    source_core: dict[str, Any],
    source_metadata: dict[str, Any],
    nodes: list[dict[str, Any]],
    raw_output: str,
    extra_metadata: dict[str, Any],
    prompt_version_suffix: str,
    allowed_audioset_nodes,
    allowed_scene_labels,
    after_validation: Callable[[Any], dict[str, Any]] | None = None,
):
    dirs = WorkflowAPipeline._make_output_dirs(output_dir)
    raw_path = dirs["raw"] / f"{image_id}.txt"
    raw_path.write_text(raw_output, encoding="utf-8")

    staged = _StagedOutput(
        raw_output=raw_output,
        visual_terms=list(source_core.get("visual_terms") or []),
        scene=source_core.get("scene") or {},
        nodes=nodes,
        caption=source_core.get("caption"),
        acoustic_caption=None,
    )
    try:
        parsed = WorkflowAPipeline._build_parsed_payload(image_id, staged)
        core, _ = WorkflowAPipeline._validate(
            parsed,
            image_id=image_id,
            use_legacy_core=False,
            include_audioset_nodes=False,
            allowed_audioset_nodes=allowed_audioset_nodes,
            allowed_scene_labels=allowed_scene_labels,
        )
    except Exception as exc:
        WorkflowAPipeline._write_failure(dirs["failed"], image_id, exc, raw_path)
        raise

    metadata = dict(source_metadata)
    metadata["prompt_version"] = (
        source_metadata.get("prompt_version", "unknown") + prompt_version_suffix
    )
    metadata.update(extra_metadata)
    if after_validation is not None:
        metadata.update(after_validation(core))
    WorkflowAPipeline._write_outputs(dirs, image_path, core, None, metadata)
    return core


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-dir", required=True,
        help="Existing call_mode=five output directory (with json/, raw/, manifests/). Read only.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Where the rule-guided (VLM) call-4 outputs are written. Must not be --source-dir.",
    )
    parser.add_argument(
        "--with-deterministic", action="store_true",
        help=(
            "Also write a second output whose nodes come from applying Workflow B's "
            "leaf rules to the same visual_terms in Python (no VLM), to "
            "--deterministic-output-dir."
        ),
    )
    parser.add_argument(
        "--deterministic-output-dir", default=None,
        help="Defaults to a sibling of --output-dir named <output-dir>_deterministic.",
    )
    parser.add_argument("--server-url", default="http://localhost:8889")
    parser.add_argument("--model-id", default="local-vlm")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--image-ids", nargs="+", default=None,
        help="Only re-run these image ids (stems of json/<id>.json in --source-dir).",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    dirs_to_check = {"source-dir": source_dir, "output-dir": output_dir}
    deterministic_dir: Path | None = None
    if args.with_deterministic:
        deterministic_dir = (
            Path(args.deterministic_output_dir)
            if args.deterministic_output_dir
            else output_dir.with_name(output_dir.name + "_deterministic")
        )
        dirs_to_check["deterministic-output-dir"] = deterministic_dir
    check_dirs_are_disjoint(dirs_to_check)

    source_json_dir = source_dir / "json"
    if not source_json_dir.is_dir():
        raise SystemExit(f"{source_json_dir} does not exist")

    json_ids = {p.stem for p in source_json_dir.glob("*.json")}
    failed_only_ids = sorted(
        {p.stem for p in (source_dir / "failed").glob("*.json")} - json_ids,
        key=image_sort_key,
    )
    image_ids = sorted(json_ids, key=image_sort_key)
    if args.image_ids:
        wanted = set(args.image_ids)
        missing = wanted - json_ids
        if missing:
            print(f"Not in {source_json_dir}, skipped: {sorted(missing, key=image_sort_key)}")
        image_ids = [i for i in image_ids if i in wanted]
    if args.limit is not None:
        image_ids = image_ids[:args.limit]

    rules = audioset_leaf_rules()
    rules_by_id = {rule.id: rule for rule in rules}
    allowed_audioset_nodes = default_allowed_audioset_nodes()
    allowed_scene_labels = allowed_places365_labels()
    source_image_paths = load_source_image_paths(source_dir)
    adapter = LlamaCppServerAdapter(model_id=args.model_id, base_url=args.server_url)

    llm_failed: list[str] = []
    deterministic_failed: list[str] = []

    for image_id in image_ids:
        source_payload = json.loads((source_json_dir / f"{image_id}.json").read_text(encoding="utf-8"))
        source_core = source_payload["core"]
        source_metadata = source_payload.get("metadata") or {}
        visual_terms = list(source_core.get("visual_terms") or [])
        image_path = Path(source_image_paths.get(image_id, f"{image_id}.jpg"))
        raw_header = (
            f"=== CALL 4 RERUN ===\nsource: {source_dir}\n"
            "calls 1-3 and 5 reused from source (scene, visual_terms, caption); "
            "acoustic_caption dropped\n"
            f"visual_terms: {json.dumps(visual_terms, ensure_ascii=False)}\n\n"
        )
        common_metadata = {
            "call4_rerun": True,
            "call4_source_dir": str(source_dir),
            "acoustic_caption_dropped": True,
        }

        if (output_dir / "json" / f"{image_id}.json").exists():
            print(f"{image_id}: LLM output already exists, skipped")
        else:
            print(f"{image_id}: call 4 (LLM, leaf rules)")
            call4_raw: str | None = None
            try:
                if visual_terms:
                    prompt = build_workflow_a_audioset_core_nodes_from_rules_prompt(
                        visual_terms=visual_terms,
                        leaf_rules=rules,
                        schema_examples=source_metadata.get("schema_examples", "concrete"),
                    )
                    call4_raw = adapter.generate(
                        image_path=None, prompt=prompt, max_new_tokens=args.max_new_tokens,
                    )
                    call4_parsed = extract_json_block(call4_raw)
                    nodes = (
                        call4_parsed.get("nodes")
                        if isinstance(call4_parsed.get("nodes"), list)
                        else []
                    )
                else:
                    call4_raw = "(skipped: visual_terms is empty, so no node could have valid evidence)"
                    nodes = []
            except Exception as exc:
                # Server/parse errors: raw/ and failed/ are still written, and
                # since json/ isn't, re-running the same command retries only
                # this image.
                dirs = WorkflowAPipeline._make_output_dirs(output_dir)
                raw_path = dirs["raw"] / f"{image_id}.txt"
                raw_path.write_text(
                    raw_header + "=== CALL 4 (rerun, leaf rules) ===\n"
                    + (call4_raw if call4_raw is not None else f"(no response: {exc})"),
                    encoding="utf-8",
                )
                WorkflowAPipeline._write_failure(dirs["failed"], image_id, exc, raw_path)
                print(f"FAILED (LLM): {image_id} -> {exc}")
                llm_failed.append(image_id)
            else:
                try:
                    write_output(
                        output_dir=output_dir,
                        image_id=image_id,
                        image_path=image_path,
                        source_core=source_core,
                        source_metadata=source_metadata,
                        nodes=nodes,
                        raw_output=raw_header + "=== CALL 4 (rerun, leaf rules) ===\n" + call4_raw,
                        extra_metadata={**common_metadata, "call4_mode": "llm"},
                        prompt_version_suffix=LLM_PROMPT_VERSION_SUFFIX,
                        allowed_audioset_nodes=allowed_audioset_nodes,
                        allowed_scene_labels=allowed_scene_labels,
                        after_validation=lambda core: {
                            "n_rule_violations": count_rule_violations(core, rules_by_id),
                        },
                    )
                except Exception as exc:
                    print(f"FAILED (LLM): {image_id} -> {exc}")
                    llm_failed.append(image_id)

        if deterministic_dir is None:
            continue
        if (deterministic_dir / "json" / f"{image_id}.json").exists():
            print(f"{image_id}: deterministic output already exists, skipped")
            continue
        nodes = deterministic_nodes_from_visual_terms(visual_terms, rules)
        try:
            write_output(
                output_dir=deterministic_dir,
                image_id=image_id,
                image_path=image_path,
                source_core=source_core,
                source_metadata=source_metadata,
                nodes=nodes,
                raw_output=(
                    raw_header + "=== CALL 4 (deterministic leaf rules, no VLM) ===\n"
                    + json.dumps({"nodes": nodes}, indent=2, ensure_ascii=False)
                ),
                extra_metadata={**common_metadata, "call4_mode": "deterministic"},
                prompt_version_suffix=DETERMINISTIC_PROMPT_VERSION_SUFFIX,
                allowed_audioset_nodes=allowed_audioset_nodes,
                allowed_scene_labels=allowed_scene_labels,
            )
        except Exception as exc:
            print(f"FAILED (deterministic): {image_id} -> {exc}")
            deterministic_failed.append(image_id)

    print(f"\nProcessed {len(image_ids)} image(s) from {source_json_dir}")
    if failed_only_ids:
        print(f"Skipped (only in source failed/, no calls 1-3 to reuse): {failed_only_ids}")
    print(f"LLM failures: {llm_failed or 'none'}")
    if deterministic_dir is not None:
        print(f"Deterministic failures: {deterministic_failed or 'none'}")


if __name__ == "__main__":
    main()
