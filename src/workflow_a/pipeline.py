from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.workflow_a.audioset_nodes import (
    default_allowed_audioset_nodes,
    default_allowed_visual_terms,
)
from src.workflow_a.parser import extract_json_block
from src.workflow_a.prompt_builder import (
    build_workflow_a_legacy_visual_prompt,
    build_workflow_a_audioset_core_prompt,
    build_workflow_a_audioset_core_free_scene_prompt,
    build_workflow_a_scene_mapping_prompt,
    build_workflow_a_audioset_core_scene_prompt,
    build_workflow_a_audioset_core_nodes_prompt,
    build_workflow_a_audioset_core_caption_prompt,
    build_workflow_a_free_visual_prompt,
    build_workflow_a_visual_terms_mapping_prompt,
)
from src.workflow_b.places365_mapping import allowed_places365_labels
from src.workflow_a.validator import (
    normalize_label,
    validate_legacy_acoustic_semantics_output,
    validate_legacy_workflow_a_output,
    validate_audioset_workflow_a_output,
)


@dataclass
class _StagedOutput:
    """Uniform intermediate state for every non-legacy call_mode ('single',
    'three', 'two', 'five'). Each _generate_*_call method parses its own VLM
    response(s) and converges on this shape, so _build_parsed_payload doesn't
    need to branch by call_mode: it always receives the same five fields and
    assembles the same core-shaped dict. extra_metadata carries call_mode-
    specific stats (currently only 'five''s n_free_visual_terms/
    n_mapped_visual_terms) that get folded into the output JSON's metadata
    block without _build_metadata needing to know which mode produced them."""

    raw_output: str
    visual_terms: list[str]
    scene: dict[str, Any]
    nodes: list[dict[str, Any]]
    caption: str | None
    acoustic_caption: str | None = None
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class WorkflowAPipeline:
    def __init__(self, vlm_adapter) -> None:
        self.vlm = vlm_adapter

    def run(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        max_new_tokens: int = 512,
        include_audioset_nodes: bool = False,
        use_legacy_core: bool = False,
        call_mode: str = "single",
        visual_terms_mapping: str = "discard",
    ) -> dict[str, Any]:
        if visual_terms_mapping not in ("discard", "force"):
            raise ValueError(
                f"visual_terms_mapping must be 'discard' or 'force', got {visual_terms_mapping!r}"
            )

        generators = {
            "single": self._generate_single_call,
            "three": self._generate_three_call,
            "two": self._generate_two_call,
            # Only call 3 of 'five' reads visual_terms_mapping (see
            # _generate_five_call) -- bound here via a small wrapper instead
            # of threading an unused parameter through the other three
            # generators, which otherwise share an identical signature.
            "five": lambda *args: self._generate_five_call(
                *args, visual_terms_mapping=visual_terms_mapping,
            ),
        }
        if call_mode not in generators:
            raise ValueError(f"call_mode must be one of {sorted(generators)}, got {call_mode!r}")

        image_path = Path(image_path)
        output_dir = Path(output_dir) / "legacy" if use_legacy_core else Path(output_dir)
        dirs = self._make_output_dirs(output_dir)

        allowed_audioset_nodes: tuple = ()
        allowed_scene_labels: tuple = ()
        staged: _StagedOutput | None = None

        if use_legacy_core:
            allowed_audioset_nodes = default_allowed_audioset_nodes() if include_audioset_nodes else ()
            raw_output = self._generate_legacy(
                image_path, include_audioset_nodes, allowed_audioset_nodes, max_new_tokens
            )
        else:
            allowed_audioset_nodes = default_allowed_audioset_nodes()
            allowed_scene_labels = allowed_places365_labels()
            allowed_visual_terms = default_allowed_visual_terms()

            staged = generators[call_mode](
                image_path, allowed_audioset_nodes, allowed_visual_terms,
                allowed_scene_labels, max_new_tokens,
            )
            raw_output = staged.raw_output

        raw_path = dirs["raw"] / f"{image_path.stem}.txt"
        raw_path.write_text(raw_output, encoding="utf-8")

        try:
            parsed = (
                self._build_parsed_payload(image_path.stem, staged)
                if staged is not None
                else extract_json_block(raw_output)
            )
            core, acoustic_semantics = self._validate(
                parsed,
                image_id=image_path.stem,
                use_legacy_core=use_legacy_core,
                include_audioset_nodes=include_audioset_nodes,
                allowed_audioset_nodes=allowed_audioset_nodes,
                allowed_scene_labels=allowed_scene_labels,
            )
        except Exception as exc:
            self._write_failure(dirs["failed"], image_path.stem, exc, raw_path)
            raise

        metadata = self._build_metadata(
            image_path.stem, use_legacy_core, include_audioset_nodes,
            call_mode, allowed_audioset_nodes, max_new_tokens,
            extra_metadata=staged.extra_metadata if staged is not None else {},
        )
        return self._write_outputs(dirs, image_path, core, acoustic_semantics, metadata)


    # -- generation -----------------------------------------------------------

    def _generate_legacy(
        self,
        image_path,
        include_audioset_nodes,
        allowed_audioset_nodes,
        max_new_tokens,
    ) -> str:
        prompt = build_workflow_a_legacy_visual_prompt(
            include_audioset_nodes=include_audioset_nodes,
            allowed_audioset_nodes=allowed_audioset_nodes,
        )
        return self.vlm.generate(image_path=image_path, prompt=prompt, max_new_tokens=max_new_tokens)


    @staticmethod
    def _unwrap_core_fields(raw: str) -> dict[str, Any]:
        """Parse a VLM response that may wrap its fields in a top-level "core"
        key (as 'single' and 'two''s phase 1 do) or not (every other stage/
        call, which each return a narrower schema of their own). Falls back
        to the parsed payload itself when there is no "core" dict."""
        parsed = extract_json_block(raw)
        core = parsed.get("core") if isinstance(parsed.get("core"), dict) else parsed
        return core


    def _generate_single_call(
        self,
        image_path,
        allowed_audioset_nodes,
        allowed_visual_terms,
        allowed_scene_labels,
        max_new_tokens,
    ) -> _StagedOutput:
        prompt = build_workflow_a_audioset_core_prompt(
            allowed_audioset_nodes=allowed_audioset_nodes,
            allowed_visual_terms=allowed_visual_terms,
            allowed_scene_labels=allowed_scene_labels,
        )
        raw_output = self.vlm.generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )
        core = self._unwrap_core_fields(raw_output)
        return _StagedOutput(
            raw_output=raw_output,
            visual_terms=[term for term in (core.get("visual_terms") or []) if isinstance(term, str)],
            scene=core.get("scene") if isinstance(core.get("scene"), dict) else {},
            nodes=core.get("nodes") if isinstance(core.get("nodes"), list) else [],
            caption=core.get("caption"),
            acoustic_caption=core.get("acoustic_caption"),
        )


    def _generate_three_call(
        self,
        image_path,
        allowed_audioset_nodes,
        allowed_visual_terms,
        allowed_scene_labels,
        max_new_tokens,
    ) -> _StagedOutput:
        """Stage 1 (scene + visual_terms) -> stage 2 (nodes, derived from stage
        1's visual_terms) -> stage 3 (caption + acoustic_caption, text-only).
        Each stage is an independent VLM call, so -- unlike the single-call
        flow -- there is no autoregressive conditioning keeping a later stage
        consistent with an earlier one."""
        stage1_prompt = build_workflow_a_audioset_core_scene_prompt(
            allowed_visual_terms=allowed_visual_terms,
            allowed_scene_labels=allowed_scene_labels,
        )
        stage1_raw = self.vlm.generate(
            image_path=image_path, prompt=stage1_prompt, max_new_tokens=max_new_tokens
        )
        stage1_parsed = extract_json_block(stage1_raw)
        visual_terms = [
            term for term in (stage1_parsed.get("visual_terms") or [])
            if isinstance(term, str)
        ]
        scene = (
            stage1_parsed.get("scene")
            if isinstance(stage1_parsed.get("scene"), dict)
            else {}
        )

        stage2_prompt = build_workflow_a_audioset_core_nodes_prompt(
            visual_terms=visual_terms,
            allowed_audioset_nodes=allowed_audioset_nodes,
            image_attached=True,
        )
        stage2_raw = self.vlm.generate(
            image_path=image_path, prompt=stage2_prompt, max_new_tokens=max_new_tokens
        )
        stage2_parsed = extract_json_block(stage2_raw)
        nodes = (
            stage2_parsed.get("nodes")
            if isinstance(stage2_parsed.get("nodes"), list)
            else []
        )

        # Everything stage 3 needs (scene/visual_terms/nodes) is already
        # decided and passed in as text below, so no image is attached --
        # attaching it would just invite the model to describe something it
        # re-notices instead of strictly summarizing the given data.
        stage3_prompt = build_workflow_a_audioset_core_caption_prompt(
            scene=scene, visual_terms=visual_terms, nodes=nodes,
        )
        stage3_raw = self.vlm.generate(
            image_path=None, prompt=stage3_prompt, max_new_tokens=max_new_tokens
        )
        stage3_parsed = extract_json_block(stage3_raw)

        raw_output = (
            "=== STAGE 1: scene + visual_terms ===\n" + stage1_raw
            + "\n\n=== STAGE 2: nodes ===\n" + stage2_raw
            + "\n\n=== STAGE 3: caption ===\n" + stage3_raw
        )
        return _StagedOutput(
            raw_output=raw_output,
            visual_terms=visual_terms,
            scene=scene,
            nodes=nodes,
            caption=stage3_parsed.get("caption"),
            acoustic_caption=stage3_parsed.get("acoustic_caption"),
        )


    def _generate_two_call(
        self,
        image_path,
        allowed_audioset_nodes,
        allowed_visual_terms,
        allowed_scene_labels,
        max_new_tokens,
    ) -> _StagedOutput:
        """Phase 1 (free scene description + visual_terms + nodes + caption +
        acoustic_caption, image attached, one call) -> phase 2 (text-only call
        mapping that free-form scene description onto the Places365
        allow-list). Unlike 'three', only the scene label itself is deferred
        to a second call -- visual_terms/nodes/caption are still decided
        together in one response, same as 'single'."""
        phase1_prompt = build_workflow_a_audioset_core_free_scene_prompt(
            allowed_audioset_nodes=allowed_audioset_nodes,
            allowed_visual_terms=allowed_visual_terms,
        )
        phase1_raw = self.vlm.generate(
            image_path=image_path, prompt=phase1_prompt, max_new_tokens=max_new_tokens
        )
        core1 = self._unwrap_core_fields(phase1_raw)
        visual_terms = [
            term for term in (core1.get("visual_terms") or [])
            if isinstance(term, str)
        ]
        free_scene = (
            core1.get("scene")
            if isinstance(core1.get("scene"), dict)
            else {}
        )
        nodes = (
            core1.get("nodes")
            if isinstance(core1.get("nodes"), list)
            else []
        )
        caption = core1.get("caption")
        acoustic_caption = core1.get("acoustic_caption")

        # Everything phase 2 needs (the free-form scene description and the
        # visual_terms already committed to) is passed in as text below, so
        # no image is attached -- this call is a lookup against the
        # Places365 allow-list, not a fresh look at the image.
        phase2_prompt = build_workflow_a_scene_mapping_prompt(
            free_scene=free_scene, visual_terms=visual_terms,
            allowed_scene_labels=allowed_scene_labels,
        )
        phase2_raw = self.vlm.generate(
            image_path=None, prompt=phase2_prompt, max_new_tokens=max_new_tokens
        )
        phase2_parsed = extract_json_block(phase2_raw)
        scene = (
            phase2_parsed.get("scene")
            if isinstance(phase2_parsed.get("scene"), dict)
            else {}
        )

        raw_output = (
            "=== PHASE 1: free scene + visual_terms + nodes + caption ===\n" + phase1_raw
            + "\n\n=== PHASE 1 SCENE (free-form, parsed) ===\n"
            + json.dumps(free_scene, ensure_ascii=False)
            + "\n\n=== PHASE 2: scene mapping ===\n" + phase2_raw
        )
        return _StagedOutput(
            raw_output=raw_output,
            visual_terms=visual_terms,
            scene=scene,
            nodes=nodes,
            caption=caption,
            acoustic_caption=acoustic_caption,
        )


    def _generate_five_call(
        self,
        image_path,
        allowed_audioset_nodes,
        allowed_visual_terms,
        allowed_scene_labels,
        max_new_tokens,
        *,
        visual_terms_mapping: str = "discard",
    ) -> _StagedOutput:
        """Call 1 (free visual_terms + free scene, image attached, the only
        call in this flow that sees the image and the only call in any
        call_mode with no closed list at all) -> call 2 (scene mapped onto
        Places365, text-only) -> call 3 (visual_terms mapped onto the
        210-term vocabulary, text-only) -> call 4 (nodes derived from the
        MAPPED visual_terms, text-only) -> call 5 (caption + acoustic_caption,
        text-only).

        The scene (call 2) is mapped from the FREE visual_terms, deliberately
        BEFORE visual_terms mapping (call 3): the 210-term vocabulary is built
        for detectable sound sources, not for discriminating Places365
        labels, so mapping visual_terms first would throw away exactly the
        evidence that most disambiguates the scene label.

        Nodes (call 4) are derived from the MAPPED visual_terms, not the free
        ones, since node evidence must cite terms from the closed
        AudioSet-detectable vocabulary. Call 4 does not receive the
        already-mapped scene either, by design -- that would reopen the same
        hallucination vector call_mode='two' closes: inferring objects
        typical of a scene label that were never actually declared as seen.

        Two cheap, pure-Python safety filters guard the boundary between
        calls, on top of the prompt-level instructions -- neither costs an
        extra VLM call:
        - After call 2, the mapped scene label is checked against the
          allowed Places365 list right away. Scene mapping is a forced 1-to-1
          pick (unlike visual-term mapping, which is allowed to drop terms),
          so a label that isn't an exact allow-list entry is guaranteed to
          fail final validation (validate_audioset_workflow_a_output) no
          matter what calls 3-5 produce. Calls 3-5 are skipped in that case
          and this returns early -- run() still writes raw/ and failed/ for
          this image exactly as it would if the rejection had only been
          caught at final validation, just without paying for 3 more calls.
        - After call 3, the mapped visual_terms are normalized, filtered
          against the allowed 210-term vocabulary, and deduplicated before
          being handed to call 4 -- the same filtering final validation would
          do anyway, just applied before call 4 spends its budget on a
          "visual-terms to infer from" list that might otherwise contain
          noise the model failed to map correctly.

        visual_terms_mapping picks call 3's prompt variant (see
        build_workflow_a_visual_terms_mapping_prompt): 'discard' (default,
        the team's chosen behavior) drops a free term with no real
        equivalent; 'force' maps every term onto its closest allowed entry
        instead, like scene mapping does. Exposed as a real option (not a
        one-off patch) so it can be A/B tested with a larger VLM and a
        larger batch before deciding whether to keep 'discard' as the
        default -- a 30-image pilot with a small local VLM found a genuine
        trade-off between the two, not a clear winner.
        """
        call1_prompt = build_workflow_a_free_visual_prompt()
        call1_raw = self.vlm.generate(
            image_path=image_path, prompt=call1_prompt, max_new_tokens=max_new_tokens
        )
        call1_parsed = extract_json_block(call1_raw)
        free_visual_terms = [
            term for term in (call1_parsed.get("visual_terms") or [])
            if isinstance(term, str)
        ]
        free_scene = (
            call1_parsed.get("scene")
            if isinstance(call1_parsed.get("scene"), dict)
            else {}
        )

        call2_prompt = build_workflow_a_scene_mapping_prompt(
            free_scene=free_scene, visual_terms=free_visual_terms,
            allowed_scene_labels=allowed_scene_labels,
        )
        call2_raw = self.vlm.generate(
            image_path=None, prompt=call2_prompt, max_new_tokens=max_new_tokens
        )
        call2_parsed = extract_json_block(call2_raw)
        scene = (
            call2_parsed.get("scene")
            if isinstance(call2_parsed.get("scene"), dict)
            else {}
        )

        calls_1_2_raw_output = (
            "=== CALL 1: free visual_terms + free scene ===\n" + call1_raw
            + "\n\n=== CALL 1 FREE TERMS (parsed) ===\n"
            + json.dumps(free_visual_terms, ensure_ascii=False)
            + "\n\n=== CALL 1 SCENE (free-form, parsed) ===\n"
            + json.dumps(free_scene, ensure_ascii=False)
            + "\n\n=== CALL 2: scene mapping ===\n" + call2_raw
        )

        allowed_scene_labels_normalized = {
            normalize_label(label, default="") for label in allowed_scene_labels
        }
        scene_label_normalized = normalize_label(
            scene.get("label") if isinstance(scene, dict) else None, default=""
        )
        if scene_label_normalized not in allowed_scene_labels_normalized:
            raw_output = (
                calls_1_2_raw_output
                + "\n\n=== SKIPPED: calls 3-5 (mapped scene label is not in the "
                "allowed Places365 list, so final validation would reject this "
                "image regardless of calls 3-5 -- skipped to save VLM calls) ==="
            )
            return _StagedOutput(
                raw_output=raw_output,
                visual_terms=[],
                scene=scene,
                nodes=[],
                caption=None,
                acoustic_caption=None,
                extra_metadata={
                    "n_free_visual_terms": len(free_visual_terms),
                    "visual_terms_mapping": visual_terms_mapping,
                },
            )

        call3_prompt = build_workflow_a_visual_terms_mapping_prompt(
            free_visual_terms=free_visual_terms, allowed_visual_terms=allowed_visual_terms,
            mapping_mode=visual_terms_mapping,
        )
        call3_raw = self.vlm.generate(
            image_path=None, prompt=call3_prompt, max_new_tokens=max_new_tokens
        )
        call3_parsed = extract_json_block(call3_raw)

        allowed_visual_terms_normalized = {
            normalize_label(term, default="") for term in allowed_visual_terms
        }
        visual_terms = []
        seen_visual_terms = set()
        for term in (call3_parsed.get("visual_terms") or []):
            if not isinstance(term, str):
                continue
            normalized_term = normalize_label(term, default="")
            if not normalized_term or normalized_term not in allowed_visual_terms_normalized:
                continue
            if normalized_term in seen_visual_terms:
                continue
            visual_terms.append(normalized_term)
            seen_visual_terms.add(normalized_term)

        if visual_terms:
            call4_prompt = build_workflow_a_audioset_core_nodes_prompt(
                visual_terms=visual_terms, allowed_audioset_nodes=allowed_audioset_nodes,
                image_attached=False,
            )
            call4_raw = self.vlm.generate(
                image_path=None, prompt=call4_prompt, max_new_tokens=max_new_tokens
            )
            call4_parsed = extract_json_block(call4_raw)
            nodes = (
                call4_parsed.get("nodes")
                if isinstance(call4_parsed.get("nodes"), list)
                else []
            )
        else:
            # Every node needs a non-empty visual_evidence_terms subset of
            # visual_terms, so an empty visual_terms list can only ever
            # produce nodes final validation would drop anyway -- skipped to
            # save a call. This also sidesteps a real failure mode observed
            # with the local Gemma model: handed an empty "visual-terms to
            # infer from" list, it doesn't reliably return {"nodes": []} and
            # sometimes emits a bare "{}" that fails to parse.
            call4_raw = "(skipped: visual_terms is empty after call 3, so no node could have valid evidence)"
            nodes = []

        call5_prompt = build_workflow_a_audioset_core_caption_prompt(
            scene=scene, visual_terms=visual_terms, nodes=nodes,
        )
        call5_raw = self.vlm.generate(
            image_path=None, prompt=call5_prompt, max_new_tokens=max_new_tokens
        )
        call5_parsed = extract_json_block(call5_raw)

        raw_output = (
            calls_1_2_raw_output
            + "\n\n=== CALL 3: visual_terms mapping ===\n" + call3_raw
            + "\n\n=== CALL 4: nodes ===\n" + call4_raw
            + "\n\n=== CALL 5: caption ===\n" + call5_raw
        )
        return _StagedOutput(
            raw_output=raw_output,
            visual_terms=visual_terms,
            scene=scene,
            nodes=nodes,
            caption=call5_parsed.get("caption"),
            acoustic_caption=call5_parsed.get("acoustic_caption"),
            extra_metadata={
                "n_free_visual_terms": len(free_visual_terms),
                "n_mapped_visual_terms": len(visual_terms),
                "visual_terms_mapping": visual_terms_mapping,
            },
        )


    # -- parsing / validation ---------------------------------------------------

    @staticmethod
    def _build_parsed_payload(image_id: str, staged: _StagedOutput) -> dict[str, Any]:
        core: dict[str, Any] = {
            "image_id": image_id,
            "visual_terms": staged.visual_terms,
            "scene": staged.scene,
            "nodes": staged.nodes,
            "caption": staged.caption,
        }
        if staged.acoustic_caption is not None:
            core["acoustic_caption"] = staged.acoustic_caption
        return {"core": core}


    @staticmethod
    def _validate(
        parsed: dict[str, Any],
        *,
        image_id: str,
        use_legacy_core: bool,
        include_audioset_nodes: bool,
        allowed_audioset_nodes,
        allowed_scene_labels,
    ):
        if use_legacy_core:
            core = validate_legacy_workflow_a_output(parsed, image_id=image_id)
            acoustic_semantics = (
                validate_legacy_acoustic_semantics_output(
                    parsed, allowed_audioset_nodes=allowed_audioset_nodes,
                )
                if include_audioset_nodes
                else None
            )
            return core, acoustic_semantics

        # Every non-legacy call_mode ('single', 'three', 'two', 'five')
        # converges on the same _StagedOutput-derived core-shaped dict here,
        # so the audioset-core validator doesn't need to know which call_mode
        # produced it.
        core = validate_audioset_workflow_a_output(
            parsed,
            image_id=image_id,
            allowed_audioset_nodes=allowed_audioset_nodes,
            allowed_scene_labels=allowed_scene_labels,
        )
        return core, None


    # -- output writing -----------------------------------------------------

    @staticmethod
    def _make_output_dirs(output_dir: Path) -> dict[str, Path]:
        dirs = {
            name: output_dir / name
            for name in ("raw", "json", "captions", "manifests", "failed")
        }
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        return dirs


    @staticmethod
    def _write_failure(
        failed_dir: Path,
        image_id: str,
        exc: Exception,
        raw_path: Path
    ) -> None:
        failed_payload = {"image_id": image_id, "error": str(exc), "raw_path": str(raw_path)}
        (failed_dir / f"{image_id}.json").write_text(
            json.dumps(failed_payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )


    def _build_metadata(
        self,
        image_id,
        use_legacy_core,
        include_audioset_nodes,
        call_mode,
        allowed_audioset_nodes,
        max_new_tokens,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if use_legacy_core:
            prompt_version = (
                "workflow_a_legacy_visual_core_audioset_v1"
                if include_audioset_nodes
                else "workflow_a_legacy_visual_core_v1"
            )
        else:
            prompt_version = {
                "single": "workflow_a_audioset_core_v1",
                "three": "workflow_a_audioset_core_three_call_v1",
                "two": "workflow_a_audioset_core_two_call_v1",
                "five": "workflow_a_audioset_core_five_call_v1",
            }[call_mode]

        metadata = {
            "workflow": "A",
            "model_id": getattr(self.vlm, "model_id", "unknown"),
            "adapter": self.vlm.__class__.__name__,
            "prompt_version": prompt_version,
            "call_mode": call_mode,
            "image_id": image_id,
            "include_audioset_nodes": bool(allowed_audioset_nodes),
            "use_legacy_core": use_legacy_core,
            "generation_params": {
                "max_new_tokens": max_new_tokens,
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if allowed_audioset_nodes:
            metadata["audioset_allowed_nodes"] = list(allowed_audioset_nodes)
        return metadata


    @staticmethod
    def _write_outputs(
        dirs: dict[str, Path],
        image_path: Path,
        core,
        acoustic_semantics,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "core": core.model_dump(exclude_none=True),
            "metadata": metadata,
        }
        if acoustic_semantics is not None:
            payload["acoustic_semantics"] = acoustic_semantics.model_dump()

        json_path = dirs["json"] / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        caption_path = dirs["captions"] / f"{image_path.stem}.txt"
        caption_path.write_text(core.caption, encoding="utf-8")

        manifest_record = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "caption_path": str(caption_path),
        }
        manifest_path = dirs["manifests"] / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

        return payload
