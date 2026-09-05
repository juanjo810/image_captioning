from __future__ import annotations

import json
from dataclasses import dataclass
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
    build_workflow_a_audioset_core_scene_prompt,
    build_workflow_a_audioset_core_nodes_prompt,
    build_workflow_a_audioset_core_caption_prompt,
)
from src.workflow_b.places365_mapping import allowed_places365_labels
from src.workflow_a.validator import (
    validate_legacy_acoustic_semantics_output,
    validate_legacy_workflow_a_output,
    validate_audioset_workflow_a_output,
)


@dataclass
class _ThreeCallStages:
    """Intermediate state threaded out of the three-call flow and into run()'s
    parse/validate step. stage3_raw is intentionally left unparsed here -- that
    happens inside run()'s try block, same as it always has, so a malformed
    stage-3 response still lands in failed/ instead of raising uncaught."""

    raw_output: str
    stage3_raw: str
    visual_terms: list[str]
    scene: dict[str, Any]
    nodes: list[dict[str, Any]]


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
    ) -> dict[str, Any]:
        if call_mode not in ("single", "three"):
            raise ValueError(f"call_mode must be 'single' or 'three', got {call_mode!r}")

        image_path = Path(image_path)
        output_dir = Path(output_dir) / "legacy" if use_legacy_core else Path(output_dir)
        dirs = self._make_output_dirs(output_dir)

        allowed_audioset_nodes: tuple = ()
        allowed_scene_labels: tuple = ()
        three_call_stages: _ThreeCallStages | None = None

        if use_legacy_core:
            allowed_audioset_nodes = default_allowed_audioset_nodes() if include_audioset_nodes else ()
            raw_output = self._generate_legacy(
                image_path, include_audioset_nodes, allowed_audioset_nodes, max_new_tokens
            )
        else:
            allowed_audioset_nodes = default_allowed_audioset_nodes()
            allowed_scene_labels = allowed_places365_labels()
            allowed_visual_terms = default_allowed_visual_terms()

            if call_mode == "single":
                raw_output = self._generate_single_call(
                    image_path, allowed_audioset_nodes, allowed_visual_terms,
                    allowed_scene_labels, max_new_tokens,
                )
            else:
                three_call_stages = self._generate_three_call(
                    image_path, allowed_audioset_nodes, allowed_visual_terms,
                    allowed_scene_labels, max_new_tokens,
                )
                raw_output = three_call_stages.raw_output

        raw_path = dirs["raw"] / f"{image_path.stem}.txt"
        raw_path.write_text(raw_output, encoding="utf-8")

        try:
            parsed = self._build_parsed_payload(image_path.stem, raw_output, three_call_stages)
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


    def _generate_single_call(
        self, 
        image_path, 
        allowed_audioset_nodes, 
        allowed_visual_terms,
        allowed_scene_labels, 
        max_new_tokens,
    ) -> str:
        prompt = build_workflow_a_audioset_core_prompt(
            allowed_audioset_nodes=allowed_audioset_nodes,
            allowed_visual_terms=allowed_visual_terms,
            allowed_scene_labels=allowed_scene_labels,
        )
        return self.vlm.generate(
            image_path=image_path, 
            prompt=prompt,
            max_new_tokens=max_new_tokens
        )


    def _generate_three_call(
        self, 
        image_path, 
        allowed_audioset_nodes, 
        allowed_visual_terms,
        allowed_scene_labels, 
        max_new_tokens,
    ) -> _ThreeCallStages:
        """Stage 1 (scene + visual_terms) -> stage 2 (nodes, derived from stage
        1's visual_terms) -> stage 3 (caption, text-only). Each stage is an
        independent VLM call, so -- unlike the single-call flow -- there is no
        autoregressive conditioning keeping a later stage consistent with an
        earlier one; see CLAUDE.md for why that matters here."""
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

        raw_output = (
            "=== STAGE 1: scene + visual_terms ===\n" + stage1_raw
            + "\n\n=== STAGE 2: nodes ===\n" + stage2_raw
            + "\n\n=== STAGE 3: caption ===\n" + stage3_raw
        )
        return _ThreeCallStages(
            raw_output=raw_output, stage3_raw=stage3_raw,
            visual_terms=visual_terms, scene=scene, nodes=nodes,
        )


    # -- parsing / validation ---------------------------------------------------

    @staticmethod
    def _build_parsed_payload(
        image_id: str, 
        raw_output: str, 
        three_call_stages: _ThreeCallStages | None,
    ) -> dict[str, Any]:
        if three_call_stages is None:
            return extract_json_block(raw_output)

        stage3_parsed = extract_json_block(three_call_stages.stage3_raw)
        return {
            "core": {
                "image_id": image_id,
                "visual_terms": three_call_stages.visual_terms,
                "scene": three_call_stages.scene,
                "nodes": three_call_stages.nodes,
                "caption": stage3_parsed.get("caption"),
            }
        }


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

        # single-call and three-call converge here: both hand this a
        # core-shaped dict, so the audioset-core validator doesn't need to
        # know which call_mode produced it.
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
    ) -> dict[str, Any]:
        if use_legacy_core:
            prompt_version = (
                "workflow_a_legacy_visual_core_audioset_v1"
                if include_audioset_nodes
                else "workflow_a_legacy_visual_core_v1"
            )
        else:
            prompt_version = (
                "workflow_a_audioset_core_v1"
                if call_mode == "single"
                else "workflow_a_audioset_core_three_call_v1"
            )

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
