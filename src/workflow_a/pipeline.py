from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.workflow_a.audioset_nodes import default_allowed_audioset_nodes
from src.workflow_a.parser import extract_json_block
from src.workflow_a.prompt_builder import build_workflow_a_prompt
from src.workflow_a.validator import (
    validate_acoustic_semantics_output,
    validate_workflow_a_output,
)


class WorkflowAPipeline:
    def __init__(self, vlm_adapter) -> None:
        self.vlm = vlm_adapter

    def run(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        include_audioset_nodes: bool = False,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        raw_dir = output_dir / "raw"
        json_dir = output_dir / "json"
        captions_dir = output_dir / "captions"
        manifests_dir = output_dir / "manifests"
        failed_dir = output_dir / "failed"

        for directory in [
            raw_dir,
            json_dir,
            captions_dir,
            manifests_dir,
            failed_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        allowed_audioset_nodes = (
            default_allowed_audioset_nodes()
            if include_audioset_nodes
            else ()
        )
        prompt = build_workflow_a_prompt(
            include_audioset_nodes=include_audioset_nodes,
            allowed_audioset_nodes=allowed_audioset_nodes,
        )

        raw_output = self.vlm.generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        raw_path = raw_dir / f"{image_path.stem}.txt"
        raw_path.write_text(raw_output, encoding="utf-8")

        try:
            parsed = extract_json_block(raw_output)

            core = validate_workflow_a_output(
                parsed,
                image_id=image_path.stem,
            )
            acoustic_semantics = (
                validate_acoustic_semantics_output(
                    parsed,
                    allowed_audioset_nodes=allowed_audioset_nodes,
                )
                if include_audioset_nodes
                else None
            )

        except Exception as exc:
            failed_payload = {
                "image_id": image_path.stem,
                "error": str(exc),
                "raw_path": str(raw_path),
            }

            failed_path = failed_dir / f"{image_path.stem}.json"
            failed_path.write_text(
                json.dumps(failed_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            raise

        metadata = {
            "workflow": "A",
            "model_id": getattr(self.vlm, "model_id", "unknown"),
            "adapter": self.vlm.__class__.__name__,
            "prompt_version": (
                "workflow_a_core_audioset_v1"
                if include_audioset_nodes
                else "workflow_a_core_v1"
            ),
            "image_id": image_path.stem,
            "include_audioset_nodes": include_audioset_nodes,
            "generation_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }
        if include_audioset_nodes:
            metadata["audioset_allowed_nodes"] = list(allowed_audioset_nodes)

        payload = {
            "core": core.model_dump(),
            "metadata": metadata,
        }
        if acoustic_semantics is not None:
            payload["acoustic_semantics"] = acoustic_semantics.model_dump()

        json_path = json_dir / f"{image_path.stem}.json"
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        caption_path = captions_dir / f"{image_path.stem}.txt"
        caption_path.write_text(core.caption, encoding="utf-8")

        manifest_record = {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "json_path": str(json_path),
            "caption_path": str(caption_path),
        }

        manifest_path = manifests_dir / "manifest.jsonl"
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

        return payload
