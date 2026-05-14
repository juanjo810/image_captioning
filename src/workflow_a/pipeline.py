from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.workflow_a.parser import extract_json_block
from src.workflow_a.prompt_builder import build_workflow_a_prompt
from src.workflow_a.validator import validate_workflow_a_output


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
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)

        raw_dir = output_dir / "raw"
        json_dir = output_dir / "json"
        captions_dir = output_dir / "captions"
        manifests_dir = output_dir / "manifests"

        for directory in [raw_dir, json_dir, captions_dir, manifests_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        prompt = build_workflow_a_prompt()

        raw_output = self.vlm.generate(
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        parsed = extract_json_block(raw_output)

        core = validate_workflow_a_output(
            parsed,
            image_id=image_path.stem,
        )

        metadata = {
            "workflow": "A",
            "model_id": getattr(self.vlm, "model_id", "unknown"),
            "adapter": self.vlm.__class__.__name__,
            "prompt_version": "workflow_a_core_v1",
            "image_id": image_path.stem,
            "generation_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        }

        payload = {
            "core": core.model_dump(),
            "metadata": metadata,
        }

        raw_path = raw_dir / f"{image_path.stem}.txt"
        raw_path.write_text(raw_output, encoding="utf-8")

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
