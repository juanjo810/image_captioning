from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

CAPTION_SPATIAL_RELATIONS = {
    "left_of",
    "right_of",
    "above",
    "below",
}


def select_caption_relevant_spatial_relations(
    core_json: dict[str, Any],
    max_relations: int = 2,
) -> list[dict[str, Any]]:
    relations = core_json.get("spatial_relations", [])
    entities = {
        entity.get("id"): entity
        for entity in core_json.get("entities", [])
    }

    selected = []

    for rel in relations:
        if rel.get("relation") not in CAPTION_SPATIAL_RELATIONS:
            continue

        subject = entities.get(rel.get("subject_id"))
        obj = entities.get(rel.get("object_id"))

        if subject is None or obj is None:
            continue

        # Avoid unnatural captions like:
        # "the building is to the right of the street lamp"
        if (
            subject.get("category") in {"object", "structure"}
            and obj.get("category") in {"object", "structure"}
        ):
            continue

        selected.append(rel)

    selected = sorted(
        selected,
        key=lambda r: r.get("confidence", 0.0),
        reverse=True,
    )

    return selected[:max_relations]


def build_caption_payload(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "scene": core_json.get("scene", {}),
        "entities": core_json.get("entities", []),
        "observed_interactions": core_json.get("observed_interactions", []),
        "spatial_relations": select_caption_relevant_spatial_relations(core_json),
        "environment": core_json.get("environment", {}),
    }

def build_llm_caption_prompt(
    core_json: dict[str, Any],
    extended_json: dict[str, Any] | None = None,
) -> str:
    payload = build_caption_payload(core_json, extended_json)

    return (
        "You are given a structured JSON representation extracted from an image.\n"
        "Write one concise natural image caption using only the information present in the JSON.\n\n"
        "Rules:\n"
        "- Do not add objects, people, animals, locations, actions, or attributes not present in the JSON.\n"
        "- Do not infer geographic locations.\n"
        "- Do not invent actions not present in observed_interactions.\n"
        "- Prefer observed_interactions when available.\n"
        "- Use spatial_relations only when they make the caption more natural.\n"
        "- Do not mention technical relations such as overlapping.\n"
        "- Do not mention bounding boxes, confidences, ids, metadata, or JSON structure.\n"
        "- Mention the scene if useful.\n"
        "- Keep the caption natural and concise.\n"
        "- If confidence is low or information is sparse, remain generic.\n"
        "- Return only the caption, with no explanation.\n\n"
        "JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

def clean_caption(caption: str) -> str:
    caption = caption.strip()
    caption = caption.replace("<end_of_turn>", "").strip()
    caption = caption.strip('"').strip("'")

    if "\n" in caption:
        caption = caption.splitlines()[0].strip()

    if caption and not caption.endswith("."):
        caption += "."

    return caption or "A visual scene."


class GemmaCaptioner:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 80,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        print(f"[INFO] Loading processor: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        print(f"[INFO] Loading model: {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text,
            return_tensors="pt",
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        decoded = self.processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
        )

        return clean_caption(decoded)


def caption_json_file(
    input_path: Path,
    output_path: Path,
    captioner: GemmaCaptioner,
) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    core_json = data.get("core", {})
    extended_json = data.get("extended")

    prompt = build_llm_caption_prompt(
        core_json=core_json,
        extended_json=extended_json,
    )

    caption = captioner.generate(prompt)

    data.setdefault("core", {})
    data["core"]["caption"] = caption

    data.setdefault("metadata", {})
    data["metadata"]["caption_mode"] = "llm"
    data["metadata"]["llm_backend"] = "gemma_transformers"
    data["metadata"]["llm_model"] = captioner.model_id
    data["metadata"]["caption_source"] = "structured_json_only"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] {input_path.name} -> {output_path}")


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file or directory containing JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where captioned JSON files will be saved.",
    )

    parser.add_argument(
        "--model-id",
        default="google/gemma-4-E4B-it",
        help="Hugging Face Gemma model id.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    input_files = collect_input_files(input_path)

    if not input_files:
        raise RuntimeError(f"No JSON files found in: {input_path}")

    captioner = GemmaCaptioner(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    for json_file in input_files:
        output_path = output_dir / json_file.name
        caption_json_file(
            input_path=json_file,
            output_path=output_path,
            captioner=captioner,
        )


if __name__ == "__main__":
    main()
    