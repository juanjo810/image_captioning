from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from src.workflow_a.base import BaseVLM


class Gemma4Adapter(BaseVLM):
    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        device_map: str = "auto",
    ) -> None:
        self.model_id = model_id

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device_map,
        )

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        generated_tokens = outputs[:, inputs["input_ids"].shape[-1]:]

        return self.processor.decode(
            generated_tokens[0],
            skip_special_tokens=True,
        ).strip()
