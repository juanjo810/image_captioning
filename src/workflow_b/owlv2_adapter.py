from __future__ import annotations

import re
from pathlib import Path

import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from src.geometry import Detection


class OWLv2Adapter:
    """OWLv2 open-vocabulary object detector adapter.

    It converts OWLv2 outputs into the internal Detection dataclass used by
    Workflow B.
    """

    def __init__(
        self,
        model_id: str = "google/owlv2-base-patch16-ensemble",
        device: str = "cuda",
    ) -> None:
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        image_path: str | Path,
        prompt: str,
        box_threshold: float = 0.10,
        text_threshold: float = 0.0,
    ) -> list[Detection]:
        """Run OWLv2 detection.

        Parameters are kept compatible with GroundingDINOAdapter.
        text_threshold is ignored because OWLv2 does not expose the same
        text-token threshold concept.
        """

        image = Image.open(image_path).convert("RGB")
        candidate_labels = self._prompt_to_labels(prompt)

        if not candidate_labels:
            return []

        inputs = self.processor(
            text=[candidate_labels],
            images=image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [(image.height, image.width)],
            device=self.device,
        )

        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=box_threshold,
        )[0]

        detections: list[Detection] = []

        for score, label_idx, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"],
        ):
            label = candidate_labels[int(label_idx)]
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]

            detections.append(
                Detection(
                    label=label.strip().lower(),
                    bbox=[x1, y1, x2, y2],
                    confidence=float(score),
                    source="owlv2",
                )
            )

        return detections

    @staticmethod
    def _prompt_to_labels(prompt: str) -> list[str]:
        """Convert a GroundingDINO-style prompt into OWLv2 candidate labels.

        GroundingDINO prompts usually look like:
            "person . horse . cart . field . sky ."

        OWLv2 expects:
            ["person", "horse", "cart", "field", "sky"]
        """

        raw_items = re.split(r"[.;,\n]+", prompt)

        labels: list[str] = []
        seen: set[str] = set()

        for item in raw_items:
            label = item.strip().lower().replace("_", " ")

            if not label:
                continue

            if label.startswith("a "):
                label = label[2:]
            elif label.startswith("an "):
                label = label[3:]
            elif label.startswith("the "):
                label = label[4:]

            if label and label not in seen:
                labels.append(label)
                seen.add(label)

        return labels
        