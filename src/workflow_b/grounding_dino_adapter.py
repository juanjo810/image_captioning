"""Grounding DINO local adapter.

This adapter wraps the original GroundingDINO repository installed in editable
mode outside this project. It converts Grounding DINO outputs into the internal
Detection dataclass used by the modular Workflow B pipeline.
"""

from __future__ import annotations

from pathlib import Path

from groundingdino.util.inference import load_model, load_image, predict

from src.geometry import Detection


class GroundingDINOAdapter:
    """Object detector adapter for local Grounding DINO inference.

    Parameters
    ----------
    config_path:
        Path to the Grounding DINO config file, e.g.
        /home/jovyan/projects/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py
    checkpoint_path:
        Path to the downloaded .pth checkpoint.
    device:
        Usually "cuda" when GPU is available, otherwise "cpu".
    """

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, device: str = "cuda"):
        self.config_path = str(config_path)
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.model = load_model(self.config_path, self.checkpoint_path).to(device)

    def predict(
        self,
        image_path: str | Path,
        prompt: str,
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        """Run open-vocabulary object detection on an image.

        Grounding DINO returns normalized boxes in cxcywh format. This method
        converts them to absolute xyxy boxes and returns project-level Detection
        objects.
        """
        image_source, image = load_image(str(image_path))

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        height, width, _ = image_source.shape
        detections: list[Detection] = []

        for box, logit, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = [float(v) for v in box.tolist()]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height

            detections.append(
                Detection(
                    label=str(phrase).strip().lower(),
                    bbox=[x1, y1, x2, y2],
                    confidence=float(logit),
                    source="grounding_dino",
                )
            )

        return detections
