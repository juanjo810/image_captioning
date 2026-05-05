from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


SUPPORTED_ARCHS = {
    "resnet50": "resnet50_places365.pth.tar",
    "densenet161": "densenet161_places365.pth.tar",
}


class Places365Adapter:
    def __init__(
        self,
        architecture: str,
        categories_path: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        if architecture not in SUPPORTED_ARCHS:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(SUPPORTED_ARCHS)}"
            )

        self.architecture = architecture
        self.categories_path = Path(categories_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        self.categories = self._load_categories(self.categories_path)
        self.model = self._load_model().to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @staticmethod
    def _load_categories(path: Path) -> list[str]:
        labels = []

        for line in path.read_text().splitlines():
            raw = line.strip().split(" ")[0]
            label = raw.split("/")[-1].replace("_", " ")
            labels.append(label)

        return labels

    def _build_architecture(self):
        if self.architecture == "resnet50":
            return models.resnet50(num_classes=365)

        if self.architecture == "densenet161":
            return models.densenet161(num_classes=365)

        raise ValueError(self.architecture)

    def _load_model(self):
        model = self._build_architecture()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        clean_state_dict = {}

        for key, value in state_dict.items():
            key = key.replace("module.", "")

            if self.architecture == "densenet161":
                key = key.replace(".norm.1.", ".norm1.")
                key = key.replace(".norm.2.", ".norm2.")
                key = key.replace(".conv.1.", ".conv1.")
                key = key.replace(".conv.2.", ".conv2.")

            clean_state_dict[key] = value

        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)

        if missing:
            print(f"[Places365:{self.architecture}] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Places365:{self.architecture}] Unexpected keys: {len(unexpected)}")

        return model

    @torch.no_grad()
    def predict(self, image_path: str | Path, topk: int = 5) -> dict:
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        values, indices = torch.topk(probs, k=topk)

        topk_predictions = [
            {
                "label": self.categories[int(idx)],
                "confidence": float(conf),
            }
            for conf, idx in zip(values, indices)
        ]

        return {
            "architecture": self.architecture,
            "label": topk_predictions[0]["label"],
            "confidence": topk_predictions[0]["confidence"],
            "topk": topk_predictions,
        }