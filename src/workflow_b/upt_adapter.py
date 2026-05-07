from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch

from src.workflow_b.hoi_adapter import RawHOITriplet

from PIL import Image


def scale_bbox(
    bbox: list[float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> list[float]:
    source_w, source_h = source_size
    target_w, target_h = target_size

    sx = target_w / source_w
    sy = target_h / source_h

    return [
        bbox[0] * sx,
        bbox[1] * sy,
        bbox[2] * sx,
        bbox[3] * sy,
    ]

class UPTAdapter:
    def __init__(
        self,
        upt_root: str | Path,
        checkpoint_path: str | Path,
        data_root: str | Path,
        device: str = "cuda",
        action_score_thresh: float = 0.10,
    ):
        self.upt_root = Path(upt_root)
        self.checkpoint_path = Path(checkpoint_path)
        self.data_root = Path(data_root)
        self.device = device
        self.action_score_thresh = action_score_thresh

        for path in [
            self.upt_root,
            self.upt_root / "detr",
            Path("/home/jovyan/projects/pocket"),
        ]:
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)
        
        from utils import DataFactory
        from upt import build_detector

        self.dataset = DataFactory(
            name="hicodet",
            partition="test2015",
            data_root=str(self.data_root),
        )

        conversion = self.dataset.dataset.object_to_verb

        args = self._build_args()
        args.num_classes = 117

        self.actions = self.dataset.dataset.verbs
        self.model = build_detector(args, conversion).to(device)
        self.model.eval()

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _build_args(self) -> Namespace:
        return Namespace(
            backbone="resnet50",
            dilation=False,
            position_embedding="sine",
            repr_dim=512,
            hidden_dim=256,
            enc_layers=6,
            dec_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            nheads=8,
            num_queries=100,
            pre_norm=False,
            aux_loss=True,
            set_cost_class=1,
            set_cost_bbox=5,
            set_cost_giou=2,
            bbox_loss_coef=5,
            giou_loss_coef=2,
            eos_coef=0.1,
            alpha=0.5,
            gamma=0.2,
            dataset="hicodet",
            partition="test2015",
            data_root=str(self.data_root),
            human_idx=0,
            device=self.device,
            pretrained="",
            box_score_thresh=0.2,
            fg_iou_thresh=0.5,
            min_instances=3,
            max_instances=15,
            resume=str(self.checkpoint_path),
            index=0,
            action=None,
            action_score_thresh=self.action_score_thresh,
            image_path=None,
        )

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> list[RawHOITriplet]:
        target_image = Image.open(image_path).convert("RGB")
        target_size = target_image.size

        image = self.dataset.dataset.load_image(str(image_path))
        image_tensor, _ = self.dataset.transforms(image, None)
        image_tensor = image_tensor.to(self.device)
        
        source_size = (
            image_tensor.shape[-1],
            image_tensor.shape[-2],
        )

        output = self.model([image_tensor])[0]

        boxes = output["boxes"].detach().cpu()
        pairing = output["pairing"].detach().cpu()
        scores = output["scores"].detach().cpu()
        objects = output["objects"].detach().cpu()

        triplets: list[RawHOITriplet] = []

        unique_actions = torch.unique(output["labels"].detach().cpu())

        for verb_idx in unique_actions:
            verb_idx_int = int(verb_idx)
            verb = self.actions[verb_idx_int]

            if verb == "no_interaction":
                continue

            sample_idx = torch.nonzero(output["labels"].detach().cpu() == verb_idx).squeeze(1)

            for idx in sample_idx:
                score = float(scores[idx])

                if score < self.action_score_thresh:
                    continue

                human_box_idx, object_box_idx = pairing[:, idx]

                triplets.append(
                    RawHOITriplet(
                        human_bbox=scale_bbox(
                            boxes[int(human_box_idx)].tolist(),
                            source_size=source_size,
                            target_size=target_size,
                        ),
                        object_bbox=scale_bbox(
                            boxes[int(object_box_idx)].tolist(),
                            source_size=source_size,
                            target_size=target_size,
                        ),
                        verb=verb,
                        confidence=score,
                    )
                )

        return triplets