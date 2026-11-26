"""PyTorch Dataset and DataLoader utilities built on split manifests and preprocess.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from edgevision.data.annotations import ImageRecord, load_records_jsonl
from edgevision.data.transforms import (
    Preprocessor,
    TrainingAugmentor,
    decode_grayscale_to_rgb_array,
)


class DefectDetectionDataset(Dataset):
    """Detection dataset reading JSONL records and applying the canonical preprocessing contract."""

    def __init__(
        self,
        manifest_path: str | Path,
        architecture: str,
        input_size: tuple[int, int] | list[int] | None = None,
        preprocess_cfg: dict[str, Any] | str | Path = "configs/preprocess.yaml",
        augmentor: TrainingAugmentor | None = None,
    ) -> None:
        self.records: list[ImageRecord] = load_records_jsonl(manifest_path)
        self.preprocessor = Preprocessor(
            architecture=architecture,
            input_size=input_size,
            preprocess_cfg=preprocess_cfg,
        )
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        rgb = decode_grayscale_to_rgb_array(rec.image_path)
        boxes = np.array(
            [[b.xmin, b.ymin, b.xmax, b.ymax] for b in rec.boxes],
            dtype=np.float32,
        ).reshape(-1, 4)
        labels = np.array([b.class_id for b in rec.boxes], dtype=np.int64)

        if self.augmentor is not None:
            rgb, boxes, labels = self.augmentor(rgb, boxes, labels)

        tensor, params = self.preprocessor.preprocess_for_pytorch(rgb, add_batch_dim=False)
        scaled_boxes = self.preprocessor.boxes_to_input_coords(boxes, params)

        return {
            "image": tensor,
            "boxes": torch.from_numpy(scaled_boxes.astype(np.float32)),
            "labels": torch.from_numpy(labels.astype(np.int64)),
            "image_id": rec.image_id,
            "orig_size": (rec.width, rec.height),
            "resize_params": params,
        }


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length bounding boxes and labels with stacked image tensors."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    targets = [
        {
            "boxes": item["boxes"],
            "labels": item["labels"],
            "image_id": item["image_id"],
            "orig_size": item["orig_size"],
            "resize_params": item["resize_params"],
        }
        for item in batch
    ]
    return {"images": images, "targets": targets}


def build_dataloader(
    manifest_path: str | Path,
    architecture: str,
    input_size: tuple[int, int] | list[int] | None,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    preprocess_cfg: dict[str, Any] | str | Path = "configs/preprocess.yaml",
    augmentor: TrainingAugmentor | None = None,
) -> DataLoader:
    """Create a deterministic DataLoader for training or evaluation."""
    dataset = DefectDetectionDataset(
        manifest_path=manifest_path,
        architecture=architecture,
        input_size=input_size,
        preprocess_cfg=preprocess_cfg,
        augmentor=augmentor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
