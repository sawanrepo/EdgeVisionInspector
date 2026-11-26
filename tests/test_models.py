"""Smoke tests for detector architectures, loss computation, and Path B resolution constraints."""

from __future__ import annotations

import pytest
import torch

from edgevision.config import load_yaml, resolve_project_root
from edgevision.models.build import build_detector

ROOT = resolve_project_root()


@pytest.mark.parametrize(
    ("cfg_rel", "input_size"),
    [
        ("configs/yolox_tiny.yaml", (224, 224)),
        ("configs/yolox_tiny.yaml", (320, 320)),
        ("configs/ssdlite.yaml", (320, 320)),
        ("configs/faster_rcnn.yaml", (224, 224)),
        ("configs/faster_rcnn.yaml", (320, 320)),
    ],
)
def test_detector_forward_loss_and_pre_nms_schema(cfg_rel: str, input_size: tuple[int, int]) -> None:
    cfg = load_yaml(ROOT / cfg_rel)
    adapter = build_detector(cfg, override_pretrained=False, override_input_size=input_size)
    h, w = input_size
    images = torch.rand((2, 3, h, w), dtype=torch.float32)
    targets = [
        {"boxes": torch.tensor([[20.0, 20.0, 80.0, 80.0]], dtype=torch.float32), "labels": torch.tensor([0])},
        {"boxes": torch.tensor([[30.0, 40.0, 110.0, 120.0]], dtype=torch.float32), "labels": torch.tensor([5])},
    ]

    adapter.train()
    losses = adapter.compute_training_loss(images, targets)
    assert "total_loss" in losses
    assert torch.isfinite(losses["total_loss"])

    adapter.eval()
    with torch.no_grad():
        boxes, scores = adapter.forward_pre_nms(images[:1])
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    assert scores.ndim == 2 and scores.shape[1] == 6
    assert boxes.shape[0] == scores.shape[0]


def test_ssdlite_rejects_non_native_resolution() -> None:
    cfg = load_yaml(ROOT / "configs/ssdlite.yaml")
    with pytest.raises(ValueError, match="native"):
        build_detector(cfg, override_pretrained=False, override_input_size=(224, 224))
