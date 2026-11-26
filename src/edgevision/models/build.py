"""Factory builders for YOLOX-Tiny, SSDLite320 MobileNetV3-Large, and Faster R-CNN ResNet50-FPN."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from edgevision.config import get_logger, load_yaml, sha256_file
from edgevision.models.adapters import (
    BaseDetectorAdapter,
    FasterRCNNAdapter,
    SSDLiteAdapter,
    YOLOXAdapter,
)
from edgevision.models.yolox import YOLOXModel, load_megvii_yolox_pretrained

logger = get_logger(__name__)


def build_detector(
    model_cfg: dict[str, Any] | str | Path,
    override_pretrained: bool | None = None,
    override_input_size: tuple[int, int] | list[int] | None = None,
) -> BaseDetectorAdapter:
    """Instantiate a detector adapter from a model configuration file or dictionary."""
    cfg = load_yaml(model_cfg) if isinstance(model_cfg, (str, Path)) else model_cfg
    mcfg = cfg["model"]
    arch = str(mcfg["architecture"])
    num_classes = int(mcfg.get("num_classes", 6))
    raw_size = override_input_size or mcfg.get("input_size", [320, 320])
    input_size = (int(raw_size[0]), int(raw_size[1]))
    pretrained = bool(mcfg.get("pretrained", True) if override_pretrained is None else override_pretrained)

    if arch == "yolox_tiny":
        strides = tuple(int(s) for s in mcfg.get("strides", [8, 16, 32]))
        core = YOLOXModel(
            num_classes=num_classes,
            depth=float(mcfg.get("depth_multiple", 0.33)),
            width=float(mcfg.get("width_multiple", 0.375)),
            strides=strides,
        )
        ckpt_path = mcfg.get("pretrained_checkpoint")
        if pretrained and ckpt_path:
            ckpt_file = Path(ckpt_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(f"YOLOX-Tiny pretrained checkpoint not found: {ckpt_file}")
            expected_sha = mcfg.get("provenance", {}).get("checkpoint_sha256")
            if expected_sha and expected_sha != "pending_verification":
                actual_sha = sha256_file(ckpt_file)
                if actual_sha.lower() != str(expected_sha).lower():
                    raise ValueError(
                        f"YOLOX-Tiny checkpoint SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                    )
            load_info = load_megvii_yolox_pretrained(core, str(ckpt_file))
            logger.info("Loaded YOLOX-Tiny checkpoint (%s): %s", ckpt_file, load_info)
        elif pretrained:
            logger.info(
                "YOLOX-Tiny initialized without local checkpoint file; set model.pretrained_checkpoint "
                "after verifying checkpoint provenance/SHA-256."
            )
        return YOLOXAdapter(core=core, input_size=input_size, num_classes=num_classes, strides=strides)

    if arch == "ssdlite320_mobilenet_v3_large":
        if input_size != (320, 320):
            raise ValueError(
                f"SSDLite320 requires native (320, 320) input size under Path B, got {input_size}"
            )
        weights = (
            torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            if pretrained
            else None
        )
        tv_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=int(mcfg.get("trainable_backbone_layers", 6)),
        )
        in_channels = torchvision.models.detection._utils.retrieve_out_channels(
            tv_model.backbone, input_size
        )
        num_anchors = tv_model.anchor_generator.num_anchors_per_location()
        tv_model.head.classification_head = SSDLiteClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes + 1,
            norm_layer=partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03),
        )
        return SSDLiteAdapter(core=tv_model, input_size=input_size, num_classes=num_classes)

    if arch == "faster_rcnn_resnet50_fpn":
        weights = (
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            if pretrained
            else None
        )
        tv_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            trainable_backbone_layers=int(mcfg.get("trainable_backbone_layers", 3)),
        )
        in_features = tv_model.roi_heads.box_predictor.cls_score.in_features
        tv_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
        return FasterRCNNAdapter(core=tv_model, input_size=input_size, num_classes=num_classes)

    raise ValueError(f"Unsupported detector architecture: '{arch}'")
