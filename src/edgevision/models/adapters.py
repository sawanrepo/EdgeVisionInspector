"""Unified detector adapters for training, PyTorch reference evaluation, and pre-NMS ONNX export."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from edgevision.data.transforms import ResizeParams
from edgevision.inference.decode_nms import (
    Detection,
    decode_yolox_raw_numpy,
    flatten_faster_rcnn_pre_nms,
    postprocess_pre_nms,
)
from edgevision.models.yolox import YOLOXModel


class BaseDetectorAdapter(ABC, nn.Module):
    """Common interface across YOLOX-Tiny, SSDLite320, and Faster R-CNN."""

    architecture: str
    input_size: tuple[int, int]
    num_classes: int

    def __init__(self, architecture: str, input_size: tuple[int, int], num_classes: int) -> None:
        super().__init__()
        self.architecture = architecture
        self.input_size = input_size
        self.num_classes = num_classes

    @abstractmethod
    def compute_training_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute training loss dictionary containing at least 'total_loss'."""

    @abstractmethod
    def forward_pre_nms(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Run PyTorch model on a batch-1 tensor and return (boxes_xyxy [A,4], scores [A,C]) pre-NMS."""

    @abstractmethod
    def build_onnx_export_module(self) -> nn.Module:
        """Return a batch-1 nn.Module outputting tensors consumed by shared CPU post-processing."""

    def predict_batch(
        self,
        images: torch.Tensor,
        resize_params_list: list[ResizeParams],
        class_names: list[str],
        conf_threshold: float = 0.05,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> list[list[Detection]]:
        """Run PyTorch inference and shared CPU-side decode/NMS for each image in the batch."""
        self.eval()
        results: list[list[Detection]] = []
        with torch.no_grad():
            for idx in range(images.shape[0]):
                single = images[idx : idx + 1]
                boxes_xyxy, scores = self.forward_pre_nms(single)
                dets = postprocess_pre_nms(
                    boxes_xyxy=boxes_xyxy,
                    class_scores=scores,
                    resize_params=resize_params_list[idx],
                    class_names=class_names,
                    conf_threshold=conf_threshold,
                    nms_iou_threshold=nms_iou_threshold,
                    max_detections=max_detections,
                    class_aware=True,
                )
                results.append(dets)
        return results


class _YOLOXOnnxWrapper(nn.Module):
    def __init__(self, core: YOLOXModel) -> None:
        super().__init__()
        self.core = core

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.core.forward_raw(images)


class YOLOXAdapter(BaseDetectorAdapter):
    """Adapter for YOLOX-Tiny with raw head export and shared external grid/stride decode."""

    def __init__(
        self,
        core: YOLOXModel,
        input_size: tuple[int, int],
        num_classes: int,
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        super().__init__("yolox_tiny", input_size=input_size, num_classes=num_classes)
        self.core = core
        self.strides = strides

    def compute_training_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.core.compute_loss(images, targets, class_weights=class_weights)

    def forward_raw(self, images: torch.Tensor) -> torch.Tensor:
        return self.core.forward_raw(images)

    def forward_pre_nms(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if images.shape[0] != 1:
            raise ValueError("forward_pre_nms expects batch size 1")
        raw = self.core.forward_raw(images).detach().cpu().numpy()
        return decode_yolox_raw_numpy(raw, input_size=self.input_size, strides=self.strides)

    def build_onnx_export_module(self) -> nn.Module:
        return _YOLOXOnnxWrapper(self.core).eval()


def _configure_identity_tv_transform(tv_model: nn.Module, input_size: tuple[int, int]) -> None:
    """Disable duplicate normalization/resizing inside torchvision GeneralizedRCNNTransform."""
    h, w = int(input_size[0]), int(input_size[1])
    if hasattr(tv_model, "transform"):
        tv_model.transform.image_mean = [0.0, 0.0, 0.0]
        tv_model.transform.image_std = [1.0, 1.0, 1.0]
        tv_model.transform.min_size = (min(h, w),)
        tv_model.transform.max_size = max(h, w)
        if hasattr(tv_model.transform, "fixed_size"):
            tv_model.transform.fixed_size = (h, w)


class _SSDLitePreNMSExportWrapper(nn.Module):
    """Exportable SSDLite wrapper emitting pre-NMS decoded boxes [1, A, 4] and foreground probs [1, A, C]."""

    def __init__(self, ssd_model: nn.Module, input_size: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = ssd_model.backbone
        self.head = ssd_model.head
        h, w = int(input_size[0]), int(input_size[1])

        # Precompute fixed 320x320 anchors once so ONNX graph has zero ImageList/GridAnchor tracing complexity.
        ssd_model.eval()
        with torch.no_grad():
            device = next(ssd_model.parameters()).device
            dummy = torch.zeros((1, 3, h, w), dtype=torch.float32, device=device)
            feats = list(self.backbone(dummy).values())
            from torchvision.models.detection.image_list import ImageList

            img_list = ImageList(dummy, [(h, w)])
            anchors = ssd_model.anchor_generator(img_list, feats)[0]  # [A, 4] xyxy

        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        self.register_buffer("anchor_ctr_x", ctr_x.unsqueeze(0))
        self.register_buffer("anchor_ctr_y", ctr_y.unsqueeze(0))
        self.register_buffer("anchor_w", widths.unsqueeze(0))
        self.register_buffer("anchor_h", heights.unsqueeze(0))
        self.weights = tuple(float(x) for x in ssd_model.box_coder.weights)
        self.clip_w = float(w)
        self.clip_h = float(h)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = list(self.backbone(images).values())
        head_out = self.head(features)
        rel_codes = head_out["bbox_regression"]  # [B, A, 4]
        cls_logits = head_out["cls_logits"]  # [B, A, 1 + C]

        wx, wy, ww, wh = self.weights
        dx = rel_codes[..., 0] / wx
        dy = rel_codes[..., 1] / wy
        dw = torch.clamp(rel_codes[..., 2] / ww, max=4.135)
        dh = torch.clamp(rel_codes[..., 3] / wh, max=4.135)

        pred_ctr_x = dx * self.anchor_w + self.anchor_ctr_x
        pred_ctr_y = dy * self.anchor_h + self.anchor_ctr_y
        pred_w = torch.exp(dw) * self.anchor_w
        pred_h = torch.exp(dh) * self.anchor_h

        x1 = torch.clamp(pred_ctr_x - 0.5 * pred_w, min=0.0, max=self.clip_w)
        y1 = torch.clamp(pred_ctr_y - 0.5 * pred_h, min=0.0, max=self.clip_h)
        x2 = torch.clamp(pred_ctr_x + 0.5 * pred_w, min=0.0, max=self.clip_w)
        y2 = torch.clamp(pred_ctr_y + 0.5 * pred_h, min=0.0, max=self.clip_h)
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        probs = F.softmax(cls_logits, dim=-1)[..., 1:]  # exclude background class 0
        return boxes, probs


class SSDLiteAdapter(BaseDetectorAdapter):
    """Adapter for Torchvision SSDLite320 MobileNetV3-Large."""

    def __init__(self, core: nn.Module, input_size: tuple[int, int], num_classes: int) -> None:
        if tuple(input_size) != (320, 320):
            raise ValueError(
                f"SSDLite320 preserves its native 320x320 anchor configuration; got {input_size}"
            )
        super().__init__("ssdlite320_mobilenet_v3_large", input_size=input_size, num_classes=num_classes)
        self.core = core
        _configure_identity_tv_transform(self.core, input_size)
        self._pre_nms_view: _SSDLitePreNMSExportWrapper | None = None

    def compute_training_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img_list = [images[i] for i in range(images.shape[0])]
        tv_targets = [
            {
                "boxes": t["boxes"].to(images.device),
                # Torchvision reserves label 0 for background; shift 0..5 -> 1..6.
                "labels": t["labels"].to(images.device) + 1,
            }
            for t in targets
        ]
        loss_dict = self.core(img_list, tv_targets)
        total = sum(loss_dict.values())
        return {
            "total_loss": total,
            **{k: v.detach() for k, v in loss_dict.items()},
        }

    def forward_pre_nms(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if images.shape[0] != 1:
            raise ValueError("forward_pre_nms expects batch size 1")
        wrapper = _SSDLitePreNMSExportWrapper(self.core, self.input_size).to(images.device).eval()
        boxes, probs = wrapper(images)
        return boxes[0].detach().cpu().numpy(), probs[0].detach().cpu().numpy()

    def build_onnx_export_module(self) -> nn.Module:
        return _SSDLitePreNMSExportWrapper(self.core, self.input_size).eval()


class _FasterRCNNPreNMSExportWrapper(nn.Module):
    """Exportable Faster R-CNN screening wrapper outputting pre-final-NMS ROI boxes and class probabilities.

    RPN proposal generation and internal proposal NMS remain inside the RPN stage as part of
    two-stage ROI generation; final detection score filtering and class-aware NMS stay outside ONNX.
    """

    def __init__(self, rcnn_model: nn.Module, input_size: tuple[int, int]) -> None:
        super().__init__()
        self.backbone = rcnn_model.backbone
        self.rpn = rcnn_model.rpn
        self.box_roi_pool = rcnn_model.roi_heads.box_roi_pool
        self.box_head = rcnn_model.roi_heads.box_head
        self.box_predictor = rcnn_model.roi_heads.box_predictor
        self.box_coder = rcnn_model.roi_heads.box_coder
        self.input_size = (int(input_size[0]), int(input_size[1]))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.models.detection.image_list import ImageList

        h, w = self.input_size
        img_list = ImageList(images, [(h, w)])
        features = self.backbone(images)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, _ = self.rpn(img_list, features, None)
        box_features = self.box_roi_pool(features, proposals, [(h, w)])
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        probs = F.softmax(class_logits, dim=-1)[:, 1:]  # [N_prop, C]
        pred_boxes_all = self.box_coder.decode(box_regression, proposals)  # [N_prop, 1+C, 4]
        pred_boxes_fg = pred_boxes_all[:, 1:, :]  # [N_prop, C, 4]

        # Flatten per-(proposal, class) to [N_prop * C, 4] and aligned sparse/full [N_prop * C, C]
        # or keep class-specific boxes by taking argmax box per proposal? No — keep exact per-class boxes!
        # Wait: returning [1, N_prop, C, 4] and [1, N_prop, C] preserves exact class-specific regression!
        pred_boxes_fg[..., 0::2] = torch.clamp(pred_boxes_fg[..., 0::2], min=0.0, max=float(w))
        pred_boxes_fg[..., 1::2] = torch.clamp(pred_boxes_fg[..., 1::2], min=0.0, max=float(h))
        return pred_boxes_fg.unsqueeze(0), probs.unsqueeze(0)


class FasterRCNNAdapter(BaseDetectorAdapter):
    """Adapter for Torchvision Faster R-CNN ResNet50-FPN (accuracy-oriented reference)."""

    def __init__(self, core: nn.Module, input_size: tuple[int, int], num_classes: int) -> None:
        super().__init__("faster_rcnn_resnet50_fpn", input_size=input_size, num_classes=num_classes)
        self.core = core
        _configure_identity_tv_transform(self.core, input_size)

    def compute_training_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        img_list = [images[i] for i in range(images.shape[0])]
        tv_targets = [
            {
                "boxes": t["boxes"].to(images.device),
                "labels": t["labels"].to(images.device) + 1,
            }
            for t in targets
        ]
        loss_dict = self.core(img_list, tv_targets)
        total = sum(loss_dict.values())
        return {
            "total_loss": total,
            **{k: v.detach() for k, v in loss_dict.items()},
        }

    def forward_pre_nms(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        if images.shape[0] != 1:
            raise ValueError("forward_pre_nms expects batch size 1")
        wrapper = _FasterRCNNPreNMSExportWrapper(self.core, self.input_size).to(images.device).eval()
        boxes_fg, probs = wrapper(images)
        return flatten_faster_rcnn_pre_nms(boxes_fg.detach().cpu().numpy(), probs.detach().cpu().numpy())

    def build_onnx_export_module(self) -> nn.Module:
        return _FasterRCNNPreNMSExportWrapper(self.core, self.input_size).eval()
