"""Shared CPU-side decode, coordinate rescaling, and class-aware NMS outside ONNX graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torchvision

from edgevision.data.transforms import ResizeParams


@dataclass(frozen=True)
class Detection:
    """Single predicted bounding box in native image coordinates."""

    class_id: int
    class_name: str
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "score": round(float(self.score), 6),
            "bbox_xyxy": [
                round(float(self.xmin), 3),
                round(float(self.ymin), 3),
                round(float(self.xmax), 3),
                round(float(self.ymax), 3),
            ],
        }


def decode_yolox_raw_numpy(
    raw_preds: np.ndarray,
    input_size: tuple[int, int],
    strides: tuple[int, ...] = (8, 16, 32),
) -> tuple[np.ndarray, np.ndarray]:
    """Decode raw YOLOX head tensor [1, A, 5 + C] or [A, 5 + C] into xyxy boxes and class probabilities."""
    arr = np.asarray(raw_preds, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for decode_yolox_raw_numpy, got {arr.shape}")
        arr = arr[0]

    input_h, input_w = int(input_size[0]), int(input_size[1])
    grids: list[np.ndarray] = []
    stride_vecs: list[np.ndarray] = []
    for stride in strides:
        fh = input_h // stride
        fw = input_w // stride
        xv, yv = np.meshgrid(np.arange(fw, dtype=np.float32), np.arange(fh, dtype=np.float32))
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        stride_vecs.append(np.full((fh * fw, 1), float(stride), dtype=np.float32))

    grid_all = np.concatenate(grids, axis=0)
    stride_all = np.concatenate(stride_vecs, axis=0)
    if grid_all.shape[0] != arr.shape[0]:
        raise ValueError(
            f"YOLOX anchor count mismatch: expected {grid_all.shape[0]} for input_size={input_size}, got {arr.shape[0]}"
        )

    xy = (arr[:, :2] + grid_all) * stride_all
    wh = np.exp(np.clip(arr[:, 2:4], -20.0, 10.0)) * stride_all

    x1 = xy[:, 0] - wh[:, 0] * 0.5
    y1 = xy[:, 1] - wh[:, 1] * 0.5
    x2 = xy[:, 0] + wh[:, 0] * 0.5
    y2 = xy[:, 1] + wh[:, 1] * 0.5
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    obj = 1.0 / (1.0 + np.exp(-np.clip(arr[:, 4:5], -40.0, 40.0)))
    cls = 1.0 / (1.0 + np.exp(-np.clip(arr[:, 5:], -40.0, 40.0)))
    scores = (obj * cls).astype(np.float32)
    return boxes_xyxy, scores


def flatten_faster_rcnn_pre_nms(
    boxes_per_cls: np.ndarray,
    probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten [N_prop, C, 4] and [N_prop, C] into [N_prop * C, 4] and [N_prop * C, C] for shared NMS."""
    b = np.asarray(boxes_per_cls, dtype=np.float32)
    p = np.asarray(probs, dtype=np.float32)
    if b.ndim == 4:
        b = b[0]
    if p.ndim == 3:
        p = p[0]
    n_prop, n_cls, _ = b.shape
    flat_boxes = b.reshape(n_prop * n_cls, 4)
    flat_scores = np.zeros((n_prop * n_cls, n_cls), dtype=np.float32)
    row_indices = np.arange(n_prop * n_cls)
    cls_indices = np.tile(np.arange(n_cls), n_prop)
    flat_scores[row_indices, cls_indices] = p.reshape(-1)
    return flat_boxes, flat_scores


def rescale_boxes_to_original(boxes_xyxy: np.ndarray, params: ResizeParams) -> np.ndarray:
    """Rescale decoded boxes from model input coordinates back to native image coordinates."""
    arr = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    if arr.size == 0:
        return arr
    arr[:, [0, 2]] = (arr[:, [0, 2]] - params.pad_left) / max(params.scale_x, 1e-8)
    arr[:, [1, 3]] = (arr[:, [1, 3]] - params.pad_top) / max(params.scale_y, 1e-8)
    arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0.0, float(params.orig_width))
    arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0.0, float(params.orig_height))
    return arr


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Deterministic class-specific greedy NMS using torchvision batched_nms or pure NumPy fallback."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    # Sort descending by score; break exact ties deterministically by original index.
    order = np.argsort(-scores, kind="mergesort")
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: list[int] = []
    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[idx], x1[rest])
        yy1 = np.maximum(y1[idx], y1[rest])
        xx2 = np.minimum(x2[idx], x2[rest])
        yy2 = np.minimum(y2[idx], y2[rest])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = np.maximum(1e-8, areas[idx] + areas[rest] - inter)
        iou = inter / union
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess_pre_nms(
    boxes_xyxy: np.ndarray,
    class_scores: np.ndarray,
    resize_params: ResizeParams,
    class_names: list[str],
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
    class_aware: bool = True,
) -> list[Detection]:
    """Apply score filtering, coordinate rescaling, and pinned NMS outside ONNX."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    scores_mat = np.asarray(class_scores, dtype=np.float32)

    if scores_mat.ndim == 2:
        # [A, C] matrix -> extract all (anchor, class) pairs above conf_threshold
        anchor_idx, cls_idx = np.where(scores_mat >= conf_threshold)
        if anchor_idx.size == 0:
            return []
        cand_boxes = boxes[anchor_idx]
        cand_scores = scores_mat[anchor_idx, cls_idx]
        cand_classes = cls_idx.astype(np.int64)
    elif scores_mat.ndim == 1:
        raise ValueError("Expected 2D [A, C] class_scores matrix in postprocess_pre_nms")
    else:
        raise ValueError(f"Unsupported class_scores shape: {scores_mat.shape}")

    # Filter degenerate boxes prior to NMS.
    valid_mask = (cand_boxes[:, 2] > cand_boxes[:, 0]) & (cand_boxes[:, 3] > cand_boxes[:, 1])
    cand_boxes = cand_boxes[valid_mask]
    cand_scores = cand_scores[valid_mask]
    cand_classes = cand_classes[valid_mask]
    if cand_boxes.shape[0] == 0:
        return []

    if class_aware:
        boxes_t = torch.from_numpy(cand_boxes)
        scores_t = torch.from_numpy(cand_scores)
        labels_t = torch.from_numpy(cand_classes)
        keep_idx = torchvision.ops.batched_nms(boxes_t, scores_t, labels_t, nms_iou_threshold).numpy()
    else:
        keep_idx = _nms_numpy(cand_boxes, cand_scores, nms_iou_threshold)

    keep_idx = keep_idx[:max_detections]
    kept_boxes = rescale_boxes_to_original(cand_boxes[keep_idx], resize_params)
    kept_scores = cand_scores[keep_idx]
    kept_classes = cand_classes[keep_idx]

    detections: list[Detection] = []
    for box, score, cid in zip(kept_boxes, kept_scores, kept_classes, strict=False):
        cid_int = int(cid)
        cname = class_names[cid_int] if 0 <= cid_int < len(class_names) else f"class_{cid_int}"
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(
            Detection(
                class_id=cid_int,
                class_name=cname,
                score=float(score),
                xmin=float(box[0]),
                ymin=float(box[1]),
                xmax=float(box[2]),
                ymax=float(box[3]),
            )
        )
    detections.sort(key=lambda d: (-d.score, d.class_id))
    return detections
