"""PyTorch vs ONNX numerical, decoded-box, coordinate-rescaling, and post-NMS detection parity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from edgevision.data.transforms import Preprocessor
from edgevision.evaluation.metrics import compute_iou_matrix
from edgevision.inference.decode_nms import (
    decode_yolox_raw_numpy,
    flatten_faster_rcnn_pre_nms,
    postprocess_pre_nms,
)
from edgevision.models.adapters import BaseDetectorAdapter


def run_onnx_pre_nms(
    session: ort.InferenceSession,
    architecture: str,
    input_size: tuple[int, int],
    input_tensor_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONNX Runtime CPU session and decode raw outputs into pre-NMS (boxes_xyxy, class_scores)."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor_np})

    if architecture == "yolox_tiny":
        return decode_yolox_raw_numpy(outputs[0], input_size=input_size)
    if architecture == "ssdlite320_mobilenet_v3_large":
        boxes = np.asarray(outputs[0], dtype=np.float32)
        probs = np.asarray(outputs[1], dtype=np.float32)
        return (boxes[0] if boxes.ndim == 3 else boxes), (probs[0] if probs.ndim == 3 else probs)
    if architecture == "faster_rcnn_resnet50_fpn":
        return flatten_faster_rcnn_pre_nms(outputs[0], outputs[1])
    raise ValueError(f"Unsupported architecture for ONNX decode: {architecture}")


def check_sample_parity(
    adapter: BaseDetectorAdapter,
    session: ort.InferenceSession,
    preprocessor: Preprocessor,
    image_source: Any,
    class_names: list[str],
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
    box_atol: float = 1e-2,
    score_atol: float = 1e-3,
    require_post_nms_match: bool = True,
) -> dict[str, Any]:
    """Compare PyTorch and ONNX Runtime on input tensor, pre-NMS decoded outputs, and post-NMS detections."""
    pt_tensor, pt_params = preprocessor.preprocess_for_pytorch(image_source, add_batch_dim=True)
    ort_np, ort_params = preprocessor.preprocess_for_onnx(image_source)

    input_max_diff = float(np.max(np.abs(pt_tensor.numpy() - ort_np)))
    input_identical = bool(np.array_equal(pt_tensor.numpy(), ort_np)) and pt_params == ort_params

    adapter.eval()
    with torch.no_grad():
        pt_boxes, pt_scores = adapter.forward_pre_nms(pt_tensor)

    ort_boxes, ort_scores = run_onnx_pre_nms(
        session=session,
        architecture=adapter.architecture,
        input_size=adapter.input_size,
        input_tensor_np=ort_np,
    )

    same_shape = pt_boxes.shape == ort_boxes.shape and pt_scores.shape == ort_scores.shape
    if same_shape and pt_boxes.size > 0:
        max_box_diff = float(np.max(np.abs(pt_boxes - ort_boxes)))
        mean_box_diff = float(np.mean(np.abs(pt_boxes - ort_boxes)))
        max_score_diff = float(np.max(np.abs(pt_scores - ort_scores)))
        mean_score_diff = float(np.mean(np.abs(pt_scores - ort_scores)))
    else:
        max_box_diff = float("inf")
        mean_box_diff = float("inf")
        max_score_diff = float("inf")
        mean_score_diff = float("inf")

    pt_dets = postprocess_pre_nms(
        pt_boxes,
        pt_scores,
        resize_params=pt_params,
        class_names=class_names,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )
    ort_dets = postprocess_pre_nms(
        ort_boxes,
        ort_scores,
        resize_params=ort_params,
        class_names=class_names,
        conf_threshold=conf_threshold,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
    )

    count_match = len(pt_dets) == len(ort_dets)
    det_iou_min = 1.0
    det_score_max_diff = 0.0
    det_class_match = count_match
    if count_match and pt_dets:
        used_ort = [False] * len(ort_dets)
        for d_pt in pt_dets:
            best_idx = -1
            best_iou = -1.0
            for idx, d_ort in enumerate(ort_dets):
                if used_ort[idx] or d_ort.class_id != d_pt.class_id:
                    continue
                iou = float(
                    compute_iou_matrix(
                        np.array([[d_pt.xmin, d_pt.ymin, d_pt.xmax, d_pt.ymax]]),
                        np.array([[d_ort.xmin, d_ort.ymin, d_ort.xmax, d_ort.ymax]]),
                    )[0, 0]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx < 0:
                det_class_match = False
                det_iou_min = 0.0
                break
            used_ort[best_idx] = True
            det_score_max_diff = max(det_score_max_diff, abs(d_pt.score - ort_dets[best_idx].score))
            det_iou_min = min(det_iou_min, best_iou)

    pre_nms_passed = (
        input_identical
        and same_shape
        and max_box_diff <= box_atol
        and max_score_diff <= score_atol
    )
    post_nms_passed = (not require_post_nms_match) or (
        count_match and det_class_match and det_iou_min >= 0.99 and det_score_max_diff <= score_atol
    )

    return {
        "input_identical": input_identical,
        "input_max_diff": input_max_diff,
        "pre_nms_shape_match": same_shape,
        "pre_nms_max_box_diff": max_box_diff,
        "pre_nms_mean_box_diff": mean_box_diff,
        "pre_nms_max_score_diff": max_score_diff,
        "pre_nms_mean_score_diff": mean_score_diff,
        "pt_det_count": len(pt_dets),
        "ort_det_count": len(ort_dets),
        "post_nms_class_match": det_class_match,
        "post_nms_min_iou": round(det_iou_min, 6),
        "post_nms_max_score_diff": det_score_max_diff,
        "passed": bool(pre_nms_passed and post_nms_passed),
    }


def validate_onnx_parity_batch(
    adapter: BaseDetectorAdapter,
    onnx_path: str | Path,
    preprocessor: Preprocessor,
    image_sources: list[Any],
    class_names: list[str],
    output_report_path: str | Path | None = None,
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
    box_atol: float = 1e-2,
    score_atol: float = 1e-3,
    require_post_nms_match: bool = True,
) -> dict[str, Any]:
    """Run PyTorch vs ONNX Runtime parity across multiple images and optionally save JSON report."""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])

    sample_results = [
        check_sample_parity(
            adapter=adapter,
            session=session,
            preprocessor=preprocessor,
            image_source=src,
            class_names=class_names,
            conf_threshold=conf_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            box_atol=box_atol,
            score_atol=score_atol,
            require_post_nms_match=require_post_nms_match,
        )
        for src in image_sources
    ]

    all_passed = all(r["passed"] for r in sample_results)
    report = {
        "architecture": adapter.architecture,
        "input_size": list(adapter.input_size),
        "onnx_path": str(onnx_path),
        "num_samples": len(sample_results),
        "all_passed": all_passed,
        "tolerances": {"box_atol": box_atol, "score_atol": score_atol},
        "max_pre_nms_box_diff": max((r["pre_nms_max_box_diff"] for r in sample_results), default=0.0),
        "max_pre_nms_score_diff": max((r["pre_nms_max_score_diff"] for r in sample_results), default=0.0),
        "samples": sample_results,
    }
    if output_report_path is not None:
        dest = Path(output_report_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report
