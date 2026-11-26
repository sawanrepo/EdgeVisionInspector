"""Shared CPU-first ONNX Runtime inference predictor with exact latency-boundary instrumentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from edgevision.data.annotations import build_class_lookup
from edgevision.data.transforms import Preprocessor, decode_grayscale_to_rgb_array
from edgevision.inference.decode_nms import (
    Detection,
    decode_yolox_raw_numpy,
    flatten_faster_rcnn_pre_nms,
    postprocess_pre_nms,
)


@dataclass(frozen=True)
class LatencyBreakdown:
    """Measured per-stage latency in milliseconds."""

    decode_image_ms: float
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float
    response_ms: float

    @property
    def model_latency_ms(self) -> float:
        return self.model_ms

    @property
    def inference_pipeline_latency_ms(self) -> float:
        return self.preprocess_ms + self.model_ms + self.postprocess_ms

    @property
    def end_to_end_in_process_latency_ms(self) -> float:
        return self.decode_image_ms + self.preprocess_ms + self.model_ms + self.postprocess_ms + self.response_ms


@dataclass
class InferenceResult:
    """Predictions and latency breakdown for one image."""

    detections: list[Detection]
    latency: LatencyBreakdown
    image_width: int
    image_height: int


class OnnxPredictor:
    """CPU-only ONNX Runtime predictor matching benchmark and FastAPI serving semantics."""

    def __init__(
        self,
        onnx_path: str | Path,
        architecture: str,
        input_size: tuple[int, int] | list[int],
        dataset_cfg: dict[str, Any] | str | Path = "configs/dataset.yaml",
        preprocess_cfg: dict[str, Any] | str | Path = "configs/preprocess.yaml",
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 1,
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_detections: int = 100,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        self.architecture = architecture
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(conf_threshold)
        self.nms_iou_threshold = float(nms_iou_threshold)
        self.max_detections = int(max_detections)
        self.intra_op_num_threads = int(intra_op_num_threads)
        self.inter_op_num_threads = int(inter_op_num_threads)

        from edgevision.config import load_yaml

        ds_cfg = load_yaml(dataset_cfg) if isinstance(dataset_cfg, (str, Path)) else dataset_cfg
        specs, _ = build_class_lookup(ds_cfg)
        self.class_names = [spec.name for spec in specs]
        self.preprocessor = Preprocessor(
            architecture=architecture,
            input_size=self.input_size,
            preprocess_cfg=preprocess_cfg,
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.intra_op_num_threads
        sess_options.inter_op_num_threads = self.inter_op_num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        active_providers = self.session.get_providers()
        if active_providers != ["CPUExecutionProvider"]:
            raise RuntimeError(
                f"Expected CPUExecutionProvider only, got {active_providers}. "
                "Do not run CPU inference on GPU execution providers."
            )
        self.input_name = self.session.get_inputs()[0].name

    def predict(
        self,
        image_source: Any,
        conf_threshold: float | None = None,
    ) -> InferenceResult:
        """Run end-to-end single-image inference with fine-grained stage timing."""
        thr = self.conf_threshold if conf_threshold is None else float(conf_threshold)

        t0 = time.perf_counter()
        rgb = decode_grayscale_to_rgb_array(image_source)
        t1 = time.perf_counter()

        input_np, resize_params = self.preprocessor.preprocess_for_onnx(rgb)
        t2 = time.perf_counter()

        outputs = self.session.run(None, {self.input_name: input_np})
        t3 = time.perf_counter()

        if self.architecture == "yolox_tiny":
            boxes_xyxy, scores = decode_yolox_raw_numpy(outputs[0], input_size=self.input_size)
        elif self.architecture == "ssdlite320_mobilenet_v3_large":
            b = np.asarray(outputs[0], dtype=np.float32)
            p = np.asarray(outputs[1], dtype=np.float32)
            boxes_xyxy = b[0] if b.ndim == 3 else b
            scores = p[0] if p.ndim == 3 else p
        elif self.architecture == "faster_rcnn_resnet50_fpn":
            boxes_xyxy, scores = flatten_faster_rcnn_pre_nms(outputs[0], outputs[1])
        else:
            raise ValueError(f"Unsupported architecture '{self.architecture}'")

        detections = postprocess_pre_nms(
            boxes_xyxy=boxes_xyxy,
            class_scores=scores,
            resize_params=resize_params,
            class_names=self.class_names,
            conf_threshold=thr,
            nms_iou_threshold=self.nms_iou_threshold,
            max_detections=self.max_detections,
            class_aware=True,
        )
        t4 = time.perf_counter()

        # Construct serializable response payload inside the measured response stage.
        _ = [d.to_dict() for d in detections]
        t5 = time.perf_counter()

        latency = LatencyBreakdown(
            decode_image_ms=(t1 - t0) * 1000.0,
            preprocess_ms=(t2 - t1) * 1000.0,
            model_ms=(t3 - t2) * 1000.0,
            postprocess_ms=(t4 - t3) * 1000.0,
            response_ms=(t5 - t4) * 1000.0,
        )
        return InferenceResult(
            detections=detections,
            latency=latency,
            image_width=resize_params.orig_width,
            image_height=resize_params.orig_height,
        )
