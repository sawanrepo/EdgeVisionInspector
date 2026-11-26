"""Batch-1 FP32 ONNX export with pinned opset, structural validation, and CPU load verification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
import torch

from edgevision.config import get_logger, sha256_file
from edgevision.models.adapters import BaseDetectorAdapter

logger = get_logger(__name__)


def export_adapter_to_onnx(
    adapter: BaseDetectorAdapter,
    output_path: str | Path,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Export a detector adapter's pre-final-NMS graph at batch size 1 and verify CPU ORT load."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    h, w = adapter.input_size
    arch = adapter.architecture

    export_module = adapter.build_onnx_export_module().to("cpu").eval()
    dummy_input = torch.zeros((1, 3, h, w), dtype=torch.float32)

    if arch == "yolox_tiny":
        output_names = ["raw_head_preds"]
    elif arch == "ssdlite320_mobilenet_v3_large":
        output_names = ["decoded_boxes", "class_probs"]
    elif arch == "faster_rcnn_resnet50_fpn":
        output_names = ["roi_boxes_per_class", "roi_class_probs"]
    else:
        output_names = ["outputs"]

    try:
        with torch.no_grad():
            torch.onnx.export(
                export_module,
                dummy_input,
                str(dest),
                export_params=True,
                opset_version=int(opset_version),
                do_constant_folding=True,
                input_names=["images"],
                output_names=output_names,
                dynamo=False,
            )

        onnx_model = onnx.load(str(dest))
        onnx.checker.check_model(onnx_model)

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        session = ort.InferenceSession(str(dest), sess_options=sess_opts, providers=["CPUExecutionProvider"])
        ort_inputs = {inp.name: list(inp.shape) for inp in session.get_inputs()}
        ort_outputs = {out.name: list(out.shape) for out in session.get_outputs()}

        metadata = {
            "status": "success",
            "architecture": arch,
            "input_size": [h, w],
            "batch_size": 1,
            "opset_version": int(opset_version),
            "onnx_path": str(dest),
            "onnx_sha256": sha256_file(dest),
            "onnx_size_mb": round(dest.stat().st_size / (1024 * 1024), 4),
            "inputs": ort_inputs,
            "outputs": ort_outputs,
            "providers_verified": session.get_providers(),
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
        }
        meta_path = dest.with_suffix(".metadata.json")
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.info("Exported %s ONNX graph to %s (%.2f MB)", arch, dest, metadata["onnx_size_mb"])
        return metadata
    except Exception as exc:
        logger.warning("ONNX export failed for %s: %s", arch, exc)
        return {
            "status": "failed",
            "architecture": arch,
            "input_size": [h, w],
            "opset_version": int(opset_version),
            "error": str(exc),
        }
