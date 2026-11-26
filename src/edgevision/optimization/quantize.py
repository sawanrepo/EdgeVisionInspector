"""Static QDQ INT8 post-training quantization, selective FP32 fallback, and operator coverage audit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from edgevision.config import get_logger, sha256_file
from edgevision.data.annotations import ImageRecord, load_records_jsonl
from edgevision.data.transforms import Preprocessor

logger = get_logger(__name__)


class DefectCalibrationDataReader(CalibrationDataReader):
    """ONNX Runtime CalibrationDataReader backed strictly by train/validation images."""

    def __init__(
        self,
        records: list[ImageRecord],
        preprocessor: Preprocessor,
        input_name: str,
    ) -> None:
        self.records = records
        self.preprocessor = preprocessor
        self.input_name = input_name
        self._iter = iter(self.records)

    def get_next(self) -> dict[str, np.ndarray] | None:
        rec = next(self._iter, None)
        if rec is None:
            return None
        arr, _ = self.preprocessor.preprocess_for_onnx(rec.image_path)
        return {self.input_name: arr}

    def rewind(self) -> None:
        self._iter = iter(self.records)


def build_calibration_records(
    manifest_paths: list[str | Path],
    num_images: int = 200,
    seed: int = 42,
) -> list[ImageRecord]:
    """Sample representative calibration images while rejecting any golden-test manifest."""
    all_records: list[ImageRecord] = []
    for path in manifest_paths:
        p = Path(path)
        if "golden" in p.name.lower() or "test" in p.name.lower():
            raise ValueError(
                f"Refusing to use test/golden manifest '{p}' for INT8 quantization calibration."
            )
        all_records.extend(load_records_jsonl(p))

    if not all_records:
        raise ValueError("Calibration manifest set produced zero records")

    rng = np.random.default_rng(seed)
    all_records.sort(key=lambda r: r.image_id)
    count = min(num_images, len(all_records))
    chosen_idx = rng.choice(len(all_records), size=count, replace=False)
    return [all_records[int(i)] for i in sorted(chosen_idx)]


def inspect_onnx_operator_coverage(onnx_path: str | Path) -> dict[str, Any]:
    """Inspect quantized vs FP32 operator distribution in an ONNX graph."""
    model = onnx.load(str(onnx_path))
    op_counts = Counter(node.op_type for node in model.graph.node)
    quant_ops = {
        op: cnt
        for op, cnt in op_counts.items()
        if "Quantize" in op or "QLinear" in op or "QGemm" in op or "Integer" in op
    }
    return {
        "total_nodes": len(model.graph.node),
        "op_type_counts": dict(sorted(op_counts.items())),
        "quantized_op_counts": dict(sorted(quant_ops.items())),
    }


def _find_excluded_nodes(onnx_path: str | Path, patterns: list[str]) -> list[str]:
    model = onnx.load(str(onnx_path))
    lowered = [p.lower() for p in patterns]
    excluded: list[str] = []
    for node in model.graph.node:
        name = (node.name or "").lower()
        outputs = [o.lower() for o in node.output]
        if any(pat in name or any(pat in out for out in outputs) for pat in lowered):
            if node.name:
                excluded.append(node.name)
    return sorted(set(excluded))


def quantize_onnx_static_qdq(
    fp32_onnx_path: str | Path,
    output_int8_path: str | Path,
    calibration_records: list[ImageRecord],
    preprocessor: Preprocessor,
    per_channel: bool = True,
    keep_head_fp32: bool = False,
    fp32_node_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Run static QDQ INT8 quantization with optional selective FP32 head preservation."""
    fp32_path = Path(fp32_onnx_path)
    out_path = Path(output_int8_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_path = out_path.with_suffix(".pre_quant.onnx")
    try:
        quant_pre_process(str(fp32_path), str(pre_path), skip_symbolic_shape=False)
        source_for_quant = pre_path
    except Exception as exc:
        logger.warning("quant_pre_process fallback to raw FP32 ONNX (%s): %s", fp32_path, exc)
        source_for_quant = fp32_path

    model = onnx.load(str(source_for_quant))
    input_name = model.graph.input[0].name
    reader = DefectCalibrationDataReader(
        records=calibration_records,
        preprocessor=preprocessor,
        input_name=input_name,
    )

    nodes_to_exclude: list[str] = []
    if keep_head_fp32:
        patterns = fp32_node_patterns or ["head", "cls_preds", "reg_preds", "obj_preds", "box_predictor"]
        nodes_to_exclude = _find_excluded_nodes(source_for_quant, patterns)

    quantize_static(
        model_input=str(source_for_quant),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=nodes_to_exclude,
    )

    if pre_path.exists():
        pre_path.unlink()

    coverage = inspect_onnx_operator_coverage(out_path)
    metadata = {
        "stage": "selective_fp32_head" if keep_head_fp32 else "per_channel_static_qdq",
        "fp32_source": str(fp32_path),
        "int8_onnx_path": str(out_path),
        "int8_sha256": sha256_file(out_path),
        "int8_size_mb": round(out_path.stat().st_size / (1024 * 1024), 4),
        "per_channel": per_channel,
        "keep_head_fp32": keep_head_fp32,
        "excluded_node_count": len(nodes_to_exclude),
        "excluded_nodes": nodes_to_exclude[:50],
        "calibration_image_ids": [r.image_id for r in calibration_records],
        "operator_coverage": coverage,
    }
    out_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    logger.info(
        "Quantized %s -> %s (%.2f MB, %d excluded nodes)",
        fp32_path.name,
        out_path.name,
        metadata["int8_size_mb"],
        len(nodes_to_exclude),
    )
    return metadata
