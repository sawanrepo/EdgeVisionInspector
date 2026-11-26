"""Tests for FP32 ONNX export, INT8 static QDQ quantization, and golden-calibration protection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgevision.config import load_yaml, resolve_project_root
from edgevision.data.transforms import Preprocessor
from edgevision.inference.predict import OnnxPredictor
from edgevision.models.build import build_detector
from edgevision.optimization.export_onnx import export_adapter_to_onnx
from edgevision.optimization.quantize import build_calibration_records, quantize_onnx_static_qdq

ROOT = resolve_project_root()


@pytest.mark.parametrize(
    ("cfg_rel", "input_size"),
    [
        ("configs/yolox_tiny.yaml", (320, 320)),
        ("configs/ssdlite.yaml", (320, 320)),
    ],
)
def test_onnx_export_and_int8_quantization_smoke(
    cfg_rel: str,
    input_size: tuple[int, int],
    sample_image_records,
    tmp_path: Path,
) -> None:
    cfg = load_yaml(ROOT / cfg_rel)
    arch = cfg["model"]["architecture"]
    adapter = build_detector(cfg, override_pretrained=False, override_input_size=input_size)

    fp32_path = tmp_path / f"{arch}_fp32.onnx"
    meta = export_adapter_to_onnx(adapter, fp32_path, opset_version=17)
    assert meta["status"] == "success"
    assert fp32_path.exists()
    assert meta["providers_verified"] == ["CPUExecutionProvider"]

    pre = Preprocessor(arch, input_size=input_size, preprocess_cfg=ROOT / "configs/preprocess.yaml")
    int8_path = tmp_path / f"{arch}_int8.onnx"
    q_meta = quantize_onnx_static_qdq(
        fp32_onnx_path=fp32_path,
        output_int8_path=int8_path,
        calibration_records=sample_image_records[:3],
        preprocessor=pre,
        per_channel=True,
        keep_head_fp32=False,
    )
    assert int8_path.exists()
    assert q_meta["operator_coverage"]["total_nodes"] > 0

    predictor = OnnxPredictor(
        onnx_path=int8_path,
        architecture=arch,
        input_size=input_size,
        dataset_cfg=ROOT / "configs/dataset.yaml",
        preprocess_cfg=ROOT / "configs/preprocess.yaml",
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        conf_threshold=0.05,
    )
    img_arr = np.full((200, 200), 128, dtype=np.uint8)
    result = predictor.predict(img_arr)
    assert result.image_width == 200
    assert result.image_height == 200
    assert result.latency.end_to_end_in_process_latency_ms > 0.0


def test_calibration_rejects_golden_test_manifest(tmp_path: Path) -> None:
    golden = tmp_path / "golden_test.jsonl"
    golden.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="golden"):
        build_calibration_records([golden], num_images=10)
