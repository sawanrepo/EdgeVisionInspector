"""Tests for YOLOX grid/stride decoding, coordinate rescaling, NMS, and PyTorch/ONNX post-NMS parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from edgevision.config import load_yaml, resolve_project_root
from edgevision.data.annotations import build_class_lookup
from edgevision.data.transforms import Preprocessor, ResizeParams
from edgevision.inference.decode_nms import decode_yolox_raw_numpy, postprocess_pre_nms
from edgevision.models.build import build_detector
from edgevision.optimization.export_onnx import export_adapter_to_onnx
from edgevision.optimization.validate_parity import validate_onnx_parity_batch

ROOT = resolve_project_root()


def test_yolox_decode_and_nms_with_small_boundary_and_duplicate_boxes() -> None:
    # 320x320 -> strides 8,16,32 => 40*40 + 20*20 + 10*10 = 2100 anchors
    raw = np.full((1, 2100, 11), -10.0, dtype=np.float32)

    # Anchor 0 (stride 8, grid (0,0)): small boundary box near (0,0) -> score ~0.9*0.9=0.81
    raw[0, 0, :4] = [0.5, 0.5, np.log(2.0), np.log(2.0)]  # cx=4, cy=4, w=16, h=16 -> [-4,-4,12,12]
    raw[0, 0, 4] = 3.0
    raw[0, 0, 5 + 0] = 3.0

    # Anchor 1 (stride 8, grid (1,0)): overlapping duplicate on same class 0 with lower score
    raw[0, 1, :4] = [-0.45, 0.5, np.log(2.0), np.log(2.0)]  # cx=4.4, cy=4, w=16, h=16
    raw[0, 1, 4] = 2.0
    raw[0, 1, 5 + 0] = 2.0

    # Anchor 50 (stride 8): separate distinct box of class 2
    raw[0, 50, :4] = [0.0, 0.0, np.log(4.0), np.log(4.0)]
    raw[0, 50, 4] = 2.5
    raw[0, 50, 5 + 2] = 2.5

    boxes, scores = decode_yolox_raw_numpy(raw, input_size=(320, 320))
    assert boxes.shape == (2100, 4)
    assert scores.shape == (2100, 6)

    params = ResizeParams(
        orig_width=200,
        orig_height=200,
        target_width=320,
        target_height=320,
        scale_x=1.6,
        scale_y=1.6,
        pad_left=0.0,
        pad_top=0.0,
        resize_mode="letterbox",
    )
    dets = postprocess_pre_nms(
        boxes,
        scores,
        resize_params=params,
        class_names=["c0", "c1", "c2", "c3", "c4", "c5"],
        conf_threshold=0.5,
        nms_iou_threshold=0.45,
        max_detections=100,
    )
    # Duplicate anchor 1 suppressed; anchor 0 clipped cleanly to >=0 native image bounds.
    assert len(dets) == 2
    assert dets[0].class_id == 0
    assert dets[0].xmin == 0.0
    assert dets[0].ymin == 0.0
    assert dets[1].class_id == 2


def test_yolox_and_ssdlite_onnx_decoded_and_post_nms_parity(tmp_path: Path) -> None:
    ds_cfg = load_yaml(ROOT / "configs/dataset.yaml")
    specs, _ = build_class_lookup(ds_cfg)
    class_names = [s.name for s in specs]

    rng = np.random.default_rng(99)
    fixtures = [Image.fromarray(rng.integers(0, 255, (200, 200), dtype=np.uint8), mode="L") for _ in range(2)]

    for cfg_rel, size in [("configs/yolox_tiny.yaml", (320, 320)), ("configs/ssdlite.yaml", (320, 320))]:
        cfg = load_yaml(ROOT / cfg_rel)
        arch = cfg["model"]["architecture"]
        adapter = build_detector(cfg, override_pretrained=False, override_input_size=size)
        # Raise one class bias cleanly so post-NMS detection matching runs on non-empty outputs without a 9,600-way tie.
        if arch == "yolox_tiny":
            with torch.no_grad():
                adapter.core.head.obj_preds[2].bias.fill_(2.0)
                adapter.core.head.cls_preds[2].bias[0] = 2.0

        onnx_path = tmp_path / f"{arch}.onnx"
        export_meta = export_adapter_to_onnx(adapter, onnx_path, opset_version=17)
        assert export_meta["status"] == "success"

        pre = Preprocessor(arch, input_size=size, preprocess_cfg=ROOT / "configs/preprocess.yaml")
        report = validate_onnx_parity_batch(
            adapter=adapter,
            onnx_path=onnx_path,
            preprocessor=pre,
            image_sources=fixtures,
            class_names=class_names,
            output_report_path=tmp_path / f"{arch}_parity.json",
            conf_threshold=0.05,
            box_atol=1e-2,
            score_atol=1e-3,
            require_post_nms_match=True,
        )
        assert report["all_passed"] is True
