"""Integration tests for FastAPI /health and /predict endpoints using a real ONNX CPU predictor."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import Headers, UploadFile

from edgevision.config import load_yaml, resolve_project_root
from edgevision.inference.predict import OnnxPredictor
from edgevision.inference.service import ServiceSettings, create_app
from edgevision.models.build import build_detector
from edgevision.optimization.export_onnx import export_adapter_to_onnx

ROOT = resolve_project_root()


def _endpoint_by_path(app, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise KeyError(f"Route {path} not found")


def test_api_health_predict_and_error_handling(tmp_path: Path) -> None:
    cfg = load_yaml(ROOT / "configs/yolox_tiny.yaml")
    adapter = build_detector(cfg, override_pretrained=False, override_input_size=(320, 320))
    onnx_path = tmp_path / "yolox_api.onnx"
    export_adapter_to_onnx(adapter, onnx_path, opset_version=17)

    predictor = OnnxPredictor(
        onnx_path=onnx_path,
        architecture="yolox_tiny",
        input_size=(320, 320),
        dataset_cfg=ROOT / "configs/dataset.yaml",
        preprocess_cfg=ROOT / "configs/preprocess.yaml",
        intra_op_num_threads=1,
        inter_op_num_threads=1,
        conf_threshold=0.25,
    )
    settings = ServiceSettings(model_path=str(onnx_path), max_file_bytes=512 * 1024)
    app = create_app(settings=settings, predictor=predictor)

    fake_request = SimpleNamespace(app=app)
    health_fn = _endpoint_by_path(app, "/health")
    predict_fn = _endpoint_by_path(app, "/predict")

    # 1. Health check with loaded predictor
    h_resp = health_fn(fake_request)
    assert h_resp.status == "ok"
    assert h_resp.model_loaded is True
    assert h_resp.execution_provider == "CPUExecutionProvider"

    # 2. Valid grayscale PNG upload
    buf = io.BytesIO()
    Image.fromarray(np.full((200, 200), 140, dtype=np.uint8), mode="L").save(buf, format="PNG")
    valid_upload = UploadFile(
        filename="sample.png",
        file=io.BytesIO(buf.getvalue()),
        headers=Headers({"content-type": "image/png"}),
    )
    p_resp = predict_fn(fake_request, valid_upload)
    assert p_resp.image_width == 200
    assert p_resp.image_height == 200
    assert p_resp.bbox_convention == "xyxy_pixels"
    assert p_resp.num_detections == len(p_resp.detections)
    assert p_resp.latency_ms.end_to_end_in_process_ms > 0.0

    # 3. Unsupported media type -> 415
    bad_mime = UploadFile(
        filename="notes.txt",
        file=io.BytesIO(b"hello"),
        headers=Headers({"content-type": "text/plain"}),
    )
    with pytest.raises(HTTPException) as exc_mime:
        predict_fn(fake_request, bad_mime)
    assert exc_mime.value.status_code == 415

    # 4. Empty upload -> 400
    empty_up = UploadFile(
        filename="empty.png",
        file=io.BytesIO(b""),
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(HTTPException) as exc_empty:
        predict_fn(fake_request, empty_up)
    assert exc_empty.value.status_code == 400

    # 5. Corrupt image payload -> 400
    corrupt_up = UploadFile(
        filename="bad.png",
        file=io.BytesIO(b"not_png_bytes"),
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(HTTPException) as exc_corrupt:
        predict_fn(fake_request, corrupt_up)
    assert exc_corrupt.value.status_code == 400

    # 6. Oversized payload (> 512 KB) -> 413
    big_up = UploadFile(
        filename="big.png",
        file=io.BytesIO(b"x" * (600 * 1024)),
        headers=Headers({"content-type": "image/png"}),
    )
    with pytest.raises(HTTPException) as exc_big:
        predict_fn(fake_request, big_up)
    assert exc_big.value.status_code == 413

    # 7. Unloaded predictor -> 503
    unloaded_app = create_app(settings=ServiceSettings(model_path="nonexistent.onnx"), predictor=None)
    with pytest.raises(HTTPException) as exc_503:
        health_fn(SimpleNamespace(app=unloaded_app))
    assert exc_503.value.status_code == 503

    # 8. Full Starlette TestClient HTTP stack on POSIX / CI runners
    if sys.platform != "win32" or os.environ.get("CI") == "true":
        from fastapi.testclient import TestClient

        with TestClient(app) as client:
            r_health = client.get("/health")
            assert r_health.status_code == 200
            r_pred = client.post("/predict", files={"file": ("sample.png", buf.getvalue(), "image/png")})
            assert r_pred.status_code == 200
            assert r_pred.json()["bbox_convention"] == "xyxy_pixels"
