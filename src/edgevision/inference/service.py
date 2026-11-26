"""CPU-first FastAPI inference service exposing /health and /predict."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from edgevision.config import get_logger
from edgevision.inference.predict import OnnxPredictor

logger = get_logger(__name__)

ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/x-portable-graymap",
    "application/octet-stream",
}


@dataclass(frozen=True)
class ServiceSettings:
    """Runtime configuration for the inference API loaded from environment variables."""

    model_path: str = os.environ.get("EDGEVISION_MODEL_PATH", "artifacts/model_fp32.onnx")
    architecture: str = os.environ.get("EDGEVISION_ARCH", "yolox_tiny")
    input_height: int = int(os.environ.get("EDGEVISION_INPUT_HEIGHT", "320"))
    input_width: int = int(os.environ.get("EDGEVISION_INPUT_WIDTH", "320"))
    dataset_cfg: str = os.environ.get("EDGEVISION_DATASET_CFG", "configs/dataset.yaml")
    preprocess_cfg: str = os.environ.get("EDGEVISION_PREPROCESS_CFG", "configs/preprocess.yaml")
    conf_threshold: float = float(os.environ.get("EDGEVISION_CONF_THRESHOLD", "0.25"))
    nms_iou_threshold: float = float(os.environ.get("EDGEVISION_NMS_IOU", "0.45"))
    max_detections: int = int(os.environ.get("EDGEVISION_MAX_DETECTIONS", "100"))
    intra_op_num_threads: int = int(os.environ.get("EDGEVISION_INTRA_THREADS", "4"))
    inter_op_num_threads: int = int(os.environ.get("EDGEVISION_INTER_THREADS", "1"))
    max_file_bytes: int = int(os.environ.get("EDGEVISION_MAX_FILE_BYTES", str(10 * 1024 * 1024)))


class DetectionResponseItem(BaseModel):
    """Single detected surface defect in native image xyxy pixel coordinates."""

    class_id: int
    class_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    bbox_xyxy: list[float] = Field(..., min_length=4, max_length=4)


class LatencyResponse(BaseModel):
    """Per-stage in-process inference latency measurements in milliseconds."""

    decode_image_ms: float
    preprocess_ms: float
    model_ms: float
    postprocess_ms: float
    end_to_end_in_process_ms: float


class PredictResponse(BaseModel):
    """Response schema for POST /predict."""

    image_width: int
    image_height: int
    bbox_convention: str = "xyxy_pixels"
    num_detections: int
    detections: list[DetectionResponseItem]
    latency_ms: LatencyResponse


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    architecture: str
    input_size: list[int]
    execution_provider: str


def _build_predictor_from_settings(settings: ServiceSettings) -> OnnxPredictor:
    return OnnxPredictor(
        onnx_path=settings.model_path,
        architecture=settings.architecture,
        input_size=(settings.input_height, settings.input_width),
        dataset_cfg=settings.dataset_cfg,
        preprocess_cfg=settings.preprocess_cfg,
        intra_op_num_threads=settings.intra_op_num_threads,
        inter_op_num_threads=settings.inter_op_num_threads,
        conf_threshold=settings.conf_threshold,
        nms_iou_threshold=settings.nms_iou_threshold,
        max_detections=settings.max_detections,
    )


def create_app(
    settings: ServiceSettings | None = None,
    predictor: OnnxPredictor | None = None,
) -> FastAPI:
    """Create FastAPI application that loads the ONNX CPU predictor once on startup."""
    cfg = settings or ServiceSettings()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        if predictor is not None:
            app_instance.state.predictor = predictor
            app_instance.state.load_error = None
        else:
            try:
                app_instance.state.predictor = _build_predictor_from_settings(cfg)
                app_instance.state.load_error = None
                logger.info("Loaded ONNX CPU predictor from %s", cfg.model_path)
            except Exception as exc:
                app_instance.state.predictor = None
                app_instance.state.load_error = str(exc)
                logger.error("Failed to load model at startup: %s", exc)
        yield

    app = FastAPI(
        title="EdgeVision Inspector API",
        version="0.1.0",
        description="CPU-first industrial steel-surface defect detection service.",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.predictor = predictor
    app.state.load_error = None

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        active_predictor: Any = getattr(request.app.state, "predictor", None)
        if active_predictor is None:
            detail = getattr(request.app.state, "load_error", None) or "Model not loaded"
            raise HTTPException(status_code=503, detail=detail)
        return HealthResponse(
            status="ok",
            model_loaded=True,
            architecture=active_predictor.architecture,
            input_size=list(active_predictor.input_size),
            execution_provider="CPUExecutionProvider",
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, file: Annotated[UploadFile, File(...)]) -> PredictResponse:
        active_predictor: OnnxPredictor | None = getattr(request.app.state, "predictor", None)
        if active_predictor is None:
            raise HTTPException(status_code=503, detail="Inference model is not loaded")

        content_type = (file.content_type or "application/octet-stream").lower()
        if content_type not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported media type '{content_type}'. Upload a valid image file.",
            )

        raw_bytes = file.file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty")
        if len(raw_bytes) > cfg.max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {cfg.max_file_bytes} bytes",
            )

        try:
            result = active_predictor.predict(raw_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

        det_items = [DetectionResponseItem(**d.to_dict()) for d in result.detections]
        lat = result.latency
        return PredictResponse(
            image_width=result.image_width,
            image_height=result.image_height,
            bbox_convention="xyxy_pixels",
            num_detections=len(det_items),
            detections=det_items,
            latency_ms=LatencyResponse(
                decode_image_ms=round(lat.decode_image_ms, 4),
                preprocess_ms=round(lat.preprocess_ms, 4),
                model_ms=round(lat.model_ms, 4),
                postprocess_ms=round(lat.postprocess_ms, 4),
                end_to_end_in_process_ms=round(lat.end_to_end_in_process_latency_ms, 4),
            ),
        )

    return app


app = create_app()
