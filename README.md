# EdgeVision Inspector

Reproducible industrial steel-surface defect detection system built around PyTorch, ONNX Runtime CPU inference, static INT8 post-training quantization, statistical operating-point evaluation, and a FastAPI microservice.

> **Benchmark Scope Note:** This system is developed and evaluated as a benchmark-based inspection prototype on NEU-DET (200 x 200 pre-cropped grayscale surface tiles). Benchmark performance on NEU-DET does not by itself validate a live steel production-line inspection deployment.

---

## 1. Problem Statement and Defect Classes

EdgeVision Inspector localizes and classifies six steel-surface defect categories using bounding-box object detection:

| Class ID | Canonical Name | Display Name |
|---:|---|---|
| 0 | `crazing` | Crazing |
| 1 | `inclusion` | Inclusion |
| 2 | `patches` | Patches |
| 3 | `pitted_surface` | Pitted surface |
| 4 | `rolled_in_scale` | Rolled-in scale |
| 5 | `scratches` | Scratches |

### Engineering Objectives

- **Deployment Inference Target:** Batch size 1, P95 end-to-end in-process CPU latency <= 100 ms on a recorded CPU-only runtime (`CPUExecutionProvider`).
- **Detection Target:** Macro recall >= 90% at IoU 0.5 using global confidence thresholds frozen strictly from validation data.
- **Measurement Discipline:** Latency boundaries (model forward pass, inference pipeline, end-to-end in-process, and HTTP API) are instrumented and reported separately.

---

## 2. Dataset, Licence Gate, and Splitting Methodology

- **Dataset:** NEU-DET (1,800 nominal 200 x 200 grayscale images across six defect categories, with Pascal VOC XML bounding-box annotations).
- **Source & Licence Gate:** Raw images and annotations are not redistributed in this repository. Before training or publishing empirical results, dataset source URL, version, access date, and redistribution terms must be recorded in [dataset.yaml](configs/dataset.yaml) and verified via `edgevision audit`.
- **Annotation Validation:** [annotations.py](src/edgevision/data/annotations.py) parses Pascal VOC XML, normalizes class aliases, and flags malformed XML, empty annotations, non-positive boxes, out-of-bounds coordinates, unknown classes, and duplicate boxes without silent repair.
- **Leakage-Free Splitting:** [audit.py](src/edgevision/data/audit.py) and [splits.py](src/edgevision/data/splits.py) detect exact pixel-SHA256 duplicates and near-duplicates (64-bit dHash + cosine similarity), group them before partitioning, and run multi-label stratified group splitting (70% train / 15% validation / 15% held-out golden test) with strict disjointness and manifest SHA-256 checksums.
- **Evaluation Views:** In addition to the held-out golden test manifest (`data/manifests/golden_test.jsonl`), `splits.py` freezes a model-independent natural hard subset (`natural_hard_subset.jsonl`, up to 50 images) and a deterministic controlled corruption suite (`corruption_suite.jsonl`, derived stress tests).

---

## 3. Preprocessing Contract and Resolution Protocol

### Canonical Preprocessing (`configs/preprocess.yaml`)

All detectors consume three-channel grayscale-replicated inputs (`L -> RGB`) defined once in [preprocess.yaml](configs/preprocess.yaml) and executed identically by [transforms.py](src/edgevision/data/transforms.py) for PyTorch, ONNX FP32, and ONNX INT8:

| Architecture | Input Sizes | Resize Mode | Pixel Scale | Mean / Std |
|---|---|---|---:|---|
| `yolox_tiny` | 224x224, 256x256, 320x320 | `letterbox` (pad 114.0) | `1.0` | `[0, 0, 0]` / `[1, 1, 1]` |
| `ssdlite320_mobilenet_v3_large` | 320x320 (native) | `direct` | `1/255` | `[0.5, 0.5, 0.5]` / `[0.5, 0.5, 0.5]` |
| `faster_rcnn_resnet50_fpn` | 224x224, 256x256, 320x320 | `direct` | `1/255` | ImageNet mean / std |

### Input-Resolution Comparison Design

Because native NEU-DET images are 200 x 200 and Torchvision's `ssdlite320_mobilenet_v3_large` relies on a native 320 x 320 anchor-generator layout:

- **Matched 320 x 320 Deployment Comparison:** YOLOX-Tiny, SSDLite320 MobileNetV3-Large, and Faster R-CNN ResNet50-FPN are all evaluated at 320 x 320 for fair candidate selection.
- **Low-Resolution 224 x 224 Comparison (Two-Model):** YOLOX-Tiny and Faster R-CNN are additionally compared at 224 x 224 (and optionally 256 x 256) without modifying SSDLite's native anchor configuration.

---

## 4. Model Architectures and Selection Rule

1. **YOLOX-Tiny** (`src/edgevision/models/yolox.py`, `configs/yolox_tiny.yaml`): Anchor-free decoupled-head detector (`depth=0.33, width=0.375`), supporting configurable input sizes (`224`, `256`, `320`) and raw-head ONNX export with external pinned grid/stride decoding and class-aware NMS.
2. **Torchvision SSDLite320 MobileNetV3-Large** (`configs/ssdlite.yaml`): Lightweight single-stage detector preserving native 320 x 320 default boxes.
3. **Torchvision Faster R-CNN ResNet50-FPN** (`configs/faster_rcnn.yaml`): Two-stage accuracy-oriented reference model.

### Predeclared Validation-Only Selection Rule

All matched-320 candidates are exported to batch-1 FP32 ONNX screening graphs, verified for decoded-output and post-NMS parity, and benchmarked on the same CPU runtime (`CPUExecutionProvider`):

1. Among deployment-eligible candidates with P95 end-to-end latency <= 100 ms at the validation recall-target operating point, pick highest validation mAP@0.5 (tie-break: higher macro recall, smaller ONNX size).
2. If none meets 100 ms but at least one satisfies >= 90% validation macro recall, pick the lowest P95 latency among recall-feasible candidates.
3. If none satisfies >= 90% validation macro recall, pick the highest validation macro recall (tie-break: lower P95 latency, higher mAP@0.5) and report the recall target as infeasible.

---

## 5. Statistical Evaluation and Golden-Test Protection

- **Two Frozen Global Thresholds (`eval/operating_points.json`):**
  - *Quality Operating Point:* Global confidence threshold maximizing validation macro-F1.
  - *Recall-Target Operating Point:* Highest global confidence threshold achieving >= 90% validation macro recall (or marked infeasible if unachievable).
- **Uncertainty and Per-Class Support:** Image-level cluster bootstrap (2,000 replicates) computes standard errors and 95% confidence intervals for per-class recall and macro recall alongside instance and image support counts.
- **Validation-Only Regression Gate (`eval/validation_baseline.json`):** Freezes validation mAP@0.5 and per-class recall floors `floor_c = clip(R_val,c - max(0.03, 1.5 * SE_c), 0, 1)` before golden-test evaluation.
- **Single-Run Golden Test (`eval/final_test_report.json`):** Held-out golden test data never participates in model selection, threshold tuning, calibration, or CI regression checks, and is evaluated once when candidate artifacts and thresholds are frozen.

---

## 6. ONNX Export, External NMS, and INT8 Optimization

- **External Decode & NMS (`src/edgevision/inference/decode_nms.py`):** Final box decoding/rescaling and final class-aware NMS execute outside the ONNX graph on CPU and are timed inside inference-pipeline and end-to-end latency.
- **Static QDQ INT8 Fallback Ladder (`configs/int8.yaml`, `src/edgevision/optimization/quantize.py`):**
  1. Per-channel static QDQ quantization calibrated strictly on train/validation images.
  2. Selective quantization retaining detection head / decode paths in FP32.
  3. Timeboxed QAT (capped at 3 GPU-hours) if prior stages fail quality/latency criteria.

---

## 7. Experimental Results and Benchmarks

> **Status:** Experimental training, Kaggle CPU benchmarking, and golden-test evaluation artifacts have not been executed yet in this repository. Populate the tables below exclusively from committed `reports/model_comparison.csv`, `benchmarks/benchmark_report.json`, and `eval/final_test_report.json` outputs after running the pipeline on verified NEU-DET data.

---

## 8. Reproducing Experiments

### Local Installation and Tests

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
pytest -v
ruff check src tests
```

### CLI Workflow

```bash
# 1. Verify runtime and CUDA GPUs (or --cpu-only)
edgevision env-check --output reports/env_gpu.json

# 2. Audit dataset and verify licence gate
edgevision audit --dataset-cfg configs/dataset.yaml

# 3. Generate leakage-free splits and evaluation manifests
edgevision split --dataset-cfg configs/dataset.yaml

# 4. Train candidates (example: two independent runs on Kaggle T4 x2)
edgevision train --model-cfg configs/yolox_tiny.yaml --output-dir runs/yolox_tiny_320 --device cuda:0
edgevision train --model-cfg configs/ssdlite.yaml --output-dir runs/ssdlite_320 --device cuda:1

# 5. Export FP32 ONNX & verify decoded + post-NMS parity
edgevision export-onnx --model-cfg configs/yolox_tiny.yaml --checkpoint runs/yolox_tiny_320/best.pt --output-onnx artifacts/model_fp32.onnx --val-manifest data/manifests/val.jsonl

# 6. Freeze operating points & validation baseline
edgevision evaluate-val --onnx-path artifacts/model_fp32.onnx --architecture yolox_tiny --input-size 320 320

# 7. INT8 static QDQ quantization (train/val calibration only)
edgevision quantize-int8 --fp32-onnx artifacts/model_fp32.onnx --output-int8 artifacts/model_int8.onnx --architecture yolox_tiny --input-size 320 320

# 8. Controlled CPU-only benchmark (run in CPU-only runtime)
edgevision benchmark-cpu --onnx-path artifacts/model_int8.onnx --architecture yolox_tiny --input-size 320 320 --manifest data/manifests/val.jsonl

# 9. Single controlled golden-test evaluation (run once after freezing artifacts)
edgevision evaluate-golden --onnx-path artifacts/model_int8.onnx --architecture yolox_tiny --input-size 320 320
```

Notebooks in [notebooks/](notebooks/) mirror this workflow for Kaggle GPU and CPU-only sessions.

---

## 9. FastAPI Inference Service and Docker

### Run Locally

```bash
export EDGEVISION_MODEL_PATH=artifacts/model_fp32.onnx
export EDGEVISION_ARCH=yolox_tiny
export EDGEVISION_INPUT_HEIGHT=320
export EDGEVISION_INPUT_WIDTH=320
uvicorn edgevision.inference.service:app --host 0.0.0.0 --port 8000
```

### Docker Build and Run

```bash
docker build -t edgevision-inspector:latest .
docker run --rm -p 8000:8000 \
  -e EDGEVISION_MODEL_PATH=/app/artifacts/model_fp32.onnx \
  -e EDGEVISION_ARCH=yolox_tiny \
  edgevision-inspector:latest
```

### Endpoints

- `GET /health`: Confirms model load status, architecture, input size, and `CPUExecutionProvider`.
- `POST /predict`: Accepts multipart image upload (`file`) and returns bounding boxes (`xyxy_pixels`), class IDs/names, confidence scores, and per-stage latency breakdown.

---

## 10. Known Limitations

- NEU-DET consists of 200 x 200 pre-cropped benchmark tiles; model performance on these tiles does not guarantee generalization to full-width industrial camera streams, varying illumination, or unseen defect morphologies.
- Dataset redistribution and pretrained `yolox_tiny.pth` checkpoint terms must be independently verified before publishing raw datasets or external weights.
- Synthetic corruption stress tests quantify sensitivity to controlled perturbations and are not substitutes for real factory validation data.
