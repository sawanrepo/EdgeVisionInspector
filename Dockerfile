FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EDGEVISION_ROOT=/app \
    EDGEVISION_MODEL_PATH=/app/artifacts/model_fp32.onnx \
    EDGEVISION_ARCH=yolox_tiny \
    EDGEVISION_INPUT_HEIGHT=320 \
    EDGEVISION_INPUT_WIDTH=320 \
    EDGEVISION_DATASET_CFG=/app/configs/dataset.yaml \
    EDGEVISION_PREPROCESS_CFG=/app/configs/preprocess.yaml \
    EDGEVISION_INTRA_THREADS=4 \
    EDGEVISION_INTER_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY artifacts/ ./artifacts/

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-deps -e .

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "edgevision.inference.service:app", "--host", "0.0.0.0", "--port", "8000"]
