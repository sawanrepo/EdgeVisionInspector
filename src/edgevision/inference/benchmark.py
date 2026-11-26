"""CPU-only ONNX Runtime benchmarking with pinned thread settings, peak RAM, and hardware audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil

from edgevision.config import collect_runtime_environment, get_logger, load_yaml
from edgevision.inference.predict import OnnxPredictor

logger = get_logger(__name__)


def _summarize_latencies(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": round(mean_ms, 4),
        "std_ms": round(float(np.std(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "throughput_fps": round(1000.0 / max(mean_ms, 1e-6), 3),
    }


def _read_cpu_info_linux() -> dict[str, Any]:
    info: dict[str, Any] = {}
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        text = cpuinfo_path.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, val = [part.strip() for part in line.split(":", 1)]
            if key == "model name" and "model_name" not in info:
                info["model_name"] = val
            elif key == "flags" and "flags" not in info:
                info["flags"] = val.split()
    return info


def run_cpu_benchmark(
    onnx_path: str | Path,
    architecture: str,
    input_size: tuple[int, int] | list[int],
    image_sources: list[bytes | str | Path],
    benchmark_cfg: dict[str, Any] | str | Path = "configs/benchmark.yaml",
    dataset_cfg: dict[str, Any] | str | Path = "configs/dataset.yaml",
    preprocess_cfg: dict[str, Any] | str | Path = "configs/preprocess.yaml",
    conf_threshold: float = 0.25,
    operating_point_label: str = "default",
    output_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark batch-1 CPU inference across warmup + timed runs and report latency boundaries."""
    cfg = load_yaml(benchmark_cfg) if isinstance(benchmark_cfg, (str, Path)) else benchmark_cfg
    bcfg = cfg.get("benchmark", cfg)

    if not image_sources:
        raise ValueError("image_sources must be non-empty for CPU benchmarking")

    intra = int(bcfg.get("intra_op_num_threads", 4))
    inter = int(bcfg.get("inter_op_num_threads", 1))
    warmup = int(bcfg.get("warmup_iterations", 50))
    timed = int(bcfg.get("timed_iterations", 1000))

    raw_inputs: list[bytes] = []
    for src in image_sources:
        if isinstance(src, (bytes, bytearray)):
            raw_inputs.append(bytes(src))
        else:
            raw_inputs.append(Path(src).read_bytes())

    predictor = OnnxPredictor(
        onnx_path=onnx_path,
        architecture=architecture,
        input_size=input_size,
        dataset_cfg=dataset_cfg,
        preprocess_cfg=preprocess_cfg,
        intra_op_num_threads=intra,
        inter_op_num_threads=inter,
        conf_threshold=conf_threshold,
    )

    for idx in range(warmup):
        predictor.predict(raw_inputs[idx % len(raw_inputs)], conf_threshold=conf_threshold)

    model_times: list[float] = []
    pipeline_times: list[float] = []
    e2e_times: list[float] = []

    proc = psutil.Process(os.getpid())
    peak_rss_bytes = proc.memory_info().rss

    for idx in range(timed):
        res = predictor.predict(raw_inputs[idx % len(raw_inputs)], conf_threshold=conf_threshold)
        model_times.append(res.latency.model_latency_ms)
        pipeline_times.append(res.latency.inference_pipeline_latency_ms)
        e2e_times.append(res.latency.end_to_end_in_process_latency_ms)
        if idx % 50 == 0:
            peak_rss_bytes = max(peak_rss_bytes, proc.memory_info().rss)

    peak_rss_bytes = max(peak_rss_bytes, proc.memory_info().rss)
    e2e_stats = _summarize_latencies(e2e_times)
    target_ms = float(bcfg.get("target_p95_end_to_end_ms", 100.0))

    env_info = collect_runtime_environment()
    env_info.update(_read_cpu_info_linux())

    report: dict[str, Any] = {
        "architecture": architecture,
        "input_size": [int(input_size[0]), int(input_size[1])],
        "onnx_path": str(onnx_path),
        "onnx_size_mb": round(Path(onnx_path).stat().st_size / (1024 * 1024), 4),
        "operating_point_label": operating_point_label,
        "conf_threshold": float(conf_threshold),
        "batch_size": 1,
        "execution_provider": "CPUExecutionProvider",
        "ort_version": ort.__version__,
        "intra_op_num_threads": intra,
        "inter_op_num_threads": inter,
        "warmup_iterations": warmup,
        "timed_iterations": timed,
        "num_unique_images": len(raw_inputs),
        "peak_process_rss_mb": round(peak_rss_bytes / (1024 * 1024), 3),
        "target_p95_end_to_end_ms": target_ms,
        "target_met": bool(e2e_stats["p95_ms"] <= target_ms),
        "latency_boundaries": {
            "model_forward_only": _summarize_latencies(model_times),
            "inference_pipeline": _summarize_latencies(pipeline_times),
            "end_to_end_in_process": e2e_stats,
        },
        "environment": env_info,
    }

    if output_report_path is not None:
        dest = Path(output_report_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        logger.info(
            "CPU benchmark saved to %s (P95 end-to-end=%.2f ms, target=%s)",
            dest,
            e2e_stats["p95_ms"],
            report["target_met"],
        )
    return report
