"""Configuration loading, hashing, logging, and runtime path resolution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("configs")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a structured logger configured once per logger hierarchy."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {resolved}")
    return data


def sha256_file(path: str | Path) -> str:
    """Compute hex SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dict(payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest of a JSON-serializable mapping."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Resolve the repository root from environment override or current directory."""
    env_root = os.environ.get("EDGEVISION_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
            return candidate
    return current


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve relative paths against base_dir or project root while preserving absolutes."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(base_dir) if base_dir is not None else resolve_project_root()
    return (root / candidate).resolve()


def is_kaggle_runtime() -> bool:
    """Return True when running inside a Kaggle kernel environment."""
    return Path("/kaggle/working").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_git_commit(repo_root: str | Path | None = None) -> str:
    """Return the current Git commit hash or 'unknown' if unavailable."""
    cwd = str(repo_root or resolve_project_root())
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_runtime_environment() -> dict[str, Any]:
    """Capture Python, OS, CPU, RAM, GPU, and package versions for reproducibility."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "is_kaggle": is_kaggle_runtime(),
        "git_commit": get_git_commit(),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 3)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    except Exception:
        pass

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            devices = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 3),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
            info["cuda_devices"] = devices
    except Exception:
        pass

    for mod_name in ("torchvision", "onnx", "onnxruntime", "numpy", "PIL", "fastapi"):
        try:
            mod = __import__(mod_name)
            info[f"{mod_name}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass

    try:
        import onnxruntime as ort

        info["ort_available_providers"] = ort.get_available_providers()
    except Exception:
        pass

    return info
