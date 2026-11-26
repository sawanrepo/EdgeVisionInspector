"""Checkpoint persistence, metadata capture, and session-resume utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from edgevision.config import collect_runtime_environment, get_logger

logger = get_logger(__name__)


def save_checkpoint(
    output_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
    config: dict[str, Any],
    manifest_sha256: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a compact resumable checkpoint with full provenance metadata."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "best_score": float(best_score),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "manifest_sha256": manifest_sha256,
        "runtime_environment": collect_runtime_environment(),
        "extra_metadata": extra_metadata or {},
    }
    torch.save(payload, dest)
    return dest


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load model weights and optional optimizer/scheduler state for evaluation or resume."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch=%s, best_score=%s)", path, ckpt.get("epoch"), ckpt.get("best_score"))
    return ckpt


def write_run_metadata(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write structured JSON run metadata alongside checkpoints."""
    dest = Path(output_dir) / "run_summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return dest
