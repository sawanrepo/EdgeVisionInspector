"""Reproducible training runner with deterministic seeding, validation tracking, and resume."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from edgevision.config import collect_runtime_environment, get_logger, load_yaml, sha256_file
from edgevision.data.annotations import build_class_lookup, load_records_jsonl
from edgevision.data.dataset import build_dataloader
from edgevision.data.transforms import TrainingAugmentor
from edgevision.evaluation.metrics import compute_coco_map, compute_operating_point_metrics
from edgevision.models.build import build_detector
from edgevision.training.checkpoints import load_checkpoint, save_checkpoint, write_run_metadata
from edgevision.training.losses import compute_inverse_freq_weights

logger = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("lr", 5e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        momentum = float(train_cfg.get("momentum", 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    raise ValueError(f"Unsupported optimizer '{opt_name}'")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = int(train_cfg.get("warmup_epochs", 0))

    def lr_lambda(epoch_idx: int) -> float:
        if warmup > 0 and epoch_idx < warmup:
            return float(epoch_idx + 1) / float(warmup)
        progress = (epoch_idx - warmup) / float(max(1, epochs - warmup))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def collect_predictions(
    adapter: Any,
    dataloader: Any,
    class_names: list[str],
    device: torch.device,
    conf_threshold: float = 0.05,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    """Run PyTorch model + shared decode/NMS across a DataLoader and align with GT boxes in native coords."""
    adapter.eval()
    preds_by_image: dict[str, list[dict[str, Any]]] = {}
    gts_by_image: dict[str, list[dict[str, Any]]] = {}

    # Invert input-scaled GT boxes cleanly via ResizeParams or read directly from dataset records.
    dataset_records = {rec.image_id: rec for rec in dataloader.dataset.records}

    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            targets = batch["targets"]
            resize_list = [t["resize_params"] for t in targets]
            batch_dets = adapter.predict_batch(
                images=images,
                resize_params_list=resize_list,
                class_names=class_names,
                conf_threshold=conf_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections,
            )
            for tgt, dets in zip(targets, batch_dets, strict=False):
                image_id = tgt["image_id"]
                preds_by_image[image_id] = [
                    {
                        "class_id": d.class_id,
                        "score": d.score,
                        "bbox_xyxy": [d.xmin, d.ymin, d.xmax, d.ymax],
                    }
                    for d in dets
                ]
                rec = dataset_records[image_id]
                gts_by_image[image_id] = [
                    {
                        "class_id": b.class_id,
                        "bbox_xyxy": [b.xmin, b.ymin, b.xmax, b.ymax],
                    }
                    for b in rec.boxes
                ]
    return preds_by_image, gts_by_image


def train_detector(
    model_cfg_path: str | Path,
    dataset_cfg_path: str | Path,
    preprocess_cfg_path: str | Path,
    train_manifest: str | Path,
    val_manifest: str | Path,
    output_dir: str | Path,
    device_str: str | None = None,
    epochs_override: int | None = None,
    input_size_override: tuple[int, int] | None = None,
    resume_from: str | Path | None = None,
    use_class_weights: bool = False,
) -> dict[str, Any]:
    """Execute one reproducible training experiment and save best/last checkpoints and metrics."""
    model_cfg = load_yaml(model_cfg_path)
    dataset_cfg = load_yaml(dataset_cfg_path)
    specs, _ = build_class_lookup(dataset_cfg)
    class_names = [s.name for s in specs]

    mcfg = model_cfg["model"]
    tcfg = model_cfg["training"]
    post_cfg = model_cfg.get("postprocessing", {})

    arch = str(mcfg["architecture"])
    input_size = tuple(int(v) for v in (input_size_override or mcfg["input_size"]))
    seed = int(tcfg.get("seed", 42))
    seed_everything(seed)

    device = torch.device(
        device_str if device_str else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    epochs = int(epochs_override if epochs_override is not None else tcfg.get("epochs", 40))
    batch_size = int(tcfg.get("batch_size", 16))
    num_workers = int(tcfg.get("num_workers", 2))
    grad_clip = float(tcfg.get("grad_clip_norm", 10.0))
    patience = int(tcfg.get("early_stopping_patience", 10))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"

    augmentor = TrainingAugmentor(model_cfg.get("augmentation", {}), seed=seed)
    train_loader = build_dataloader(
        manifest_path=train_manifest,
        architecture=arch,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        preprocess_cfg=preprocess_cfg_path,
        augmentor=augmentor,
    )
    val_loader = build_dataloader(
        manifest_path=val_manifest,
        architecture=arch,
        input_size=input_size,
        batch_size=min(batch_size, 8),
        num_workers=num_workers,
        shuffle=False,
        preprocess_cfg=preprocess_cfg_path,
        augmentor=None,
    )

    class_weights: torch.Tensor | None = None
    if use_class_weights:
        train_records = load_records_jsonl(train_manifest)
        class_weights = compute_inverse_freq_weights(train_records, num_classes=len(class_names))

    adapter = build_detector(model_cfg, override_input_size=input_size).to(device)
    optimizer = _build_optimizer(adapter, tcfg)
    scheduler = _build_scheduler(optimizer, tcfg, epochs=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_map50 = -1.0
    resume_target = resume_from or tcfg.get("resume_from")
    if resume_target:
        ckpt = load_checkpoint(resume_target, adapter, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_map50 = float(ckpt.get("best_score", -1.0))

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_file = out_path / "training_log.jsonl"
    manifest_sha = sha256_file(train_manifest) if Path(train_manifest).exists() else None

    epochs_without_improvement = 0
    start_wall = time.perf_counter()
    best_epoch = start_epoch
    best_val_metrics: dict[str, Any] = {}

    logger.info(
        "Starting training: arch=%s input_size=%s device=%s epochs=%d batch_size=%d",
        arch,
        input_size,
        device,
        epochs,
        batch_size,
    )

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        adapter.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            targets = batch["targets"]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = adapter.compute_training_loss(images, targets, class_weights=class_weights)
                loss = loss_dict["total_loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss encountered at epoch {epoch}: {loss.item()}")

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps += 1

        scheduler.step()
        train_loss = running_loss / max(1, steps)

        preds_by_img, gts_by_img = collect_predictions(
            adapter=adapter,
            dataloader=val_loader,
            class_names=class_names,
            device=device,
            conf_threshold=float(post_cfg.get("score_threshold_pre_nms", 0.05)),
            nms_iou_threshold=float(post_cfg.get("nms_iou_threshold", 0.45)),
            max_detections=int(post_cfg.get("max_detections", 100)),
        )
        coco_metrics = compute_coco_map(preds_by_img, gts_by_img, num_classes=len(class_names))
        op_metrics = compute_operating_point_metrics(
            preds_by_img,
            gts_by_img,
            class_names=class_names,
            conf_threshold=0.25,
            iou_threshold=0.50,
        )
        val_map50 = float(coco_metrics["map_50"])
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_sec = time.perf_counter() - epoch_start
        max_mem_gb = (
            round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
            if device.type == "cuda"
            else 0.0
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_map50": round(val_map50, 5),
            "val_map50_95": round(float(coco_metrics["map_50_95"]), 5),
            "val_macro_recall_at_0_25": round(float(op_metrics["macro_recall"] or 0.0), 5),
            "val_macro_f1_at_0_25": round(float(op_metrics["macro_f1"] or 0.0), 5),
            "lr": lr_now,
            "epoch_duration_sec": round(epoch_sec, 3),
            "gpu_max_memory_gb": max_mem_gb,
        }
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        logger.info(
            "Epoch %d/%d | loss=%.4f | val_mAP50=%.4f | val_mAP=%.4f | lr=%.6f | %.1fs",
            epoch + 1,
            epochs,
            train_loss,
            val_map50,
            coco_metrics["map_50_95"],
            lr_now,
            epoch_sec,
        )

        save_checkpoint(
            out_path / "last.pt",
            adapter,
            optimizer,
            scheduler,
            epoch=epoch,
            best_score=max(best_map50, val_map50),
            config=model_cfg,
            manifest_sha256=manifest_sha,
        )

        if val_map50 > best_map50:
            best_map50 = val_map50
            best_epoch = epoch
            best_val_metrics = {**coco_metrics, "op_at_0_25": op_metrics}
            epochs_without_improvement = 0
            save_checkpoint(
                out_path / "best.pt",
                adapter,
                optimizer,
                scheduler,
                epoch=epoch,
                best_score=best_map50,
                config=model_cfg,
                manifest_sha256=manifest_sha,
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                logger.info("Early stopping triggered after %d epochs without validation mAP@0.5 gain.", patience)
                break

    total_sec = time.perf_counter() - start_wall
    summary = {
        "architecture": arch,
        "input_size": list(input_size),
        "resolution_group": mcfg.get("resolution_group", f"{input_size[0]}x{input_size[1]}"),
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_map50": round(best_map50, 5),
        "best_val_metrics": best_val_metrics,
        "total_duration_sec": round(total_sec, 3),
        "best_checkpoint": str(out_path / "best.pt"),
        "last_checkpoint": str(out_path / "last.pt"),
        "train_manifest_sha256": manifest_sha,
        "runtime_environment": collect_runtime_environment(),
    }
    write_run_metadata(out_path, summary)
    return summary
