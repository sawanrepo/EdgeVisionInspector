"""Duplicate-aware multi-label stratified splitting and evaluation-view manifest generation."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from edgevision.config import get_logger, sha256_dict
from edgevision.data.annotations import ImageRecord, save_records_jsonl

logger = get_logger(__name__)

PARTITIONS = ("train", "val", "test")


@dataclass
class SplitResult:
    """Partitioned records and audit metadata."""

    train: list[ImageRecord]
    val: list[ImageRecord]
    test: list[ImageRecord]
    manifest: dict[str, Any]


def _group_records(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        gid = rec.duplicate_group_id or f"single_{rec.image_id}"
        groups[gid].append(rec)
    for gid in groups:
        groups[gid].sort(key=lambda r: r.image_id)
    return dict(sorted(groups.items()))


def stratified_group_split(
    records: list[ImageRecord],
    num_classes: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """Split records into train/val/test while keeping duplicate groups intact and stratifying multi-label presence."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    if len(records) < 3:
        raise ValueError("At least 3 valid records are required for train/val/test splitting")

    rng = random.Random(seed)
    groups = _group_records(records)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_total = len(records)
    target_counts = {
        "train": train_ratio * n_total,
        "val": val_ratio * n_total,
        "test": test_ratio * n_total,
    }

    total_class_images = Counter()
    group_class_counts: dict[str, Counter[int]] = {}
    for gid, members in groups.items():
        counts: Counter[int] = Counter()
        for rec in members:
            for cid in rec.class_ids:
                counts[cid] += 1
                total_class_images[cid] += 1
        group_class_counts[gid] = counts

    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    desired_class = {
        part: {cid: total_class_images[cid] * ratios[part] for cid in range(num_classes)}
        for part in PARTITIONS
    }

    assigned_groups: dict[str, list[str]] = {part: [] for part in PARTITIONS}
    current_counts: dict[str, int] = {part: 0 for part in PARTITIONS}
    current_class_counts: dict[str, Counter[int]] = {part: Counter() for part in PARTITIONS}

    unassigned = set(group_ids)
    # Iterative rarest-class-first multi-label group assignment.
    while unassigned:
        remaining_class_totals: Counter[int] = Counter()
        for gid in unassigned:
            remaining_class_totals.update(group_class_counts[gid])

        positive_classes = [cid for cid, cnt in remaining_class_totals.items() if cnt > 0]
        if positive_classes:
            rarest_cid = min(positive_classes, key=lambda c: (remaining_class_totals[c], c))
            candidate_gids = [gid for gid in unassigned if group_class_counts[gid][rarest_cid] > 0]
        else:
            rarest_cid = None
            candidate_gids = sorted(unassigned)

        # Assign larger duplicate groups first within the rarest class so small splits do not overshoot.
        candidate_gids.sort(key=lambda g: (-len(groups[g]), g))
        for gid in candidate_gids:
            if gid not in unassigned:
                continue
            members = groups[gid]
            best_part = None
            best_score = -1e18
            for part in PARTITIONS:
                size_need = target_counts[part] - current_counts[part]
                class_need = (
                    desired_class[part][rarest_cid] - current_class_counts[part][rarest_cid]
                    if rarest_cid is not None
                    else 0.0
                )
                multi_need = sum(
                    max(0.0, desired_class[part][cid] - current_class_counts[part][cid])
                    for cid, cnt in group_class_counts[gid].items()
                    if cnt > 0
                )
                empty_bonus = 100.0 if current_counts[part] == 0 else 0.0
                score = empty_bonus + class_need * 10.0 + multi_need * 2.0 + size_need * 0.25
                if score > best_score:
                    best_score = score
                    best_part = part
            assert best_part is not None
            assigned_groups[best_part].append(gid)
            current_counts[best_part] += len(members)
            current_class_counts[best_part].update(group_class_counts[gid])
            unassigned.remove(gid)

    partitions: dict[str, list[ImageRecord]] = {part: [] for part in PARTITIONS}
    for part in PARTITIONS:
        for gid in assigned_groups[part]:
            partitions[part].extend(groups[gid])
        partitions[part].sort(key=lambda r: r.image_id)

    validate_split_integrity(partitions, num_classes=num_classes)

    manifest = build_split_manifest(partitions, num_classes=num_classes, seed=seed)
    return SplitResult(
        train=partitions["train"],
        val=partitions["val"],
        test=partitions["test"],
        manifest=manifest,
    )


def validate_split_integrity(
    partitions: dict[str, list[ImageRecord]],
    num_classes: int,
    require_all_classes: bool = True,
) -> None:
    """Verify zero image/duplicate-group/pixel-hash leakage and positive per-class support."""
    seen_images: dict[str, str] = {}
    seen_groups: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}

    for part, recs in partitions.items():
        if not recs:
            raise ValueError(f"Partition '{part}' is empty")
        class_support = Counter()
        for rec in recs:
            if rec.image_id in seen_images:
                raise ValueError(
                    f"Image leakage detected: '{rec.image_id}' appears in '{seen_images[rec.image_id]}' and '{part}'"
                )
            seen_images[rec.image_id] = part

            if rec.duplicate_group_id:
                prev = seen_groups.get(rec.duplicate_group_id)
                if prev is not None and prev != part:
                    raise ValueError(
                        f"Duplicate group leakage detected: '{rec.duplicate_group_id}' in '{prev}' and '{part}'"
                    )
                seen_groups[rec.duplicate_group_id] = part

            if rec.pixel_sha256:
                prev_hash = seen_hashes.get(rec.pixel_sha256)
                if prev_hash is not None and prev_hash != part:
                    raise ValueError(
                        f"Exact pixel hash leakage detected for '{rec.image_id}' between '{prev_hash}' and '{part}'"
                    )
                seen_hashes[rec.pixel_sha256] = part

            for box in rec.boxes:
                class_support[box.class_id] += 1

        if require_all_classes:
            missing = [cid for cid in range(num_classes) if class_support[cid] <= 0]
            if missing:
                raise ValueError(f"Partition '{part}' lacks ground-truth instances for classes: {missing}")


def build_split_manifest(
    partitions: dict[str, list[ImageRecord]],
    num_classes: int,
    seed: int,
) -> dict[str, Any]:
    """Build deterministic split manifest dictionary with per-partition summaries and checksum."""
    summary: dict[str, Any] = {}
    partition_ids: dict[str, list[str]] = {}
    for part in PARTITIONS:
        recs = partitions[part]
        partition_ids[part] = [r.image_id for r in recs]
        img_presence = Counter()
        inst_counts = Counter()
        for rec in recs:
            for cid in rec.class_ids:
                img_presence[cid] += 1
            for box in rec.boxes:
                inst_counts[box.class_id] += 1
        summary[part] = {
            "num_images": len(recs),
            "num_instances": sum(inst_counts.values()),
            "image_presence_by_class": {str(c): int(img_presence[c]) for c in range(num_classes)},
            "instance_counts_by_class": {str(c): int(inst_counts[c]) for c in range(num_classes)},
        }

    manifest: dict[str, Any] = {
        "seed": seed,
        "summary": summary,
        "partitions": partition_ids,
    }
    manifest["manifest_sha256"] = sha256_dict(manifest)
    return manifest


def select_natural_hard_subset(
    test_records: list[ImageRecord],
    max_images: int = 50,
) -> list[ImageRecord]:
    """Select up to max_images from test_records using model-independent difficulty heuristics."""
    scored: list[tuple[float, ImageRecord]] = []
    for rec in test_records:
        with Image.open(rec.image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        contrast = float(np.std(gray))
        img_area = float(max(1, rec.width * rec.height))
        min_rel_area = min((box.area / img_area for box in rec.boxes), default=1.0)
        num_boxes = len(rec.boxes)
        multi_class_bonus = 1.0 if len(rec.class_ids) > 1 else 0.0

        # Higher difficulty score for low contrast, tiny bounding boxes, and crowded scenes.
        difficulty = (
            (1.0 / max(contrast, 5.0)) * 50.0
            + (1.0 / max(min_rel_area, 1e-4)) * 0.01
            + num_boxes * 0.4
            + multi_class_bonus * 1.5
        )
        scored.append((difficulty, rec))

    scored.sort(key=lambda item: (-item[0], item[1].image_id))
    return [rec for _, rec in scored[:max_images]]


def _apply_corruption_to_image_and_boxes(
    gray: np.ndarray,
    boxes: list[dict[str, Any]],
    spec: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    name = spec["name"]
    h, w = gray.shape[:2]
    out_boxes = [dict(b) for b in boxes]

    if name == "brightness_shift":
        delta = float(spec.get("delta", -35.0))
        arr = np.clip(gray.astype(np.float32) + delta, 0.0, 255.0).astype(np.uint8)
        return arr, out_boxes

    if name == "contrast_reduce":
        alpha = float(spec.get("alpha", 0.65))
        mean_val = float(np.mean(gray))
        arr = np.clip((gray.astype(np.float32) - mean_val) * alpha + mean_val, 0.0, 255.0).astype(np.uint8)
        return arr, out_boxes

    if name == "gaussian_blur":
        radius = float(spec.get("radius", 1.5))
        pil = Image.fromarray(gray.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=radius))
        return np.asarray(pil, dtype=np.uint8), out_boxes

    if name == "gaussian_noise":
        sigma = float(spec.get("sigma", 15.0))
        noise = rng.normal(0.0, sigma, size=gray.shape)
        arr = np.clip(gray.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        return arr, out_boxes

    if name == "rotation_small":
        angle_deg = float(spec.get("angle_deg", 7.0))
        pil = Image.fromarray(gray.astype(np.uint8), mode="L").rotate(
            angle_deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=114
        )
        theta = np.deg2rad(-angle_deg)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        cx, cy = w / 2.0, h / 2.0
        rotated_boxes: list[dict[str, Any]] = []
        for b in out_boxes:
            corners = np.array(
                [
                    [b["xmin"], b["ymin"]],
                    [b["xmax"], b["ymin"]],
                    [b["xmax"], b["ymax"]],
                    [b["xmin"], b["ymax"]],
                ],
                dtype=np.float64,
            )
            shifted = corners - np.array([cx, cy])
            rot = np.column_stack(
                [
                    shifted[:, 0] * cos_t - shifted[:, 1] * sin_t + cx,
                    shifted[:, 0] * sin_t + shifted[:, 1] * cos_t + cy,
                ]
            )
            nxmin = float(np.clip(np.min(rot[:, 0]), 0.0, float(w)))
            nymin = float(np.clip(np.min(rot[:, 1]), 0.0, float(h)))
            nxmax = float(np.clip(np.max(rot[:, 0]), 0.0, float(w)))
            nymax = float(np.clip(np.max(rot[:, 1]), 0.0, float(h)))
            if nxmax > nxmin + 1.0 and nymax > nymin + 1.0:
                updated = dict(b)
                updated.update({"xmin": nxmin, "ymin": nymin, "xmax": nxmax, "ymax": nymax})
                rotated_boxes.append(updated)
        return np.asarray(pil, dtype=np.uint8), rotated_boxes

    raise ValueError(f"Unsupported corruption specification: {name}")


def build_corruption_suite_manifest(
    test_records: list[ImageRecord],
    corruptions: list[dict[str, Any]],
    output_images_dir: str | Path,
    seed: int = 42,
    max_source_images: int = 50,
) -> list[dict[str, Any]]:
    """Materialize deterministic corrupted test images and transformed boxes for stress testing."""
    rng = np.random.default_rng(seed)
    out_dir = Path(output_images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(test_records, key=lambda r: r.image_id)[:max_source_images]

    entries: list[dict[str, Any]] = []
    for rec in selected:
        with Image.open(rec.image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        raw_boxes = [
            {
                "class_id": b.class_id,
                "class_name": b.class_name,
                "xmin": b.xmin,
                "ymin": b.ymin,
                "xmax": b.xmax,
                "ymax": b.ymax,
            }
            for b in rec.boxes
        ]
        for spec in corruptions:
            corr_name = spec["name"]
            corr_img, corr_boxes = _apply_corruption_to_image_and_boxes(gray, raw_boxes, spec, rng)
            if not corr_boxes:
                continue
            corr_id = f"{rec.image_id}__{corr_name}"
            corr_path = out_dir / f"{corr_id}.png"
            Image.fromarray(corr_img, mode="L").save(corr_path)
            entries.append(
                {
                    "image_id": corr_id,
                    "source_image_id": rec.image_id,
                    "corruption_name": corr_name,
                    "corruption_params": spec,
                    "seed": seed,
                    "is_derived_stress_sample": True,
                    "image_path": str(corr_path),
                    "annotation_path": None,
                    "width": rec.width,
                    "height": rec.height,
                    "mode": "L",
                    "channels": 1,
                    "file_format": "PNG",
                    "boxes": corr_boxes,
                }
            )
    return entries


def save_split_manifests(
    split_result: SplitResult,
    manifests_dir: str | Path,
    dataset_cfg: dict[str, Any],
    corrupted_images_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Write split manifests, golden test manifest, natural hard subset, and corruption suite."""
    out_dir = Path(manifests_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits.json"
    splits_path.write_text(json.dumps(split_result.manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    train_path = save_records_jsonl(split_result.train, out_dir / "train.jsonl")
    val_path = save_records_jsonl(split_result.val, out_dir / "val.jsonl")
    golden_path = save_records_jsonl(split_result.test, out_dir / "golden_test.jsonl")

    views_cfg = dataset_cfg.get("evaluation_views", {})
    hard_subset = select_natural_hard_subset(
        split_result.test,
        max_images=int(views_cfg.get("natural_hard_max_images", 50)),
    )
    hard_path = save_records_jsonl(hard_subset, out_dir / "natural_hard_subset.jsonl")

    corr_dir = Path(corrupted_images_dir) if corrupted_images_dir else (out_dir / "corrupted_images")
    corr_entries = build_corruption_suite_manifest(
        test_records=split_result.test,
        corruptions=views_cfg.get("corruptions", []),
        output_images_dir=corr_dir,
        seed=int(views_cfg.get("corruption_seed", 42)),
        max_source_images=int(views_cfg.get("natural_hard_max_images", 50)),
    )
    corr_path = out_dir / "corruption_suite.jsonl"
    with corr_path.open("w", encoding="utf-8") as handle:
        for entry in corr_entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    checksum_payload = {
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "val_sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
        "golden_test_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "natural_hard_subset_sha256": hashlib.sha256(hard_path.read_bytes()).hexdigest(),
        "corruption_suite_sha256": hashlib.sha256(corr_path.read_bytes()).hexdigest(),
    }
    checksums_path = out_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksum_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    logger.info(
        "Saved split manifests to %s (train=%d, val=%d, golden_test=%d, natural_hard=%d, corrupted=%d)",
        out_dir,
        len(split_result.train),
        len(split_result.val),
        len(split_result.test),
        len(hard_subset),
        len(corr_entries),
    )
    return {
        "splits": splits_path,
        "train": train_path,
        "val": val_path,
        "golden_test": golden_path,
        "natural_hard_subset": hard_path,
        "corruption_suite": corr_path,
        "checksums": checksums_path,
    }
