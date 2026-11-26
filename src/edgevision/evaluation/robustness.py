"""Evaluation of natural hard subset and controlled corruption stress tests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from edgevision.evaluation.metrics import compute_coco_map, compute_operating_point_metrics


def evaluate_natural_hard_subset(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate predictions on the pre-frozen natural hard test subset."""
    coco = compute_coco_map(preds_by_image, gts_by_image, num_classes=len(class_names))
    op = compute_operating_point_metrics(
        preds_by_image, gts_by_image, class_names, conf_threshold=conf_threshold, iou_threshold=iou_threshold
    )
    return {
        "view": "natural_hard_subset",
        "num_images": len(gts_by_image),
        "coco_metrics": coco,
        "operating_point_metrics": op,
    }


def evaluate_corruption_suite(
    corruption_entries: list[dict[str, Any]],
    preds_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate derived corruption stress tests grouped by corruption type and parameters."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in corruption_entries:
        by_type[str(entry["corruption_name"])].append(entry)

    per_corruption: dict[str, Any] = {}
    for corr_name, items in sorted(by_type.items()):
        sub_gts = {item["image_id"]: item["boxes"] for item in items}
        sub_preds = {item["image_id"]: preds_by_image.get(item["image_id"], []) for item in items}
        coco = compute_coco_map(sub_preds, sub_gts, num_classes=len(class_names))
        op = compute_operating_point_metrics(
            sub_preds, sub_gts, class_names, conf_threshold=conf_threshold, iou_threshold=iou_threshold
        )
        per_corruption[corr_name] = {
            "num_derived_samples": len(items),
            "corruption_params": items[0].get("corruption_params", {}),
            "seed": items[0].get("seed"),
            "coco_metrics": coco,
            "operating_point_metrics": op,
        }

    return {
        "view": "controlled_corruption_suite",
        "note": "Derived stress-test samples; not counted as independent test images or real factory proof.",
        "per_corruption": per_corruption,
    }


def save_robustness_report(
    hard_report: dict[str, Any],
    corruption_report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Persist combined robustness report JSON."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "natural_hard_subset": hard_report,
        "controlled_corruption_suite": corruption_report,
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return dest
