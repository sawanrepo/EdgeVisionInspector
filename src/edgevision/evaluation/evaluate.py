"""Operating-point selection, validation baseline freeze, candidate selection, regression gate, and single golden evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from edgevision.config import get_logger, sha256_dict, sha256_file
from edgevision.evaluation.metrics import (
    compute_coco_map,
    compute_operating_point_metrics,
    compute_validation_recall_floors,
    image_cluster_bootstrap_recall,
)

logger = get_logger(__name__)


def select_operating_points_from_validation(
    val_preds_by_image: dict[str, list[dict[str, Any]]],
    val_gts_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    iou_threshold: float = 0.50,
    target_macro_recall: float = 0.90,
    min_conf: float = 0.05,
    max_conf: float = 0.95,
    num_steps: int = 91,
    artifact_label: str = "pytorch_fp32",
) -> dict[str, Any]:
    """Select and freeze quality and recall-target thresholds strictly from validation predictions."""
    thresholds = [round(float(t), 4) for t in np.linspace(min_conf, max_conf, num_steps)]
    sweep: list[dict[str, Any]] = []

    for t in thresholds:
        metrics = compute_operating_point_metrics(
            preds_by_image=val_preds_by_image,
            gts_by_image=val_gts_by_image,
            class_names=class_names,
            conf_threshold=t,
            iou_threshold=iou_threshold,
        )
        if metrics["undefined_classes"]:
            raise ValueError(
                f"Validation set lacks support for classes {metrics['undefined_classes']}; "
                "cannot select operating points."
            )
        sweep.append(
            {
                "threshold": t,
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "fppi": float(metrics["fppi"]),
            }
        )

    # 1. Quality operating point: max macro-F1, ties by higher macro-recall, macro-precision, then threshold.
    best_quality = max(
        sweep,
        key=lambda row: (row["macro_f1"], row["macro_recall"], row["macro_precision"], row["threshold"]),
    )
    quality_metrics = compute_operating_point_metrics(
        val_preds_by_image, val_gts_by_image, class_names, best_quality["threshold"], iou_threshold
    )

    # 2. Recall-target operating point: highest threshold with validation macro recall >= 0.90.
    feasible_rows = [row for row in sweep if row["macro_recall"] >= target_macro_recall]
    if feasible_rows:
        best_recall_row = max(
            feasible_rows,
            key=lambda row: (row["threshold"], row["macro_precision"]),
        )
        recall_metrics = compute_operating_point_metrics(
            val_preds_by_image, val_gts_by_image, class_names, best_recall_row["threshold"], iou_threshold
        )
        recall_op = {
            "feasible": True,
            "threshold": best_recall_row["threshold"],
            "target_macro_recall": target_macro_recall,
            "validation_metrics": recall_metrics,
        }
    else:
        max_achieved = max(row["macro_recall"] for row in sweep) if sweep else 0.0
        recall_op = {
            "feasible": False,
            "threshold": None,
            "target_macro_recall": target_macro_recall,
            "max_validation_macro_recall": round(float(max_achieved), 6),
            "validation_metrics": None,
        }

    payload: dict[str, Any] = {
        "artifact_label": artifact_label,
        "selection_split": "val_only",
        "iou_threshold": iou_threshold,
        "quality_operating_point": {
            "rule": "max_validation_macro_f1",
            "threshold": best_quality["threshold"],
            "validation_metrics": quality_metrics,
        },
        "recall_target_operating_point": recall_op,
        "sweep": sweep,
    }
    payload["config_hash"] = sha256_dict(
        {
            "artifact_label": artifact_label,
            "iou_threshold": iou_threshold,
            "target_macro_recall": target_macro_recall,
            "thresholds": thresholds,
            "quality_threshold": best_quality["threshold"],
            "recall_threshold": recall_op["threshold"],
        }
    )
    return payload


def create_validation_baseline(
    val_preds_by_image: dict[str, list[dict[str, Any]]],
    val_gts_by_image: dict[str, list[dict[str, Any]]],
    operating_points: dict[str, Any],
    class_names: list[str],
    val_manifest_path: str | Path,
    model_info: dict[str, Any],
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    p95_cpu_latency_ms: float | None = None,
) -> dict[str, Any]:
    """Freeze validation-only quality baseline and per-class regression floors before golden evaluation."""
    coco_val = compute_coco_map(val_preds_by_image, val_gts_by_image, num_classes=len(class_names))
    recall_op = operating_points["recall_target_operating_point"]
    if recall_op["feasible"]:
        floor_op_name = "recall_target_operating_point"
        floor_threshold = float(recall_op["threshold"])
        floor_op_metrics = recall_op["validation_metrics"]
    else:
        floor_op_name = "quality_operating_point"
        floor_threshold = float(operating_points["quality_operating_point"]["threshold"])
        floor_op_metrics = operating_points["quality_operating_point"]["validation_metrics"]

    bootstrap = image_cluster_bootstrap_recall(
        preds_by_image=val_preds_by_image,
        gts_by_image=val_gts_by_image,
        class_names=class_names,
        conf_threshold=floor_threshold,
        iou_threshold=float(operating_points.get("iou_threshold", 0.50)),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    class_floors = compute_validation_recall_floors(floor_op_metrics, bootstrap)
    val_sha = sha256_file(val_manifest_path) if Path(val_manifest_path).exists() else None

    return {
        "baseline_split": "val_only",
        "model_info": model_info,
        "val_manifest_sha256": val_sha,
        "operating_point_used_for_floors": floor_op_name,
        "recall_target_feasible": bool(recall_op["feasible"]),
        "threshold": floor_threshold,
        "operating_points_hash": operating_points["config_hash"],
        "validation_coco_metrics": coco_val,
        "validation_operating_point_metrics": floor_op_metrics,
        "bootstrap": bootstrap,
        "class_recall_floors": class_floors,
        "measured_p95_cpu_latency_ms": p95_cpu_latency_ms,
    }


def check_validation_regression(
    candidate_val_coco: dict[str, Any],
    candidate_op_metrics: dict[str, Any],
    baseline: dict[str, Any],
    candidate_p95_cpu_latency_ms: float | None = None,
    max_map50_drop: float = 0.01,
    max_p95_latency_ms: float = 100.0,
) -> dict[str, Any]:
    """Compare a candidate's validation-only metrics against eval/validation_baseline.json."""
    failures: list[str] = []
    base_map50 = float(baseline["validation_coco_metrics"]["map_50"])
    cand_map50 = float(candidate_val_coco["map_50"])
    map50_delta = cand_map50 - base_map50
    if map50_delta < -max_map50_drop:
        failures.append(
            f"Validation mAP@0.5 degraded by {-map50_delta:.4f} (threshold: {max_map50_drop:.4f})"
        )

    floor_checks: dict[str, dict[str, Any]] = {}
    for cname, floor_info in baseline["class_recall_floors"].items():
        floor_val = float(floor_info["floor"])
        cand_rec = candidate_op_metrics["per_class"][cname]["recall"]
        passed = cand_rec is not None and float(cand_rec) >= floor_val
        floor_checks[cname] = {
            "candidate_recall": cand_rec,
            "committed_floor": floor_val,
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"Class '{cname}' validation recall {cand_rec} fell below committed floor {floor_val:.4f}"
            )

    latency_passed = True
    if candidate_p95_cpu_latency_ms is not None and candidate_p95_cpu_latency_ms > max_p95_latency_ms:
        latency_passed = False
        failures.append(
            f"CPU P95 latency {candidate_p95_cpu_latency_ms:.2f} ms exceeds {max_p95_latency_ms:.2f} ms target"
        )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "baseline_val_map50": base_map50,
        "candidate_val_map50": cand_map50,
        "map50_delta": round(map50_delta, 6),
        "floor_checks": floor_checks,
        "latency_passed": latency_passed,
    }


def select_deployment_candidate(
    candidates: list[dict[str, Any]],
    latency_budget_ms: float = 100.0,
    target_macro_recall: float = 0.90,
) -> dict[str, Any]:
    """Apply the predeclared validation-only selection rule on matched-320 deployment-eligible candidates."""
    matched_eligible = [
        c
        for c in candidates
        if c.get("resolution_group") == "matched_320" and bool(c.get("deployment_eligible", False))
    ]
    if not matched_eligible:
        raise ValueError("No deployment-eligible candidates in matched_320 resolution group")

    within_latency = [
        c
        for c in matched_eligible
        if c.get("recall_target_feasible", False)
        and c.get("p95_latency_recall_op_ms") is not None
        and float(c["p95_latency_recall_op_ms"]) <= latency_budget_ms
    ]

    if within_latency:
        selected = max(
            within_latency,
            key=lambda c: (
                float(c["val_map50"]),
                float(c.get("val_macro_recall_at_op") or 0.0),
                -float(c.get("onnx_size_mb") or 1e9),
            ),
        )
        rule_branch = "tier_1_within_latency_and_recall_target"
        latency_target_met = True
        recall_target_feasible = True
    else:
        recall_feasible = [c for c in matched_eligible if c.get("recall_target_feasible", False)]
        if recall_feasible:
            selected = min(
                recall_feasible,
                key=lambda c: (
                    float(c.get("p95_latency_recall_op_ms") or 1e9),
                    -float(c["val_map50"]),
                ),
            )
            rule_branch = "tier_2_recall_feasible_latency_missed_min_latency"
            latency_target_met = False
            recall_target_feasible = True
        else:
            selected = max(
                matched_eligible,
                key=lambda c: (
                    float(c.get("val_macro_recall_at_quality_op") or 0.0),
                    -float(c.get("p95_latency_quality_op_ms") or 1e9),
                    float(c["val_map50"]),
                ),
            )
            rule_branch = "tier_3_recall_target_infeasible_max_recall"
            latency_target_met = False
            recall_target_feasible = False

    return {
        "selection_rule_branch": rule_branch,
        "selected_architecture": selected["architecture"],
        "selected_input_size": selected["input_size"],
        "latency_target_met": latency_target_met,
        "recall_target_feasible": recall_target_feasible,
        "selected_candidate": selected,
    }


def write_model_comparison_csv(candidates: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write reports/model_comparison.csv with resolution groups and eligibility metadata."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "architecture",
        "input_size",
        "resolution_group",
        "deployment_eligible",
        "onnx_export_status",
        "onnx_parity_passed",
        "val_map50",
        "val_map50_95",
        "recall_target_feasible",
        "recall_target_threshold",
        "quality_threshold",
        "val_macro_recall_at_op",
        "val_macro_f1_at_quality_op",
        "p95_latency_recall_op_ms",
        "p95_latency_quality_op_ms",
        "onnx_size_mb",
        "selected_for_deployment",
        "selection_note",
    ]
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return dest


def evaluate_golden_test_once(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    operating_points: dict[str, Any],
    class_names: list[str],
    golden_manifest_path: str | Path,
    output_report_path: str | Path,
    uncertainty_report_path: str | Path | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 42,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Run the single controlled golden-test evaluation using frozen validation thresholds."""
    dest = Path(output_report_path)
    if dest.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Golden test report already exists at {dest}. "
            "The clean golden test set must be evaluated once after artifacts are frozen."
        )

    coco_test = compute_coco_map(preds_by_image, gts_by_image, num_classes=len(class_names))
    quality_thr = float(operating_points["quality_operating_point"]["threshold"])
    iou_thr = float(operating_points.get("iou_threshold", 0.50))

    quality_test = compute_operating_point_metrics(
        preds_by_image, gts_by_image, class_names, conf_threshold=quality_thr, iou_threshold=iou_thr
    )
    quality_boot = image_cluster_bootstrap_recall(
        preds_by_image,
        gts_by_image,
        class_names,
        conf_threshold=quality_thr,
        iou_threshold=iou_thr,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )

    recall_op = operating_points["recall_target_operating_point"]
    if recall_op["feasible"]:
        recall_thr = float(recall_op["threshold"])
        recall_test = compute_operating_point_metrics(
            preds_by_image, gts_by_image, class_names, conf_threshold=recall_thr, iou_threshold=iou_thr
        )
        recall_boot = image_cluster_bootstrap_recall(
            preds_by_image,
            gts_by_image,
            class_names,
            conf_threshold=recall_thr,
            iou_threshold=iou_thr,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        achieved = (recall_test["macro_recall"] or 0.0) >= float(recall_op["target_macro_recall"])
        recall_result: dict[str, Any] = {
            "feasible_on_validation": True,
            "frozen_threshold": recall_thr,
            "target_achieved_on_test": bool(achieved),
            "test_metrics": recall_test,
            "bootstrap_ci": recall_boot,
        }
    else:
        recall_result = {
            "feasible_on_validation": False,
            "frozen_threshold": None,
            "target_achieved_on_test": False,
            "test_metrics": None,
            "bootstrap_ci": None,
        }

    report = {
        "evaluation_split": "golden_test",
        "golden_manifest_sha256": sha256_file(golden_manifest_path) if Path(golden_manifest_path).exists() else None,
        "operating_points_hash": operating_points["config_hash"],
        "artifact_label": operating_points.get("artifact_label", "unknown"),
        "coco_metrics": coco_test,
        "quality_operating_point": {
            "frozen_threshold": quality_thr,
            "test_metrics": quality_test,
            "bootstrap_ci": quality_boot,
        },
        "recall_target_operating_point": recall_result,
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    if uncertainty_report_path is not None:
        u_dest = Path(uncertainty_report_path)
        u_dest.parent.mkdir(parents=True, exist_ok=True)
        u_payload = {
            "artifact_label": report["artifact_label"],
            "quality_operating_point_ci": quality_boot,
            "recall_target_operating_point_ci": recall_result["bootstrap_ci"],
        }
        u_dest.write_text(json.dumps(u_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    return report
