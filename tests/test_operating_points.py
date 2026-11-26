"""Tests for validation operating-point selection, candidate selection rule, regression gate, and golden guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgevision.evaluation.evaluate import (
    check_validation_regression,
    create_validation_baseline,
    evaluate_golden_test_once,
    select_deployment_candidate,
    select_operating_points_from_validation,
)
from edgevision.evaluation.metrics import compute_coco_map


def _synthetic_val_data():
    class_names = ["c0", "c1"]
    gts = {
        f"val_{i}": [
            {"class_id": 0, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "bbox_xyxy": [80.0, 80.0, 120.0, 120.0]},
        ]
        for i in range(10)
    }
    preds = {}
    for i in range(10):
        # 9/10 objects detected at 0.55; 10/10 at 0.30; noisy FP at 0.20
        score_c1 = 0.55 if i < 9 else 0.30
        preds[f"val_{i}"] = [
            {"class_id": 0, "score": 0.85, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "score": score_c1, "bbox_xyxy": [80.0, 80.0, 120.0, 120.0]},
            {"class_id": 0, "score": 0.20, "bbox_xyxy": [150.0, 150.0, 180.0, 180.0]},
        ]
    return class_names, gts, preds


def test_operating_point_selection_and_infeasible_recall(tmp_path: Path) -> None:
    class_names, gts, preds = _synthetic_val_data()
    op = select_operating_points_from_validation(preds, gts, class_names)

    assert op["selection_split"] == "val_only"
    assert op["recall_target_operating_point"]["feasible"] is True
    # Highest threshold achieving >= 0.90 macro recall (0.95 recall at <=0.55, 1.0 recall at <=0.30).
    assert op["recall_target_operating_point"]["threshold"] == pytest.approx(0.55, abs=0.01)
    assert op["quality_operating_point"]["threshold"] >= 0.30

    # When one class is completely missed, recall >= 0.90 must be marked infeasible.
    preds_low_recall = {
        k: [p for p in v if p["class_id"] == 0] for k, v in preds.items()
    }
    op_inf = select_operating_points_from_validation(preds_low_recall, gts, class_names)
    assert op_inf["recall_target_operating_point"]["feasible"] is False
    assert op_inf["recall_target_operating_point"]["threshold"] is None

    # Create validation baseline and check regression pass/fail
    val_manifest = tmp_path / "val.jsonl"
    val_manifest.write_text("{}", encoding="utf-8")
    baseline = create_validation_baseline(
        preds,
        gts,
        op,
        class_names,
        val_manifest_path=val_manifest,
        model_info={"architecture": "yolox_tiny", "input_size": [320, 320]},
        bootstrap_replicates=100,
    )
    coco = compute_coco_map(preds, gts, num_classes=2)
    pass_check = check_validation_regression(
        coco, op["recall_target_operating_point"]["validation_metrics"], baseline, candidate_p95_cpu_latency_ms=45.0
    )
    assert pass_check["passed"] is True

    degraded_coco = dict(coco, map_50=coco["map_50"] - 0.05)
    fail_check = check_validation_regression(
        degraded_coco, op["recall_target_operating_point"]["validation_metrics"], baseline, candidate_p95_cpu_latency_ms=125.0
    )
    assert fail_check["passed"] is False
    assert len(fail_check["failures"]) >= 2


def test_predeclared_deployment_selection_rule() -> None:
    candidates = [
        {
            "architecture": "yolox_tiny",
            "input_size": "320x320",
            "resolution_group": "matched_320",
            "deployment_eligible": True,
            "val_map50": 0.78,
            "val_macro_recall_at_op": 0.92,
            "recall_target_feasible": True,
            "p95_latency_recall_op_ms": 42.0,
            "onnx_size_mb": 19.2,
        },
        {
            "architecture": "ssdlite320_mobilenet_v3_large",
            "input_size": "320x320",
            "resolution_group": "matched_320",
            "deployment_eligible": True,
            "val_map50": 0.74,
            "val_macro_recall_at_op": 0.91,
            "recall_target_feasible": True,
            "p95_latency_recall_op_ms": 28.0,
            "onnx_size_mb": 9.5,
        },
        {
            "architecture": "faster_rcnn_resnet50_fpn",
            "input_size": "320x320",
            "resolution_group": "matched_320",
            "deployment_eligible": True,
            "val_map50": 0.85,
            "val_macro_recall_at_op": 0.95,
            "recall_target_feasible": True,
            "p95_latency_recall_op_ms": 380.0,
            "onnx_size_mb": 160.0,
        },
    ]
    decision = select_deployment_candidate(candidates, latency_budget_ms=100.0)
    assert decision["selected_architecture"] == "yolox_tiny"
    assert decision["selection_rule_branch"] == "tier_1_within_latency_and_recall_target"


def test_single_golden_evaluation_guard(tmp_path: Path) -> None:
    class_names, gts, preds = _synthetic_val_data()
    op = select_operating_points_from_validation(preds, gts, class_names)
    golden_manifest = tmp_path / "golden_test.jsonl"
    golden_manifest.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "final_test_report.json"

    evaluate_golden_test_once(
        preds, gts, op, class_names, golden_manifest, report_path, bootstrap_replicates=50
    )
    assert report_path.exists()
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_golden_test_once(
            preds, gts, op, class_names, golden_manifest, report_path, bootstrap_replicates=50
        )
