"""Tests for detection metric correctness, one-to-one matching, bootstrap CI, and recall floors."""

from __future__ import annotations

import pytest

from edgevision.evaluation.metrics import (
    compute_coco_map,
    compute_operating_point_metrics,
    compute_validation_recall_floors,
    image_cluster_bootstrap_recall,
)


def test_hand_verified_greedy_matching_and_macro_metrics() -> None:
    class_names = ["c0", "c1"]
    gts = {
        "img1": [
            {"class_id": 0, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 1, "bbox_xyxy": [100.0, 100.0, 140.0, 140.0]},
        ],
        "img2": [
            {"class_id": 0, "bbox_xyxy": [20.0, 20.0, 60.0, 60.0]},
            {"class_id": 1, "bbox_xyxy": [120.0, 120.0, 160.0, 160.0]},
        ],
    }
    preds = {
        "img1": [
            # Duplicate detection on img1 class 0 -> 1 TP, 1 FP
            {"class_id": 0, "score": 0.90, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]},
            {"class_id": 0, "score": 0.80, "bbox_xyxy": [12.0, 12.0, 52.0, 52.0]},
            # Wrong location on class 1 -> 1 FP, 1 FN
            {"class_id": 1, "score": 0.70, "bbox_xyxy": [0.0, 0.0, 20.0, 20.0]},
        ],
        "img2": [
            # Exact match on class 0, missed class 1
            {"class_id": 0, "score": 0.85, "bbox_xyxy": [20.0, 20.0, 60.0, 60.0]},
        ],
    }

    res = compute_operating_point_metrics(preds, gts, class_names, conf_threshold=0.5, iou_threshold=0.5)
    c0 = res["per_class"]["c0"]
    c1 = res["per_class"]["c1"]

    assert (c0["tp"], c0["fp"], c0["fn"], c0["gt_support"], c0["image_support"]) == (2, 1, 0, 2, 2)
    assert (c1["tp"], c1["fp"], c1["fn"], c1["gt_support"], c1["image_support"]) == (0, 1, 2, 2, 2)
    assert c0["precision"] == pytest.approx(2 / 3, abs=1e-5)
    assert c0["recall"] == pytest.approx(1.0, abs=1e-5)
    assert c1["precision"] == pytest.approx(0.0, abs=1e-5)
    assert c1["recall"] == pytest.approx(0.0, abs=1e-5)
    assert res["macro_precision"] == pytest.approx(1 / 3, abs=1e-5)
    assert res["macro_recall"] == pytest.approx(0.5, abs=1e-5)
    assert res["fppi"] == pytest.approx(1.0, abs=1e-5)

    coco = compute_coco_map(preds, gts, num_classes=2)
    assert 0.0 <= coco["map_50"] <= 1.0
    assert coco["map_50"] >= coco["map_50_95"]


def test_cluster_bootstrap_and_recall_floors() -> None:
    class_names = ["c0", "c1"]
    gts = {
        f"img_{i}": [
            {"class_id": 0, "bbox_xyxy": [10.0, 10.0, 40.0, 40.0]},
            {"class_id": 1, "bbox_xyxy": [60.0, 60.0, 90.0, 90.0]},
        ]
        for i in range(10)
    }
    preds = {
        f"img_{i}": (
            [
                {"class_id": 0, "score": 0.9, "bbox_xyxy": [10.0, 10.0, 40.0, 40.0]},
                {"class_id": 1, "score": 0.9, "bbox_xyxy": [60.0, 60.0, 90.0, 90.0]},
            ]
            if i < 8
            else [{"class_id": 0, "score": 0.9, "bbox_xyxy": [10.0, 10.0, 40.0, 40.0]}]
        )
        for i in range(10)
    }

    op = compute_operating_point_metrics(preds, gts, class_names, conf_threshold=0.5)
    boot = image_cluster_bootstrap_recall(preds, gts, class_names, conf_threshold=0.5, replicates=200, seed=42)
    floors = compute_validation_recall_floors(op, boot)

    assert boot["per_class"]["c0"]["point_recall"] == pytest.approx(1.0)
    assert boot["per_class"]["c1"]["point_recall"] == pytest.approx(0.8)
    assert boot["per_class"]["c1"]["ci95_low"] <= 0.8 <= boot["per_class"]["c1"]["ci95_high"]
    assert floors["c0"]["floor"] == pytest.approx(0.97)
    assert 0.0 <= floors["c1"]["floor"] <= 0.77
