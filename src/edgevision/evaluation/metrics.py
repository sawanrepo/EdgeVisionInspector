"""COCO mAP, class-aware greedy matching, macro metrics, image-cluster bootstrap, and recall floors."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between [N, 4] and [M, 4] xyxy boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    ix1 = np.maximum(a[:, 0:1], b[:, 0:1].T)
    iy1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    ix2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    iy2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = np.maximum(0.0, a[:, 2:3] - a[:, 0:1]) * np.maximum(0.0, a[:, 3:4] - a[:, 1:2])
    area_b = np.maximum(0.0, b[:, 2:3].T - b[:, 0:1].T) * np.maximum(0.0, b[:, 3:4].T - b[:, 1:2].T)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / union, 0.0)


def match_image_predictions_at_iou(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    num_classes: int,
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Perform class-aware greedy one-to-one matching for a single image at conf_threshold."""
    filtered = [p for p in preds if float(p["score"]) >= conf_threshold]
    filtered.sort(key=lambda p: (-float(p["score"]), int(p["class_id"])))

    tp_by_cls = np.zeros(num_classes, dtype=np.int64)
    fp_by_cls = np.zeros(num_classes, dtype=np.int64)
    fn_by_cls = np.zeros(num_classes, dtype=np.int64)
    gt_count_by_cls = np.zeros(num_classes, dtype=np.int64)
    img_has_cls = np.zeros(num_classes, dtype=np.int64)

    matched_pairs: list[dict[str, Any]] = []
    unmatched_preds: list[dict[str, Any]] = []
    unmatched_gts: list[dict[str, Any]] = []

    for cid in range(num_classes):
        cls_gts = [g for g in gts if int(g["class_id"]) == cid]
        cls_preds = [p for p in filtered if int(p["class_id"]) == cid]
        gt_count_by_cls[cid] = len(cls_gts)
        if cls_gts:
            img_has_cls[cid] = 1

        used_gt = [False] * len(cls_gts)
        gt_boxes = np.array([g["bbox_xyxy"] for g in cls_gts], dtype=np.float64).reshape(-1, 4)

        for pred in cls_preds:
            if not cls_gts:
                fp_by_cls[cid] += 1
                unmatched_preds.append(pred)
                continue
            p_box = np.array([pred["bbox_xyxy"]], dtype=np.float64)
            ious = compute_iou_matrix(p_box, gt_boxes)[0]
            best_idx = -1
            best_iou = -1.0
            for g_idx, iou_val in enumerate(ious):
                if not used_gt[g_idx] and iou_val > best_iou:
                    best_iou = float(iou_val)
                    best_idx = g_idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                used_gt[best_idx] = True
                tp_by_cls[cid] += 1
                matched_pairs.append({"pred": pred, "gt": cls_gts[best_idx], "iou": best_iou})
            else:
                fp_by_cls[cid] += 1
                unmatched_preds.append(pred)

        for g_idx, was_used in enumerate(used_gt):
            if not was_used:
                fn_by_cls[cid] += 1
                unmatched_gts.append(cls_gts[g_idx])

    return {
        "tp": tp_by_cls,
        "fp": fp_by_cls,
        "fn": fn_by_cls,
        "gt": gt_count_by_cls,
        "img_present": img_has_cls,
        "matched_pairs": matched_pairs,
        "unmatched_preds": unmatched_preds,
        "unmatched_gts": unmatched_gts,
    }


def _aggregate_counts(
    tp_sum: np.ndarray,
    fp_sum: np.ndarray,
    fn_sum: np.ndarray,
    gt_sum: np.ndarray,
    img_present_sum: np.ndarray,
    num_images: int,
    class_names: list[str],
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    undefined_classes: list[str] = []

    for cid, cname in enumerate(class_names):
        tp = int(tp_sum[cid])
        fp = int(fp_sum[cid])
        fn = int(fn_sum[cid])
        n_gt = int(gt_sum[cid])
        n_imgs = int(img_present_sum[cid])

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        if n_gt > 0:
            rec: float | None = float(tp / (tp + fn))
            f1: float | None = float((2.0 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
            recalls.append(rec)
            f1s.append(f1)
        else:
            rec = None
            f1 = None
            undefined_classes.append(cname)
        precisions.append(prec)

        per_class[cname] = {
            "class_id": cid,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_support": n_gt,
            "image_support": n_imgs,
            "precision": round(prec, 6),
            "recall": round(rec, 6) if rec is not None else None,
            "f1": round(f1, 6) if f1 is not None else None,
        }

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    if undefined_classes:
        macro_recall = None
        macro_f1 = None
    else:
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    fppi = float(fp_sum.sum() / max(1, num_images))
    return {
        "per_class": per_class,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6) if macro_recall is not None else None,
        "macro_f1": round(macro_f1, 6) if macro_f1 is not None else None,
        "fppi": round(fppi, 6),
        "num_images": num_images,
        "undefined_classes": undefined_classes,
    }


def compute_operating_point_metrics(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Compute per-class TP/FP/FN/support and macro P/R/F1/FPPI at a single confidence threshold."""
    num_classes = len(class_names)
    image_ids = sorted(gts_by_image.keys())
    tp_sum = np.zeros(num_classes, dtype=np.int64)
    fp_sum = np.zeros(num_classes, dtype=np.int64)
    fn_sum = np.zeros(num_classes, dtype=np.int64)
    gt_sum = np.zeros(num_classes, dtype=np.int64)
    img_present_sum = np.zeros(num_classes, dtype=np.int64)

    for img_id in image_ids:
        res = match_image_predictions_at_iou(
            preds=preds_by_image.get(img_id, []),
            gts=gts_by_image[img_id],
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        tp_sum += res["tp"]
        fp_sum += res["fp"]
        fn_sum += res["fn"]
        gt_sum += res["gt"]
        img_present_sum += res["img_present"]

    summary = _aggregate_counts(
        tp_sum, fp_sum, fn_sum, gt_sum, img_present_sum, len(image_ids), class_names
    )
    summary["conf_threshold"] = round(float(conf_threshold), 6)
    summary["iou_threshold"] = round(float(iou_threshold), 4)
    return summary


def image_cluster_bootstrap_recall(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Estimate per-class and macro recall SE and 95% CIs via image-level cluster bootstrap."""
    num_classes = len(class_names)
    image_ids = sorted(gts_by_image.keys())
    n_imgs = len(image_ids)
    if n_imgs == 0:
        raise ValueError("Cannot run cluster bootstrap on empty image set")

    per_img_tp = np.zeros((n_imgs, num_classes), dtype=np.int64)
    per_img_fn = np.zeros((n_imgs, num_classes), dtype=np.int64)

    for idx, img_id in enumerate(image_ids):
        res = match_image_predictions_at_iou(
            preds=preds_by_image.get(img_id, []),
            gts=gts_by_image[img_id],
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        per_img_tp[idx] = res["tp"]
        per_img_fn[idx] = res["fn"]

    rng = np.random.default_rng(seed)
    boot_recalls = np.full((replicates, num_classes), np.nan, dtype=np.float64)
    boot_macro = np.full((replicates,), np.nan, dtype=np.float64)

    for rep in range(replicates):
        sample_idx = rng.integers(0, n_imgs, size=n_imgs)
        tp_rep = per_img_tp[sample_idx].sum(axis=0)
        fn_rep = per_img_fn[sample_idx].sum(axis=0)
        denom = tp_rep + fn_rep
        with np.errstate(divide="ignore", invalid="ignore"):
            rec_rep = np.where(denom > 0, tp_rep / np.maximum(denom, 1), np.nan)
        boot_recalls[rep] = rec_rep
        if not np.isnan(rec_rep).any():
            boot_macro[rep] = float(np.mean(rec_rep))

    point_tp = per_img_tp.sum(axis=0)
    point_fn = per_img_fn.sum(axis=0)
    point_denom = point_tp + point_fn

    per_class_ci: dict[str, dict[str, Any]] = {}
    for cid, cname in enumerate(class_names):
        valid_vals = boot_recalls[~np.isnan(boot_recalls[:, cid]), cid]
        point = float(point_tp[cid] / point_denom[cid]) if point_denom[cid] > 0 else None
        if valid_vals.size > 0 and point is not None:
            se = float(np.std(valid_vals, ddof=1)) if valid_vals.size > 1 else 0.0
            ci_low = float(np.percentile(valid_vals, 2.5))
            ci_high = float(np.percentile(valid_vals, 97.5))
        else:
            se, ci_low, ci_high = 0.0, 0.0, 0.0
        per_class_ci[cname] = {
            "point_recall": round(point, 6) if point is not None else None,
            "se": round(se, 6),
            "ci95_low": round(ci_low, 6),
            "ci95_high": round(ci_high, 6),
            "tp": int(point_tp[cid]),
            "fn": int(point_fn[cid]),
            "support": int(point_denom[cid]),
        }

    valid_macro = boot_macro[~np.isnan(boot_macro)]
    if valid_macro.size > 0 and not (point_denom == 0).any():
        macro_point = float(np.mean(point_tp / point_denom))
        macro_se = float(np.std(valid_macro, ddof=1)) if valid_macro.size > 1 else 0.0
        macro_low = float(np.percentile(valid_macro, 2.5))
        macro_high = float(np.percentile(valid_macro, 97.5))
    else:
        macro_point, macro_se, macro_low, macro_high = None, 0.0, 0.0, 0.0

    return {
        "method": "image_level_cluster_bootstrap",
        "replicates": int(replicates),
        "seed": int(seed),
        "conf_threshold": round(float(conf_threshold), 6),
        "iou_threshold": round(float(iou_threshold), 4),
        "per_class": per_class_ci,
        "macro_recall": {
            "point": round(macro_point, 6) if macro_point is not None else None,
            "se": round(macro_se, 6),
            "ci95_low": round(macro_low, 6),
            "ci95_high": round(macro_high, 6),
        },
    }


def compute_validation_recall_floors(
    op_metrics: dict[str, Any],
    bootstrap_report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute per-class validation regression floors: floor_c = clip(R_val,c - max(0.03, 1.5 * SE_c), 0, 1)."""
    floors: dict[str, dict[str, Any]] = {}
    for cname, cls_stats in op_metrics["per_class"].items():
        r_val = cls_stats["recall"]
        se = float(bootstrap_report["per_class"][cname]["se"])
        if r_val is None:
            raise ValueError(f"Cannot compute recall floor for class '{cname}' with zero validation support")
        margin = max(0.03, 1.5 * se)
        floor_val = float(np.clip(float(r_val) - margin, 0.0, 1.0))
        floors[cname] = {
            "val_recall": round(float(r_val), 6),
            "bootstrap_se": round(se, 6),
            "margin": round(margin, 6),
            "floor": round(floor_val, 6),
            "gt_support": int(cls_stats["gt_support"]),
            "image_support": int(cls_stats["image_support"]),
        }
    return floors


def compute_coco_map(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    num_classes: int,
    max_dets: int = 100,
) -> dict[str, Any]:
    """Compute COCO mAP@0.5, mAP@0.5:0.95, and per-class AP@0.5 using pycocotools (with AP fallback)."""
    image_ids = sorted(gts_by_image.keys())
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt_dict: dict[str, Any] = {
            "images": [{"id": idx + 1, "file_name": img_id} for idx, img_id in enumerate(image_ids)],
            "categories": [{"id": cid, "name": str(cid)} for cid in range(num_classes)],
            "annotations": [],
        }
        img_id_to_int = {img_id: idx + 1 for idx, img_id in enumerate(image_ids)}
        ann_id = 1
        for img_id in image_ids:
            for g in gts_by_image[img_id]:
                x1, y1, x2, y2 = [float(v) for v in g["bbox_xyxy"]]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                coco_gt_dict["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img_id_to_int[img_id],
                        "category_id": int(g["class_id"]),
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        det_list: list[dict[str, Any]] = []
        for img_id in image_ids:
            for p in preds_by_image.get(img_id, []):
                x1, y1, x2, y2 = [float(v) for v in p["bbox_xyxy"]]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0.0 or h <= 0.0:
                    continue
                det_list.append(
                    {
                        "image_id": img_id_to_int[img_id],
                        "category_id": int(p["class_id"]),
                        "bbox": [x1, y1, w, h],
                        "score": float(p["score"]),
                    }
                )

        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO()
            coco_gt.dataset = coco_gt_dict
            coco_gt.createIndex()
            if det_list:
                coco_dt = coco_gt.loadRes(det_list)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.params.maxDets = [1, 10, max_dets]
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                map_50_95 = float(max(0.0, coco_eval.stats[0]))
                map_50 = float(max(0.0, coco_eval.stats[1]))
                precision_tensor = coco_eval.eval["precision"]  # [T, R, K, A, M]
                per_class_ap50 = {}
                for cid in range(num_classes):
                    slice_50 = precision_tensor[0, :, cid, 0, 2]
                    valid = slice_50[slice_50 > -1]
                    per_class_ap50[str(cid)] = round(float(np.mean(valid)), 6) if valid.size > 0 else 0.0
            else:
                map_50_95 = 0.0
                map_50 = 0.0
                per_class_ap50 = {str(cid): 0.0 for cid in range(num_classes)}

        return {
            "evaluator": "pycocotools.COCOeval",
            "max_dets": max_dets,
            "map_50": round(map_50, 6),
            "map_50_95": round(map_50_95, 6),
            "per_class_ap50": per_class_ap50,
        }
    except ImportError:
        return _compute_voc_style_map_fallback(preds_by_image, gts_by_image, num_classes, max_dets=max_dets)


def _compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for idx in range(mpre.size - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = np.mean([np.max(mpre[mrec >= r]) if np.any(mrec >= r) else 0.0 for r in recall_points])
    return float(ap)


def _compute_voc_style_map_fallback(
    preds_by_image: dict[str, list[dict[str, Any]]],
    gts_by_image: dict[str, list[dict[str, Any]]],
    num_classes: int,
    max_dets: int = 100,
) -> dict[str, Any]:
    """Pure-NumPy 101-point COCO AP fallback when pycocotools is not installed."""
    iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    image_ids = sorted(gts_by_image.keys())
    ap_matrix = np.zeros((len(iou_thresholds), num_classes), dtype=np.float64)

    for t_idx, iou_thr in enumerate(iou_thresholds):
        for cid in range(num_classes):
            all_preds: list[tuple[str, float, list[float]]] = []
            gt_by_img: dict[str, np.ndarray] = {}
            total_gt = 0
            for img_id in image_ids:
                c_gts = [g["bbox_xyxy"] for g in gts_by_image[img_id] if int(g["class_id"]) == cid]
                gt_by_img[img_id] = np.array(c_gts, dtype=np.float64).reshape(-1, 4)
                total_gt += len(c_gts)
                c_preds = sorted(
                    [p for p in preds_by_image.get(img_id, []) if int(p["class_id"]) == cid],
                    key=lambda x: -float(x["score"]),
                )[:max_dets]
                for p in c_preds:
                    all_preds.append((img_id, float(p["score"]), p["bbox_xyxy"]))

            if total_gt == 0 or not all_preds:
                ap_matrix[t_idx, cid] = 0.0
                continue

            all_preds.sort(key=lambda x: -x[1])
            used = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_by_img.items()}
            tp = np.zeros(len(all_preds), dtype=np.float64)
            fp = np.zeros(len(all_preds), dtype=np.float64)

            for p_idx, (img_id, _, p_box) in enumerate(all_preds):
                gt_boxes = gt_by_img[img_id]
                if gt_boxes.shape[0] == 0:
                    fp[p_idx] = 1.0
                    continue
                ious = compute_iou_matrix(np.array([p_box], dtype=np.float64), gt_boxes)[0]
                best_g = int(np.argmax(ious))
                if ious[best_g] >= iou_thr and not used[img_id][best_g]:
                    used[img_id][best_g] = True
                    tp[p_idx] = 1.0
                else:
                    fp[p_idx] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / float(total_gt)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)
            ap_matrix[t_idx, cid] = _compute_ap_101(rec, prec)

    return {
        "evaluator": "numpy_101pt_fallback",
        "max_dets": max_dets,
        "map_50": round(float(np.mean(ap_matrix[0])), 6),
        "map_50_95": round(float(np.mean(ap_matrix)), 6),
        "per_class_ap50": {str(cid): round(float(ap_matrix[0, cid]), 6) for cid in range(num_classes)},
    }
