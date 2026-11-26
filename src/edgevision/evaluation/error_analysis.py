"""Detection-specific error analysis: FN/FP breakdown, size buckets, class confusion, and visual overlays."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgevision.data.annotations import ImageRecord
from edgevision.evaluation.metrics import compute_iou_matrix, match_image_predictions_at_iou


def _size_bucket(area: float) -> str:
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def analyze_detection_errors(
    records: list[ImageRecord],
    preds_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    conf_threshold: float,
    iou_threshold: float = 0.50,
    confusion_iou_threshold: float = 0.30,
) -> dict[str, Any]:
    """Compute per-class FN/FP, localization errors, size-bucket recall, and class-confusion counts."""
    num_classes = len(class_names)
    fn_by_class: Counter[str] = Counter()
    fp_by_class: Counter[str] = Counter()
    low_iou_loc_errors: Counter[str] = Counter()
    confusion_pairs: Counter[str] = Counter()

    bucket_tp: Counter[str] = Counter()
    bucket_gt: Counter[str] = Counter()

    image_error_scores: list[tuple[int, str]] = []

    for rec in records:
        gts = [
            {"class_id": b.class_id, "bbox_xyxy": [b.xmin, b.ymin, b.xmax, b.ymax]}
            for b in rec.boxes
        ]
        preds = preds_by_image.get(rec.image_id, [])
        res = match_image_predictions_at_iou(
            preds=preds,
            gts=gts,
            num_classes=num_classes,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        for pair in res["matched_pairs"]:
            gt_box = pair["gt"]["bbox_xyxy"]
            area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            bucket = _size_bucket(area)
            bucket_tp[bucket] += 1
            bucket_gt[bucket] += 1

        for ug in res["unmatched_gts"]:
            cname = class_names[int(ug["class_id"])]
            fn_by_class[cname] += 1
            gt_box = ug["bbox_xyxy"]
            area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            bucket_gt[_size_bucket(area)] += 1

        # Classify unmatched predictions into localization error vs class confusion vs background FP.
        unmatched_gt_boxes = (
            np.array([g["bbox_xyxy"] for g in res["unmatched_gts"]], dtype=np.float64).reshape(-1, 4)
        )
        for up in res["unmatched_preds"]:
            p_cname = class_names[int(up["class_id"])]
            fp_by_class[p_cname] += 1
            if unmatched_gt_boxes.shape[0] > 0:
                ious = compute_iou_matrix(np.array([up["bbox_xyxy"]]), unmatched_gt_boxes)[0]
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])
                if best_iou >= 0.10:
                    gt_cid = int(res["unmatched_gts"][best_idx]["class_id"])
                    if gt_cid == int(up["class_id"]):
                        low_iou_loc_errors[p_cname] += 1
                    elif best_iou >= confusion_iou_threshold:
                        confusion_pairs[f"{class_names[gt_cid]}->{p_cname}"] += 1

        err_count = len(res["unmatched_gts"]) + len(res["unmatched_preds"])
        image_error_scores.append((err_count, rec.image_id))

    size_recall = {
        bucket: {
            "tp": int(bucket_tp[bucket]),
            "gt": int(bucket_gt[bucket]),
            "recall": round(float(bucket_tp[bucket] / bucket_gt[bucket]), 4)
            if bucket_gt[bucket] > 0
            else None,
        }
        for bucket in ("small", "medium", "large")
    }

    image_error_scores.sort(key=lambda item: (-item[0], item[1]))
    return {
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "fn_by_class": dict(fn_by_class),
        "fp_by_class": dict(fp_by_class),
        "low_iou_localization_errors": dict(low_iou_loc_errors),
        "class_confusion_pairs": dict(confusion_pairs),
        "size_bucket_recall": size_recall,
        "top_error_image_ids": [img_id for _, img_id in image_error_scores[:12]],
    }


def save_error_overlays(
    records: list[ImageRecord],
    preds_by_image: dict[str, list[dict[str, Any]]],
    class_names: list[str],
    output_dir: str | Path,
    image_ids: list[str],
    conf_threshold: float,
) -> list[Path]:
    """Render side-by-side ground-truth (green) and predicted (red) box overlays."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_map = {rec.image_id: rec for rec in records}
    saved: list[Path] = []

    for img_id in image_ids:
        rec = rec_map.get(img_id)
        if rec is None:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        with Image.open(rec.image_path) as img:
            ax.imshow(img.convert("L"), cmap="gray")

        for b in rec.boxes:
            rect = patches.Rectangle(
                (b.xmin, b.ymin),
                b.width,
                b.height,
                linewidth=1.5,
                edgecolor="#00e676",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(b.xmin, max(0.0, b.ymin - 2.0), f"GT:{b.class_name}", color="#00e676", fontsize=7)

        for p in preds_by_image.get(img_id, []):
            if float(p["score"]) < conf_threshold:
                continue
            x1, y1, x2, y2 = p["bbox_xyxy"]
            cname = class_names[int(p["class_id"])]
            rect = patches.Rectangle(
                (x1, y1),
                max(0.0, x2 - x1),
                max(0.0, y2 - y1),
                linewidth=1.5,
                linestyle="--",
                edgecolor="#ff5252",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(x1, min(float(rec.height), y2 + 7.0), f"P:{cname} {p['score']:.2f}", color="#ff5252", fontsize=7)

        ax.set_title(img_id, fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        dest = out_dir / f"error_{img_id}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        saved.append(dest)

    return saved


def write_failure_analysis_markdown(
    analysis: dict[str, Any],
    overlay_paths: list[Path],
    output_path: str | Path,
) -> Path:
    """Write reports/failure_analysis.md summarizing detection error patterns."""
    lines = [
        "# Detection Error and Failure Analysis",
        "",
        f"- **Confidence threshold:** `{analysis['conf_threshold']}`",
        f"- **IoU threshold:** `{analysis['iou_threshold']}`",
        "",
        "## 1. False Negatives and False Positives by Class",
        "",
        "| Class | False Negatives | False Positives | Low-IoU Localization Errors |",
        "|---|---:|---:|---:|",
    ]
    all_classes = sorted(
        set(analysis["fn_by_class"].keys())
        | set(analysis["fp_by_class"].keys())
        | set(analysis["low_iou_localization_errors"].keys())
    )
    for cname in all_classes:
        lines.append(
            f"| `{cname}` | {analysis['fn_by_class'].get(cname, 0)} | "
            f"{analysis['fp_by_class'].get(cname, 0)} | "
            f"{analysis['low_iou_localization_errors'].get(cname, 0)} |"
        )

    lines.extend(["", "## 2. Recall by Bounding-Box Size Bucket", ""])
    lines.append("| Size Bucket | TP | GT | Recall |")
    lines.append("|---|---:|---:|---:|")
    for bucket, stats in analysis["size_bucket_recall"].items():
        lines.append(f"| {bucket} | {stats['tp']} | {stats['gt']} | {stats['recall']} |")

    lines.extend(["", "## 3. Cross-Class Confusions (GT -> Predicted)", ""])
    if analysis["class_confusion_pairs"]:
        for pair, count in sorted(analysis["class_confusion_pairs"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{pair}`: {count}")
    else:
        lines.append("- None observed at the configured threshold.")

    if overlay_paths:
        lines.extend(["", "## 4. Representative Error Overlays", ""])
        for p in overlay_paths:
            lines.append(f"- `{p.as_posix()}`")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest
