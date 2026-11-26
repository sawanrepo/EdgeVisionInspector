"""Dataset auditing, exact/near-duplicate grouping, visualization, and markdown reporting."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from edgevision.config import get_logger
from edgevision.data.annotations import ClassSpec, DatasetParseResult, ImageRecord

logger = get_logger(__name__)


@dataclass
class DuplicateAnalysis:
    """Exact and near-duplicate groupings over image records."""

    exact_groups: list[list[str]]
    near_groups: list[list[str]]
    group_map: dict[str, str]


def _compute_dhash(gray_arr: np.ndarray, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) from a 2D grayscale array."""
    pil_img = Image.fromarray(gray_arr.astype(np.uint8), mode="L").resize(
        (hash_size + 1, hash_size), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(pil_img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _normalized_vector(gray_arr: np.ndarray, size: int = 32) -> np.ndarray:
    pil_img = Image.fromarray(gray_arr.astype(np.uint8), mode="L").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    vec = np.asarray(pil_img, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else vec


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def analyze_duplicates(
    records: list[ImageRecord],
    phash_max_dist: int = 4,
    cosine_min: float = 0.995,
) -> DuplicateAnalysis:
    """Detect exact pixel duplicates and perceptual near-duplicates and assign group IDs."""
    if not records:
        return DuplicateAnalysis(exact_groups=[], near_groups=[], group_map={})

    ids = [rec.image_id for rec in records]
    uf_exact = _UnionFind(ids)
    uf_all = _UnionFind(ids)

    by_sha: dict[str, list[str]] = defaultdict(list)
    dhashes: list[int] = []
    vectors: list[np.ndarray] = []

    for rec in records:
        with Image.open(rec.image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        pixel_sha = hashlib.sha256(gray.tobytes()).hexdigest()
        rec.pixel_sha256 = pixel_sha
        by_sha[pixel_sha].append(rec.image_id)
        dhashes.append(_compute_dhash(gray))
        vectors.append(_normalized_vector(gray))

    exact_groups: list[list[str]] = []
    for members in by_sha.values():
        if len(members) > 1:
            exact_groups.append(sorted(members))
            for member in members[1:]:
                uf_exact.union(members[0], member)
                uf_all.union(members[0], member)

    vec_matrix = np.stack(vectors, axis=0) if vectors else np.zeros((0, 1), dtype=np.float32)
    sim_matrix = vec_matrix @ vec_matrix.T if len(records) > 1 else np.ones((len(records), len(records)))

    n = len(records)
    for i in range(n):
        for j in range(i + 1, n):
            if _hamming_distance(dhashes[i], dhashes[j]) <= phash_max_dist and float(sim_matrix[i, j]) >= cosine_min:
                uf_all.union(ids[i], ids[j])

    all_components: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        all_components[uf_all.find(image_id)].append(image_id)

    near_groups: list[list[str]] = []
    group_map: dict[str, str] = {}
    for root_id, members in sorted(all_components.items()):
        sorted_members = sorted(members)
        group_id = f"grp_{root_id}"
        for member in sorted_members:
            group_map[member] = group_id
        if len(sorted_members) > 1:
            # Distinguish near-only groups from pure exact groups
            if len({uf_exact.find(m) for m in sorted_members}) > 1:
                near_groups.append(sorted_members)

    for rec in records:
        rec.duplicate_group_id = group_map[rec.image_id]

    return DuplicateAnalysis(exact_groups=exact_groups, near_groups=near_groups, group_map=group_map)


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
    }


def compute_audit_summary(
    parse_result: DatasetParseResult,
    dup_analysis: DuplicateAnalysis,
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute image, instance, box-geometry, intensity, and integrity audit metrics."""
    records = parse_result.records
    class_specs = parse_result.classes
    id_to_spec = {spec.class_id: spec for spec in class_specs}

    resolutions = Counter((rec.width, rec.height) for rec in records)
    modes = Counter(rec.mode for rec in records)
    formats = Counter(rec.file_format for rec in records)

    image_presence: Counter[int] = Counter()
    instance_counts: Counter[int] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    aspect_ratios: list[float] = []
    rel_areas: list[float] = []

    mean_intensities: list[float] = []
    std_contrasts: list[float] = []
    hist_accum = np.zeros(16, dtype=np.int64)

    for rec in records:
        img_area = float(max(1, rec.width * rec.height))
        for cid in rec.class_ids:
            image_presence[cid] += 1
        for box in rec.boxes:
            instance_counts[box.class_id] += 1
            widths.append(box.width)
            heights.append(box.height)
            areas.append(box.area)
            aspect_ratios.append(box.width / max(1e-6, box.height))
            rel_areas.append(box.area / img_area)

        with Image.open(rec.image_path) as img:
            gray = np.asarray(img.convert("L"), dtype=np.float32)
        mean_intensities.append(float(np.mean(gray)))
        std_contrasts.append(float(np.std(gray)))
        hist, _ = np.histogram(gray, bins=16, range=(0.0, 256.0))
        hist_accum += hist

    class_table: list[dict[str, Any]] = []
    for cid, spec in sorted(id_to_spec.items()):
        class_table.append(
            {
                "class_id": cid,
                "name": spec.name,
                "display_name": spec.display_name,
                "image_count": int(image_presence[cid]),
                "instance_count": int(instance_counts[cid]),
            }
        )

    issue_counts = Counter(issue.issue_type for issue in parse_result.issues)
    total_hist = int(hist_accum.sum()) or 1
    hist_frac = [round(float(x / total_hist), 5) for x in hist_accum]

    return {
        "dataset_name": dataset_cfg.get("dataset_name", "NEU-DET"),
        "licence_gate": dataset_cfg.get("licence_gate", {}),
        "num_valid_images": len(records),
        "num_excluded_records": len(parse_result.excluded_records),
        "total_instances": sum(instance_counts.values()),
        "resolutions": {f"{w}x{h}": count for (w, h), count in resolutions.items()},
        "image_modes": dict(modes),
        "file_formats": dict(formats),
        "class_distribution": class_table,
        "bbox_statistics": {
            "width_px": _quantiles(widths),
            "height_px": _quantiles(heights),
            "area_px2": _quantiles(areas),
            "aspect_ratio": _quantiles(aspect_ratios),
            "relative_area": _quantiles(rel_areas),
        },
        "intensity_statistics": {
            "mean_intensity": _quantiles(mean_intensities),
            "contrast_std": _quantiles(std_contrasts),
            "histogram_16_bins": hist_frac,
        },
        "duplicates": {
            "exact_group_count": len(dup_analysis.exact_groups),
            "near_group_count": len(dup_analysis.near_groups),
            "exact_groups": dup_analysis.exact_groups,
            "near_groups": dup_analysis.near_groups,
        },
        "issue_counts": dict(issue_counts),
        "excluded_records": parse_result.excluded_records,
    }


def save_audit_visualizations(
    records: list[ImageRecord],
    class_specs: list[ClassSpec],
    output_dir: str | Path,
    samples_per_class: int = 3,
) -> list[Path]:
    """Render annotated bounding-box sample grids for each defect class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    by_class: dict[int, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        for cid in rec.class_ids:
            by_class[cid].append(rec)

    for spec in class_specs:
        candidates = by_class.get(spec.class_id, [])[:samples_per_class]
        if not candidates:
            continue
        fig, axes = plt.subplots(1, len(candidates), figsize=(4 * len(candidates), 4))
        if len(candidates) == 1:
            axes = [axes]
        for ax, rec in zip(axes, candidates, strict=False):
            with Image.open(rec.image_path) as img:
                ax.imshow(img.convert("L"), cmap="gray")
            for box in rec.boxes:
                color = "#ff5252" if box.class_id == spec.class_id else "#40c4ff"
                rect = patches.Rectangle(
                    (box.xmin, box.ymin),
                    box.width,
                    box.height,
                    linewidth=1.5,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    box.xmin,
                    max(0.0, box.ymin - 3.0),
                    box.class_name,
                    color=color,
                    fontsize=8,
                    bbox={"facecolor": "black", "alpha": 0.5, "pad": 1},
                )
            ax.set_title(rec.image_id, fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Class {spec.class_id}: {spec.display_name}", fontsize=11)
        fig.tight_layout()
        dest = out_dir / f"audit_class_{spec.class_id}_{spec.name}.png"
        fig.savefig(dest, dpi=150)
        plt.close(fig)
        saved_paths.append(dest)

    return saved_paths


def render_audit_markdown(summary: dict[str, Any], figure_paths: list[Path] | None = None) -> str:
    """Format the dataset audit summary as a structured markdown report."""
    licence = summary.get("licence_gate", {})
    lines = [
        f"# {summary.get('dataset_name', 'NEU-DET')} Dataset Audit Report",
        "",
        "## 1. Source and Licence Gate",
        "",
        f"- **Source URL:** {licence.get('source_url') or 'Not specified'}",
        f"- **Dataset version:** {licence.get('dataset_version') or 'Not specified'}",
        f"- **Access date:** {licence.get('access_date') or 'Not specified'}",
        f"- **Licence status:** `{licence.get('licence_status', 'pending_verification')}`",
        f"- **Redistribution permitted:** `{licence.get('redistribution_permitted', False)}`",
        "",
        "## 2. File Integrity and Format Summary",
        "",
        f"- **Valid images:** {summary['num_valid_images']}",
        f"- **Excluded images/annotations:** {summary['num_excluded_records']}",
        f"- **Total valid bounding boxes:** {summary['total_instances']}",
        f"- **Image resolutions:** {summary['resolutions']}",
        f"- **Color modes:** {summary['image_modes']}",
        f"- **File formats:** {summary['file_formats']}",
        "",
        "## 3. Class-Level Presence and Instance Counts",
        "",
        "| Class ID | Name | Display Name | Images Containing Class | Object Instances |",
        "|---:|---|---|---:|---:|",
    ]
    for row in summary["class_distribution"]:
        lines.append(
            f"| {row['class_id']} | `{row['name']}` | {row['display_name']} | "
            f"{row['image_count']} | {row['instance_count']} |"
        )

    bbox = summary["bbox_statistics"]
    lines.extend(
        [
            "",
            "## 4. Bounding-Box Geometry Distributions",
            "",
            "| Statistic | Min | P50 | P95 | Max | Mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in [
        ("Width (px)", "width_px"),
        ("Height (px)", "height_px"),
        ("Area (px^2)", "area_px2"),
        ("Aspect Ratio (W/H)", "aspect_ratio"),
        ("Relative Image Area", "relative_area"),
    ]:
        q = bbox[key]
        lines.append(
            f"| {label} | {q['min']} | {q['p50']} | {q['p95']} | {q['max']} | {q['mean']} |"
        )

    intensity = summary["intensity_statistics"]
    lines.extend(
        [
            "",
            "## 5. Grayscale Intensity and Contrast",
            "",
            f"- **Mean image intensity (0-255):** {intensity['mean_intensity']}",
            f"- **Within-image contrast (std):** {intensity['contrast_std']}",
            f"- **16-bin normalized histogram:** `{intensity['histogram_16_bins']}`",
            "",
            "## 6. Duplicate and Annotation-Issue Checks",
            "",
            f"- **Exact duplicate groups:** {summary['duplicates']['exact_group_count']}",
            f"- **Near-duplicate groups:** {summary['duplicates']['near_group_count']}",
            f"- **Annotation issue counts:** {summary['issue_counts'] or 'None'}",
            "",
            "## 7. Known Dataset Limitations",
            "",
            "- Images are small 200 x 200 pre-cropped benchmark tiles rather than full high-resolution steel coil strips.",
            "- Image-level class balance does not imply object-instance or box-size balance.",
            "- Benchmark performance on NEU-DET does not by itself validate live production-line inspection.",
        ]
    )

    if summary["excluded_records"]:
        lines.extend(["", "### Excluded Files", ""])
        for item in summary["excluded_records"]:
            lines.append(f"- `{item['image_id']}` ({item['path']}): {'; '.join(item['reasons'])}")

    if figure_paths:
        lines.extend(["", "## 8. Representative Annotated Samples", ""])
        for fig_path in figure_paths:
            lines.append(f"- `{fig_path.as_posix()}`")

    return "\n".join(lines) + "\n"


def run_data_audit(
    parse_result: DatasetParseResult,
    dataset_cfg: dict[str, Any],
    report_path: str | Path,
    figures_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run duplicate grouping, compute audit statistics, and write reports/data_audit.md."""
    split_cfg = dataset_cfg.get("split", {})
    dup_analysis = analyze_duplicates(
        parse_result.records,
        phash_max_dist=int(split_cfg.get("near_duplicate_phash_max_dist", 4)),
        cosine_min=float(split_cfg.get("near_duplicate_cosine_min", 0.995)),
    )
    summary = compute_audit_summary(parse_result, dup_analysis, dataset_cfg)

    fig_paths: list[Path] = []
    if figures_dir is not None and parse_result.records:
        fig_paths = save_audit_visualizations(parse_result.records, parse_result.classes, figures_dir)

    md_text = render_audit_markdown(summary, fig_paths)
    dest = Path(report_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md_text, encoding="utf-8")
    logger.info("Wrote dataset audit report to %s", dest)
    return summary
