"""Annotation parsing, validation, and canonical schema representation."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from edgevision.config import get_logger, load_yaml

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ClassSpec:
    """Canonical defect class definition."""

    class_id: int
    name: str
    display_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in absolute xyxy pixel coordinates."""

    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class AnnotationIssue:
    """Recorded annotation or image integrity defect without silent repair."""

    image_id: str
    file_path: str
    issue_type: str
    detail: str
    fatal: bool = True


@dataclass
class ImageRecord:
    """Canonical single-image detection record."""

    image_id: str
    image_path: str
    annotation_path: str | None
    width: int
    height: int
    mode: str
    channels: int
    file_format: str
    boxes: list[BoundingBox] = field(default_factory=list)
    duplicate_group_id: str | None = None
    pixel_sha256: str | None = None

    @property
    def class_ids(self) -> list[int]:
        return sorted({box.class_id for box in self.boxes})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ImageRecord:
        boxes = [BoundingBox(**box) for box in payload.get("boxes", [])]
        return cls(
            image_id=payload["image_id"],
            image_path=payload["image_path"],
            annotation_path=payload.get("annotation_path"),
            width=int(payload["width"]),
            height=int(payload["height"]),
            mode=str(payload["mode"]),
            channels=int(payload["channels"]),
            file_format=str(payload["file_format"]),
            boxes=boxes,
            duplicate_group_id=payload.get("duplicate_group_id"),
            pixel_sha256=payload.get("pixel_sha256"),
        )


@dataclass
class DatasetParseResult:
    """Result of scanning and validating a dataset directory."""

    records: list[ImageRecord]
    excluded_records: list[dict[str, Any]]
    issues: list[AnnotationIssue]
    classes: list[ClassSpec]


def build_class_lookup(dataset_cfg: dict[str, Any]) -> tuple[list[ClassSpec], dict[str, ClassSpec]]:
    """Build canonical class list and case-insensitive alias lookup table."""
    specs: list[ClassSpec] = []
    alias_map: dict[str, ClassSpec] = {}
    for entry in dataset_cfg.get("classes", []):
        spec = ClassSpec(
            class_id=int(entry["id"]),
            name=str(entry["name"]),
            display_name=str(entry.get("display_name", entry["name"])),
            aliases=tuple(str(a) for a in entry.get("aliases", [entry["name"]])),
        )
        specs.append(spec)
        for alias in (*spec.aliases, spec.name, spec.display_name):
            alias_map[alias.strip().lower()] = spec
    specs.sort(key=lambda item: item.class_id)
    return specs, alias_map


def _parse_float(text: str | None, field_name: str) -> float:
    if text is None:
        raise ValueError(f"Missing coordinate element '{field_name}'")
    return float(text.strip())


def parse_voc_xml(
    xml_path: str | Path,
    alias_map: dict[str, ClassSpec],
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> tuple[list[BoundingBox], list[AnnotationIssue], tuple[int, int] | None]:
    """Parse one Pascal VOC XML file and flag invalid, empty, duplicate, or out-of-bounds boxes."""
    path = Path(xml_path)
    image_id = path.stem
    issues: list[AnnotationIssue] = []
    boxes: list[BoundingBox] = []

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as exc:
        issues.append(
            AnnotationIssue(
                image_id=image_id,
                file_path=str(path),
                issue_type="malformed_xml",
                detail=f"XML parse error: {exc}",
                fatal=True,
            )
        )
        return [], issues, None

    xml_size: tuple[int, int] | None = None
    size_node = root.find("size")
    if size_node is not None:
        w_text = size_node.findtext("width")
        h_text = size_node.findtext("height")
        if w_text is not None and h_text is not None:
            try:
                xml_w = int(float(w_text))
                xml_h = int(float(h_text))
                xml_size = (xml_w, xml_h)
                if expected_width is not None and expected_height is not None:
                    if (xml_w, xml_h) != (expected_width, expected_height):
                        issues.append(
                            AnnotationIssue(
                                image_id=image_id,
                                file_path=str(path),
                                issue_type="size_mismatch",
                                detail=(
                                    f"XML size {(xml_w, xml_h)} does not match image size "
                                    f"{(expected_width, expected_height)}"
                                ),
                                fatal=True,
                            )
                        )
            except ValueError:
                issues.append(
                    AnnotationIssue(
                        image_id=image_id,
                        file_path=str(path),
                        issue_type="invalid_size_metadata",
                        detail=f"Non-numeric XML size: width={w_text}, height={h_text}",
                        fatal=True,
                    )
                )

    bound_w = expected_width if expected_width is not None else (xml_size[0] if xml_size else None)
    bound_h = expected_height if expected_height is not None else (xml_size[1] if xml_size else None)

    objects = root.findall("object")
    if not objects:
        issues.append(
            AnnotationIssue(
                image_id=image_id,
                file_path=str(path),
                issue_type="empty_annotation",
                detail="XML contains zero <object> entries",
                fatal=True,
            )
        )
        return [], issues, xml_size

    seen_exact: set[tuple[int, float, float, float, float]] = set()
    for idx, obj in enumerate(objects):
        raw_name = (obj.findtext("name") or "").strip()
        spec = alias_map.get(raw_name.lower())
        if spec is None:
            issues.append(
                AnnotationIssue(
                    image_id=image_id,
                    file_path=str(path),
                    issue_type="unknown_class",
                    detail=f"Object #{idx} has unrecognized class '{raw_name}'",
                    fatal=True,
                )
            )
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            issues.append(
                AnnotationIssue(
                    image_id=image_id,
                    file_path=str(path),
                    issue_type="missing_bndbox",
                    detail=f"Object #{idx} lacks <bndbox>",
                    fatal=True,
                )
            )
            continue

        try:
            xmin = _parse_float(bndbox.findtext("xmin"), "xmin")
            ymin = _parse_float(bndbox.findtext("ymin"), "ymin")
            xmax = _parse_float(bndbox.findtext("xmax"), "xmax")
            ymax = _parse_float(bndbox.findtext("ymax"), "ymax")
        except ValueError as exc:
            issues.append(
                AnnotationIssue(
                    image_id=image_id,
                    file_path=str(path),
                    issue_type="invalid_coordinates",
                    detail=f"Object #{idx}: {exc}",
                    fatal=True,
                )
            )
            continue

        if xmax <= xmin or ymax <= ymin:
            issues.append(
                AnnotationIssue(
                    image_id=image_id,
                    file_path=str(path),
                    issue_type="non_positive_box",
                    detail=f"Object #{idx} ({spec.name}) has non-positive dimensions ({xmin}, {ymin}, {xmax}, {ymax})",
                    fatal=True,
                )
            )
            continue

        if bound_w is not None and bound_h is not None:
            if xmin < 0 or ymin < 0 or xmax > bound_w or ymax > bound_h:
                issues.append(
                    AnnotationIssue(
                        image_id=image_id,
                        file_path=str(path),
                        issue_type="out_of_bounds",
                        detail=(
                            f"Object #{idx} ({spec.name}) box ({xmin}, {ymin}, {xmax}, {ymax}) "
                            f"exceeds image bounds ({bound_w}, {bound_h})"
                        ),
                        fatal=True,
                    )
                )
                continue

        dedup_key = (spec.class_id, round(xmin, 3), round(ymin, 3), round(xmax, 3), round(ymax, 3))
        if dedup_key in seen_exact:
            issues.append(
                AnnotationIssue(
                    image_id=image_id,
                    file_path=str(path),
                    issue_type="duplicate_box",
                    detail=f"Object #{idx} ({spec.name}) duplicates an identical box in the same image",
                    fatal=True,
                )
            )
            continue
        seen_exact.add(dedup_key)

        boxes.append(
            BoundingBox(
                class_id=spec.class_id,
                class_name=spec.name,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return boxes, issues, xml_size


def _discover_files(raw_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """Index image and XML annotation files recursively under raw_dir."""
    images_by_stem: dict[str, Path] = {}
    xmls_by_stem: dict[str, Path] = {}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            images_by_stem[path.stem] = path
        elif suffix == ".xml":
            xmls_by_stem[path.stem] = path
    return images_by_stem, xmls_by_stem


def _channel_count_from_mode(mode: str) -> int:
    mode_map = {"1": 1, "L": 1, "P": 1, "I": 1, "F": 1, "LA": 2, "RGB": 3, "YCbCr": 3, "RGBA": 4, "CMYK": 4}
    return mode_map.get(mode, 3)


def parse_dataset(raw_dir: str | Path, dataset_cfg: dict[str, Any] | str | Path) -> DatasetParseResult:
    """Scan raw_dir, parse and validate images and Pascal VOC XML annotations."""
    cfg = load_yaml(dataset_cfg) if isinstance(dataset_cfg, (str, Path)) else dataset_cfg
    specs, alias_map = build_class_lookup(cfg)
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    images_by_stem, xmls_by_stem = _discover_files(root)
    all_issues: list[AnnotationIssue] = []
    valid_records: list[ImageRecord] = []
    excluded_records: list[dict[str, Any]] = []

    for stem, xml_path in sorted(xmls_by_stem.items()):
        if stem not in images_by_stem:
            issue = AnnotationIssue(
                image_id=stem,
                file_path=str(xml_path),
                issue_type="orphan_annotation",
                detail="Annotation XML has no matching image file",
                fatal=True,
            )
            all_issues.append(issue)
            excluded_records.append({"image_id": stem, "path": str(xml_path), "reasons": [issue.detail]})

    for stem, img_path in sorted(images_by_stem.items()):
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                width, height = img.size
                mode = img.mode
                file_format = img.format or img_path.suffix.lstrip(".").upper()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            issue = AnnotationIssue(
                image_id=stem,
                file_path=str(img_path),
                issue_type="corrupt_image",
                detail=f"Unable to decode image: {exc}",
                fatal=True,
            )
            all_issues.append(issue)
            excluded_records.append({"image_id": stem, "path": str(img_path), "reasons": [issue.detail]})
            continue

        xml_path = xmls_by_stem.get(stem)
        if xml_path is None:
            issue = AnnotationIssue(
                image_id=stem,
                file_path=str(img_path),
                issue_type="missing_annotation",
                detail="Image has no matching XML annotation file",
                fatal=True,
            )
            all_issues.append(issue)
            excluded_records.append({"image_id": stem, "path": str(img_path), "reasons": [issue.detail]})
            continue

        boxes, img_issues, _ = parse_voc_xml(
            xml_path=xml_path,
            alias_map=alias_map,
            expected_width=width,
            expected_height=height,
        )
        all_issues.extend(img_issues)
        fatal_issues = [item for item in img_issues if item.fatal]
        if fatal_issues or not boxes:
            reasons = [item.detail for item in fatal_issues] or ["No valid bounding boxes parsed"]
            excluded_records.append({"image_id": stem, "path": str(img_path), "reasons": reasons})
            continue

        valid_records.append(
            ImageRecord(
                image_id=stem,
                image_path=str(img_path),
                annotation_path=str(xml_path),
                width=width,
                height=height,
                mode=mode,
                channels=_channel_count_from_mode(mode),
                file_format=file_format,
                boxes=boxes,
            )
        )

    logger.info(
        "Parsed dataset '%s': %d valid images, %d excluded, %d total issues.",
        root,
        len(valid_records),
        len(excluded_records),
        len(all_issues),
    )
    return DatasetParseResult(
        records=valid_records,
        excluded_records=excluded_records,
        issues=all_issues,
        classes=specs,
    )


def save_records_jsonl(records: list[ImageRecord], output_path: str | Path) -> Path:
    """Persist image records to a UTF-8 JSONL manifest."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=True) + "\n")
    return dest


def load_records_jsonl(jsonl_path: str | Path) -> list[ImageRecord]:
    """Load image records from a JSONL manifest."""
    records: list[ImageRecord] = []
    with Path(jsonl_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(ImageRecord.from_dict(json.loads(stripped)))
    return records
