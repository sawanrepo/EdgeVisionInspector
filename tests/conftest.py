"""Shared synthetic fixtures for fast, dataset-free unit and integration tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from edgevision.config import load_yaml, resolve_project_root
from edgevision.data.annotations import BoundingBox, ImageRecord

PROJECT_ROOT = resolve_project_root()


@pytest.fixture()
def dataset_cfg() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs/dataset.yaml")


@pytest.fixture()
def preprocess_cfg() -> dict[str, Any]:
    return load_yaml(PROJECT_ROOT / "configs/preprocess.yaml")


def write_synthetic_voc_sample(
    root_dir: Path,
    stem: str,
    width: int = 200,
    height: int = 200,
    objects: list[tuple[str, tuple[int, int, int, int]]] | None = None,
    pixel_array: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Create a synthetic grayscale image and matching Pascal VOC XML annotation."""
    img_dir = root_dir / "IMAGES"
    ann_dir = root_dir / "ANNOTATIONS"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    if pixel_array is None:
        rng = np.random.default_rng(abs(hash(stem)) % (2**32))
        arr = rng.integers(40, 220, size=(height, width), dtype=np.uint8)
    else:
        arr = pixel_array.astype(np.uint8)

    img_path = img_dir / f"{stem}.png"
    Image.fromarray(arr, mode="L").save(img_path)

    objs_xml = []
    for cls_name, (xmin, ymin, xmax, ymax) in (objects or []):
        objs_xml.append(
            f"""  <object>
    <name>{cls_name}</name>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""
        )

    xml_content = f"""<annotation>
  <filename>{stem}.png</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>1</depth>
  </size>
{chr(10).join(objs_xml)}
</annotation>
"""
    xml_path = ann_dir / f"{stem}.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    return img_path, xml_path


@pytest.fixture()
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a 30-image synthetic 6-class NEU-DET-style dataset with a known exact duplicate pair."""
    class_names = ["Crazing", "inclusion", "Patches", "Pitted surface", "Rolled-in scale", "scratches"]
    base_arr = np.full((200, 200), 120, dtype=np.uint8)
    base_arr[40:90, 40:90] = 200

    idx = 0
    for round_idx in range(5):
        for cls_name in class_names:
            stem = f"sample_{idx:03d}"
            arr = base_arr.copy() if idx in (0, 1) else None
            write_synthetic_voc_sample(
                tmp_path,
                stem=stem,
                objects=[(cls_name, (20 + round_idx * 4, 25, 80 + round_idx * 4, 95))],
                pixel_array=arr,
            )
            idx += 1
    return tmp_path


@pytest.fixture()
def sample_image_records(tmp_path: Path) -> list[ImageRecord]:
    """Create 6 synthetic ImageRecord items on disk covering all classes."""
    class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]
    records: list[ImageRecord] = []
    for cid, cname in enumerate(class_names):
        img_path = tmp_path / f"img_{cid}.png"
        arr = np.full((200, 200), 50 + cid * 25, dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(img_path)
        records.append(
            ImageRecord(
                image_id=f"img_{cid}",
                image_path=str(img_path),
                annotation_path=None,
                width=200,
                height=200,
                mode="L",
                channels=1,
                file_format="PNG",
                boxes=[BoundingBox(class_id=cid, class_name=cname, xmin=15.0, ymin=20.0, xmax=75.0, ymax=90.0)],
                duplicate_group_id=f"grp_{cid}",
            )
        )
    return records
