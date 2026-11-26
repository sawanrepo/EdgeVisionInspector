"""Tests for Pascal VOC annotation parsing, class mapping, and invalid-box flagging."""

from __future__ import annotations

from pathlib import Path

from conftest import write_synthetic_voc_sample
from edgevision.data.annotations import build_class_lookup, parse_dataset, parse_voc_xml


def test_class_alias_mapping(dataset_cfg: dict) -> None:
    specs, alias_map = build_class_lookup(dataset_cfg)
    assert len(specs) == 6
    assert alias_map["crazing"].class_id == 0
    assert alias_map["pitted surface"].class_id == 3
    assert alias_map["rolled-in scale"].class_id == 4


def test_parse_valid_and_invalid_annotations(tmp_path: Path, dataset_cfg: dict) -> None:
    _, alias_map = build_class_lookup(dataset_cfg)

    # Valid annotation
    _, valid_xml = write_synthetic_voc_sample(
        tmp_path, "valid_01", objects=[("Pitted surface", (10, 20, 60, 80))]
    )
    boxes, issues, _ = parse_voc_xml(valid_xml, alias_map, expected_width=200, expected_height=200)
    assert len(boxes) == 1
    assert boxes[0].class_id == 3
    assert boxes[0].class_name == "pitted_surface"
    assert not issues

    # Invalid boxes: non-positive width, out-of-bounds, unknown class, duplicate box
    _, bad_xml = write_synthetic_voc_sample(
        tmp_path,
        "bad_01",
        objects=[
            ("Crazing", (50, 20, 50, 80)),
            ("Crazing", (10, 10, 220, 100)),
            ("unknown_defect", (10, 10, 40, 40)),
            ("Scratches", (15, 15, 45, 45)),
            ("Scratches", (15, 15, 45, 45)),
        ],
    )
    bad_boxes, bad_issues, _ = parse_voc_xml(bad_xml, alias_map, expected_width=200, expected_height=200)
    issue_types = {i.issue_type for i in bad_issues}
    assert {"non_positive_box", "out_of_bounds", "unknown_class", "duplicate_box"}.issubset(issue_types)
    assert len(bad_boxes) == 1  # Only the first valid Scratches box kept; duplicate flagged


def test_parse_dataset_excludes_corrupt_and_empty(tmp_path: Path, dataset_cfg: dict) -> None:
    write_synthetic_voc_sample(tmp_path, "good_01", objects=[("Inclusion", (10, 10, 50, 50))])
    write_synthetic_voc_sample(tmp_path, "empty_01", objects=[])
    corrupt_img = tmp_path / "IMAGES" / "corrupt_01.png"
    corrupt_img.write_bytes(b"not_a_real_image")

    res = parse_dataset(tmp_path, dataset_cfg)
    assert [r.image_id for r in res.records] == ["good_01"]
    excluded_ids = {item["image_id"] for item in res.excluded_records}
    assert {"empty_01", "corrupt_01"}.issubset(excluded_ids)
