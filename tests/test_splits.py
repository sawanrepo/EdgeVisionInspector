"""Tests for duplicate grouping, split disjointness, leakage prevention, and evaluation views."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgevision.data.annotations import BoundingBox, ImageRecord, parse_dataset
from edgevision.data.audit import analyze_duplicates
from edgevision.data.splits import (
    save_split_manifests,
    stratified_group_split,
    validate_split_integrity,
)


def test_duplicate_grouping_and_leakage_free_split(synthetic_dataset_dir: Path, dataset_cfg: dict, tmp_path: Path) -> None:
    parse_res = parse_dataset(synthetic_dataset_dir, dataset_cfg)
    dup = analyze_duplicates(parse_res.records, phash_max_dist=4, cosine_min=0.995)
    assert dup.exact_groups, "Expected synthetic exact duplicate pair to be detected"

    split_res = stratified_group_split(
        records=parse_res.records,
        num_classes=6,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42,
    )
    part_map = {
        "train": {r.image_id for r in split_res.train},
        "val": {r.image_id for r in split_res.val},
        "test": {r.image_id for r in split_res.test},
    }
    assert not (part_map["train"] & part_map["val"])
    assert not (part_map["train"] & part_map["test"])
    assert not (part_map["val"] & part_map["test"])

    # Exact duplicate pair sample_000 & sample_001 must remain in the exact same split.
    for part_ids in part_map.values():
        if "sample_000" in part_ids or "sample_001" in part_ids:
            assert {"sample_000", "sample_001"}.issubset(part_ids)

    paths = save_split_manifests(
        split_res,
        manifests_dir=tmp_path / "manifests",
        dataset_cfg=dataset_cfg,
    )
    assert paths["splits"].exists()
    assert paths["golden_test"].exists()
    assert paths["natural_hard_subset"].exists()
    assert paths["corruption_suite"].exists()
    assert paths["checksums"].exists()


def test_validate_split_integrity_catches_group_leakage() -> None:
    r1 = ImageRecord("a", "a.png", None, 200, 200, "L", 1, "PNG", [BoundingBox(0, "crazing", 1, 1, 10, 10)], "grp_dup")
    r2 = ImageRecord("b", "b.png", None, 200, 200, "L", 1, "PNG", [BoundingBox(0, "crazing", 1, 1, 10, 10)], "grp_dup")
    with pytest.raises(ValueError, match="Duplicate group leakage"):
        validate_split_integrity({"train": [r1], "val": [r2], "test": [r2]}, num_classes=1)
