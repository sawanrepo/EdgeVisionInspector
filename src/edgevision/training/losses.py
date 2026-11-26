"""Class-imbalance diagnostics and optional inverse-frequency loss weighting."""

from __future__ import annotations

from collections import Counter

import torch

from edgevision.data.annotations import ImageRecord


def compute_class_instance_counts(records: list[ImageRecord], num_classes: int) -> list[int]:
    """Count bounding-box instances per canonical class ID."""
    counts = Counter()
    for rec in records:
        for box in rec.boxes:
            counts[box.class_id] += 1
    return [int(counts[cid]) for cid in range(num_classes)]


def compute_inverse_freq_weights(
    records: list[ImageRecord],
    num_classes: int,
    power: float = 0.5,
) -> torch.Tensor:
    """Compute mean-normalized smoothed inverse-frequency weights for class-imbalance ablations."""
    counts = compute_class_instance_counts(records, num_classes)
    raw = [1.0 / (max(1, c) ** power) for c in counts]
    tensor = torch.tensor(raw, dtype=torch.float32)
    return tensor / tensor.mean()
