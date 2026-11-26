"""Tests for geometric and photometric bounding-box augmentations."""

from __future__ import annotations

import numpy as np

from edgevision.data.transforms import TrainingAugmentor


def test_augmentor_keeps_boxes_valid_and_grayscale_replicated() -> None:
    rgb = np.full((200, 200, 3), 100, dtype=np.uint8)
    rgb[30:80, 40:100, :] = 200
    boxes = np.array([[40.0, 30.0, 100.0, 80.0]], dtype=np.float32)
    labels = np.array([2], dtype=np.int64)

    aug = TrainingAugmentor(
        {
            "horizontal_flip_prob": 1.0,
            "vertical_flip_prob": 1.0,
            "max_rotation_deg": 5.0,
            "rotation_prob": 1.0,
            "brightness_delta": 0.1,
            "contrast_range": [0.9, 1.1],
            "gaussian_noise_std": 3.0,
            "gaussian_blur_prob": 0.5,
        },
        seed=7,
    )
    out_img, out_boxes, out_labels = aug(rgb, boxes, labels)

    assert out_img.shape == (200, 200, 3)
    assert np.array_equal(out_img[:, :, 0], out_img[:, :, 1])
    assert np.array_equal(out_img[:, :, 1], out_img[:, :, 2])
    assert out_boxes.shape[0] == 1
    assert out_labels.tolist() == [2]
    xmin, ymin, xmax, ymax = out_boxes[0]
    assert 0.0 <= xmin < xmax <= 200.0
    assert 0.0 <= ymin < ymax <= 200.0
