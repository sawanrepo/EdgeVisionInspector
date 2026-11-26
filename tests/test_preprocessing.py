"""Tests for the shared preprocessing contract across PyTorch and ONNX Runtime inputs."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from edgevision.data.transforms import Preprocessor, decode_grayscale_to_rgb_array


@pytest.mark.parametrize(
    ("arch", "size"),
    [
        ("yolox_tiny", (224, 224)),
        ("yolox_tiny", (320, 320)),
        ("ssdlite320_mobilenet_v3_large", (320, 320)),
        ("faster_rcnn_resnet50_fpn", (224, 224)),
        ("faster_rcnn_resnet50_fpn", (320, 320)),
    ],
)
def test_pytorch_and_onnx_input_tensor_parity(arch: str, size: tuple[int, int], preprocess_cfg: dict) -> None:
    rng = np.random.default_rng(123)
    gray = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
    pil_img = Image.fromarray(gray, mode="L")

    pre = Preprocessor(architecture=arch, input_size=size, preprocess_cfg=preprocess_cfg)
    pt_tensor, pt_params = pre.preprocess_for_pytorch(pil_img, add_batch_dim=True)
    ort_np, ort_params = pre.preprocess_for_onnx(pil_img)

    assert pt_params == ort_params
    assert tuple(pt_tensor.shape) == (1, 3, size[0], size[1])
    assert ort_np.shape == (1, 3, size[0], size[1])
    assert ort_np.dtype == np.float32
    assert ort_np.flags["C_CONTIGUOUS"]
    assert np.array_equal(pt_tensor.numpy(), ort_np)

    # Grayscale replication produces identical values across all 3 channels.
    assert np.allclose(ort_np[0, 0], ort_np[0, 1]) or arch == "faster_rcnn_resnet50_fpn"


def test_grayscale_replication_and_box_roundtrip(preprocess_cfg: dict) -> None:
    arr = np.arange(200 * 200, dtype=np.uint8).reshape(200, 200)
    rgb = decode_grayscale_to_rgb_array(arr)
    assert rgb.shape == (200, 200, 3)
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
    assert np.array_equal(rgb[:, :, 1], rgb[:, :, 2])

    pre = Preprocessor("yolox_tiny", input_size=(320, 320), preprocess_cfg=preprocess_cfg)
    params = pre.compute_resize_params(200, 200)
    orig_boxes = np.array([[20.0, 30.0, 120.0, 150.0]], dtype=np.float32)
    scaled = pre.boxes_to_input_coords(orig_boxes, params)
    restored = pre.boxes_to_original_coords(scaled, params)
    assert np.allclose(orig_boxes, restored, atol=1e-4)
