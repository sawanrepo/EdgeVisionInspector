"""Unified preprocessing contract and training-only bounding-box augmentations."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from edgevision.config import load_yaml


@dataclass(frozen=True)
class ResizeParams:
    """Geometric metadata mapping between original image space and model input space."""

    orig_width: int
    orig_height: int
    target_width: int
    target_height: int
    scale_x: float
    scale_y: float
    pad_left: float
    pad_top: float
    resize_mode: str


def decode_grayscale_to_rgb_array(source: bytes | str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Decode any supported input into a 3-channel RGB uint8 array via grayscale replication."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            gray = arr.astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            gray = arr[:, :, 0].astype(np.uint8)
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            pil = Image.fromarray(arr[:, :, :3].astype(np.uint8)).convert("L")
            gray = np.asarray(pil, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported image array shape: {arr.shape}")
        return np.repeat(gray[:, :, None], 3, axis=2)

    if isinstance(source, (bytes, bytearray)):
        with Image.open(io.BytesIO(source)) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            gray = np.asarray(img.convert("L"), dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    if isinstance(source, Image.Image):
        gray = np.asarray(source.convert("L"), dtype=np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    raise TypeError(f"Unsupported image source type: {type(source)!r}")


def _pil_resample(name: str) -> Image.Resampling:
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }
    key = name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported interpolation mode: {name}")
    return mapping[key]


class Preprocessor:
    """Single source of truth preprocessor backed by configs/preprocess.yaml."""

    def __init__(
        self,
        architecture: str,
        input_size: tuple[int, int] | list[int] | None = None,
        preprocess_cfg: dict[str, Any] | str | Path = "configs/preprocess.yaml",
    ) -> None:
        self.cfg = load_yaml(preprocess_cfg) if isinstance(preprocess_cfg, (str, Path)) else preprocess_cfg
        archs = self.cfg.get("architectures", {})
        if architecture not in archs:
            raise KeyError(f"Architecture '{architecture}' missing from preprocess.yaml")

        self.architecture = architecture
        self.arch_cfg = archs[architecture]
        target = tuple(int(v) for v in (input_size or self.arch_cfg["default_input_size"]))
        if len(target) != 2:
            raise ValueError(f"input_size must be (H, W), got {target}")

        supported = [tuple(int(v) for v in pair) for pair in self.arch_cfg.get("supported_input_sizes", [])]
        if supported and target not in supported:
            raise ValueError(
                f"Input size {target} is not supported for '{architecture}'. Allowed: {supported}"
            )

        self.target_height, self.target_width = target
        self.resize_mode = str(self.arch_cfg["resize_mode"])
        self.interpolation = str(self.arch_cfg.get("interpolation", "bilinear"))
        self.pad_value = float(self.arch_cfg.get("pad_value", 0.0))
        self.pixel_scale = float(self.arch_cfg.get("pixel_scale", 1.0))
        self.mean = np.asarray(self.arch_cfg.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(self.arch_cfg.get("std", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(1, 1, 3)
        self.grayscale_to_rgb = bool(self.cfg.get("grayscale_to_rgb", True))
        self.channel_order = str(self.cfg.get("channel_order", "RGB"))

    def compute_resize_params(self, orig_width: int, orig_height: int) -> ResizeParams:
        """Compute deterministic resize/padding parameters for an input resolution."""
        if orig_width <= 0 or orig_height <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_width}x{orig_height}")
        if self.resize_mode == "letterbox":
            scale = min(self.target_width / orig_width, self.target_height / orig_height)
            new_w = int(round(orig_width * scale))
            new_h = int(round(orig_height * scale))
            pad_left = float((self.target_width - new_w) // 2)
            pad_top = float((self.target_height - new_h) // 2)
            return ResizeParams(
                orig_width=orig_width,
                orig_height=orig_height,
                target_width=self.target_width,
                target_height=self.target_height,
                scale_x=new_w / float(orig_width),
                scale_y=new_h / float(orig_height),
                pad_left=pad_left,
                pad_top=pad_top,
                resize_mode=self.resize_mode,
            )
        if self.resize_mode == "direct":
            return ResizeParams(
                orig_width=orig_width,
                orig_height=orig_height,
                target_width=self.target_width,
                target_height=self.target_height,
                scale_x=self.target_width / float(orig_width),
                scale_y=self.target_height / float(orig_height),
                pad_left=0.0,
                pad_top=0.0,
                resize_mode=self.resize_mode,
            )
        raise ValueError(f"Unsupported resize_mode '{self.resize_mode}'")

    def preprocess_array(
        self,
        source: bytes | str | Path | Image.Image | np.ndarray,
        add_batch_dim: bool = True,
    ) -> tuple[np.ndarray, ResizeParams]:
        """Convert image source into contiguous float32 NCHW (or CHW) NumPy array."""
        rgb = decode_grayscale_to_rgb_array(source)
        orig_h, orig_w = rgb.shape[:2]
        params = self.compute_resize_params(orig_w, orig_h)
        resample = _pil_resample(self.interpolation)

        if self.resize_mode == "letterbox":
            new_w = max(1, int(round(orig_w * params.scale_x)))
            new_h = max(1, int(round(orig_h * params.scale_y)))
            resized = np.asarray(
                Image.fromarray(rgb, mode="RGB").resize((new_w, new_h), resample),
                dtype=np.float32,
            )
            canvas = np.full(
                (self.target_height, self.target_width, 3),
                fill_value=self.pad_value,
                dtype=np.float32,
            )
            left = int(params.pad_left)
            top = int(params.pad_top)
            canvas[top : top + new_h, left : left + new_w, :] = resized
            img_f32 = canvas
        else:
            resized = Image.fromarray(rgb, mode="RGB").resize(
                (self.target_width, self.target_height), resample
            )
            img_f32 = np.asarray(resized, dtype=np.float32)

        if self.channel_order.upper() == "BGR":
            img_f32 = img_f32[:, :, ::-1]

        normalized = (img_f32 * self.pixel_scale - self.mean) / self.std
        chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32, copy=False)
        if add_batch_dim:
            chw = chw[None, ...]
        return np.ascontiguousarray(chw, dtype=np.float32), params

    def preprocess_for_onnx(
        self,
        source: bytes | str | Path | Image.Image | np.ndarray,
    ) -> tuple[np.ndarray, ResizeParams]:
        """Return contiguous batch-1 NCHW float32 array for ONNX Runtime."""
        return self.preprocess_array(source, add_batch_dim=True)

    def preprocess_for_pytorch(
        self,
        source: bytes | str | Path | Image.Image | np.ndarray,
        add_batch_dim: bool = True,
    ):
        """Return PyTorch float32 tensor with identical values, shape, and layout as ONNX input."""
        import torch

        arr, params = self.preprocess_array(source, add_batch_dim=add_batch_dim)
        return torch.from_numpy(arr.copy()), params

    @staticmethod
    def boxes_to_input_coords(boxes_xyxy: np.ndarray, params: ResizeParams) -> np.ndarray:
        """Transform xyxy boxes from native image pixels into model input coordinates."""
        arr = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
        if arr.size == 0:
            return arr
        arr[:, [0, 2]] = arr[:, [0, 2]] * params.scale_x + params.pad_left
        arr[:, [1, 3]] = arr[:, [1, 3]] * params.scale_y + params.pad_top
        arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0.0, float(params.target_width))
        arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0.0, float(params.target_height))
        return arr

    @staticmethod
    def boxes_to_original_coords(boxes_xyxy: np.ndarray, params: ResizeParams) -> np.ndarray:
        """Transform xyxy boxes from model input coordinates back to native image pixels."""
        arr = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
        if arr.size == 0:
            return arr
        arr[:, [0, 2]] = (arr[:, [0, 2]] - params.pad_left) / max(params.scale_x, 1e-8)
        arr[:, [1, 3]] = (arr[:, [1, 3]] - params.pad_top) / max(params.scale_y, 1e-8)
        arr[:, [0, 2]] = np.clip(arr[:, [0, 2]], 0.0, float(params.orig_width))
        arr[:, [1, 3]] = np.clip(arr[:, [1, 3]], 0.0, float(params.orig_height))
        return arr


class TrainingAugmentor:
    """Conservative, reproducible training-only augmentations with exact box updates."""

    def __init__(self, aug_cfg: dict[str, Any] | None = None, seed: int = 42) -> None:
        cfg = aug_cfg or {}
        self.hflip_prob = float(cfg.get("horizontal_flip_prob", 0.5))
        self.vflip_prob = float(cfg.get("vertical_flip_prob", 0.2))
        self.max_rotation_deg = float(cfg.get("max_rotation_deg", 5.0))
        self.rotation_prob = float(cfg.get("rotation_prob", 0.3))
        self.brightness_delta = float(cfg.get("brightness_delta", 0.15))
        contrast = cfg.get("contrast_range", [0.85, 1.15])
        self.contrast_min = float(contrast[0])
        self.contrast_max = float(contrast[1])
        self.noise_std = float(cfg.get("gaussian_noise_std", 5.0))
        self.blur_prob = float(cfg.get("gaussian_blur_prob", 0.15))
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        rgb_uint8: np.ndarray,
        boxes_xyxy: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply geometric and photometric augmentations while keeping boxes valid."""
        img = rgb_uint8.copy()
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
        cls_ids = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        h, w = img.shape[:2]

        if self.rng.random() < self.hflip_prob:
            img = img[:, ::-1, :]
            if boxes.shape[0] > 0:
                old_xmin = boxes[:, 0].copy()
                boxes[:, 0] = w - boxes[:, 2]
                boxes[:, 2] = w - old_xmin

        if self.rng.random() < self.vflip_prob:
            img = img[::-1, :, :]
            if boxes.shape[0] > 0:
                old_ymin = boxes[:, 1].copy()
                boxes[:, 1] = h - boxes[:, 3]
                boxes[:, 3] = h - old_ymin

        if self.max_rotation_deg > 0 and self.rng.random() < self.rotation_prob:
            angle = float(self.rng.uniform(-self.max_rotation_deg, self.max_rotation_deg))
            img, boxes, cls_ids = self._rotate_image_and_boxes(img, boxes, cls_ids, angle)

        # Photometric grayscale-consistent jitter (preserves R == G == B).
        gray = img[:, :, 0].astype(np.float32)
        if self.contrast_max > self.contrast_min:
            alpha = float(self.rng.uniform(self.contrast_min, self.contrast_max))
            mean_val = float(np.mean(gray))
            gray = (gray - mean_val) * alpha + mean_val
        if self.brightness_delta > 0:
            shift = float(self.rng.uniform(-self.brightness_delta, self.brightness_delta)) * 255.0
            gray = gray + shift
        if self.noise_std > 0 and self.rng.random() < 0.5:
            gray = gray + self.rng.normal(0.0, self.noise_std, size=gray.shape).astype(np.float32)
        gray_u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)

        if self.blur_prob > 0 and self.rng.random() < self.blur_prob:
            pil_blur = Image.fromarray(gray_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=0.8))
            gray_u8 = np.asarray(pil_blur, dtype=np.uint8)

        img = np.repeat(gray_u8[:, :, None], 3, axis=2)
        return img, boxes, cls_ids

    @staticmethod
    def _rotate_image_and_boxes(
        img: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        angle_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        pil = Image.fromarray(img, mode="RGB").rotate(
            angle_deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(114, 114, 114)
        )
        rot_img = np.asarray(pil, dtype=np.uint8)
        if boxes.shape[0] == 0:
            return rot_img, boxes, labels

        theta = np.deg2rad(-angle_deg)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
        cx, cy = w / 2.0, h / 2.0

        kept_boxes: list[list[float]] = []
        kept_labels: list[int] = []
        for box, cls_id in zip(boxes, labels, strict=False):
            corners = np.array(
                [
                    [box[0], box[1]],
                    [box[2], box[1]],
                    [box[2], box[3]],
                    [box[0], box[3]],
                ],
                dtype=np.float32,
            )
            shifted = corners - np.array([cx, cy], dtype=np.float32)
            rx = shifted[:, 0] * cos_t - shifted[:, 1] * sin_t + cx
            ry = shifted[:, 0] * sin_t + shifted[:, 1] * cos_t + cy
            nxmin = float(np.clip(np.min(rx), 0.0, float(w)))
            nymin = float(np.clip(np.min(ry), 0.0, float(h)))
            nxmax = float(np.clip(np.max(rx), 0.0, float(w)))
            nymax = float(np.clip(np.max(ry), 0.0, float(h)))
            if nxmax > nxmin + 1.0 and nymax > nymin + 1.0:
                kept_boxes.append([nxmin, nymin, nxmax, nymax])
                kept_labels.append(int(cls_id))

        if not kept_boxes:
            return img, boxes, labels
        return rot_img, np.asarray(kept_boxes, dtype=np.float32), np.asarray(kept_labels, dtype=np.int64)
