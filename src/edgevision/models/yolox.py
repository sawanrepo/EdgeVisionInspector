"""Pure-PyTorch YOLOX-Tiny implementation compatible with arbitrary square input resolutions."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """Export-friendly SiLU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """Conv2d + BatchNorm2d + SiLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, ksize: int, stride: int = 1) -> None:
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    """Standard CSP bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        conv_cls = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1)
        self.conv2 = conv_cls(hidden, out_channels, 3, stride=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.use_add else y


class CSPLayer(nn.Module):
    """C3/CSP bottleneck module with 3 convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden, 1, stride=1)
        self.conv3 = BaseConv(2 * hidden, out_channels, 1, stride=1)
        self.m = nn.Sequential(
            *[
                Bottleneck(hidden, hidden, shortcut=shortcut, expansion=1.0, depthwise=depthwise)
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.m(self.conv1(x))
        x_2 = self.conv2(x)
        return self.conv3(torch.cat((x_1, x_2), dim=1))


class Focus(nn.Module):
    """Space-to-depth Focus stem."""

    def __init__(self, in_channels: int, out_channels: int, ksize: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_left = x[..., 0::2, 0::2]
        bot_left = x[..., 1::2, 0::2]
        top_right = x[..., 0::2, 1::2]
        bot_right = x[..., 1::2, 1::2]
        return self.conv(torch.cat((top_left, bot_left, top_right, bot_right), dim=1))


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden, 1, stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
        )
        self.conv2 = BaseConv(hidden * (len(kernel_sizes) + 1), out_channels, 1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        return self.conv2(x)


class CSPDarknet(nn.Module):
    """CSPDarknet backbone producing dark3, dark4, and dark5 feature maps."""

    def __init__(
        self,
        dep_mul: float = 0.33,
        wid_mul: float = 0.375,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        conv_cls = DWConv if depthwise else BaseConv

        self.stem = Focus(3, base_channels, ksize=3)
        self.dark2 = nn.Sequential(
            conv_cls(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise),
        )
        self.dark3 = nn.Sequential(
            conv_cls(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark4 = nn.Sequential(
            conv_cls(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark5 = nn.Sequential(
            conv_cls(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.dark2(x)
        dark3 = self.dark3(x)
        dark4 = self.dark4(dark3)
        dark5 = self.dark5(dark4)
        return dark3, dark4, dark5


class YOLOPAFPN(nn.Module):
    """PAFPN feature pyramid neck for YOLOX."""

    def __init__(
        self,
        depth: float = 0.33,
        width: float = 0.375,
        in_channels: tuple[int, int, int] = (256, 512, 1024),
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = CSPDarknet(dep_mul=depth, wid_mul=width, depthwise=depthwise)
        conv_cls = DWConv if depthwise else BaseConv
        c3, c4, c5 = [int(ch * width) for ch in in_channels]

        self.lateral_conv0 = BaseConv(c5, c4, 1, 1)
        self.C3_p4 = CSPLayer(2 * c4, c4, round(3 * depth), shortcut=False, depthwise=depthwise)

        self.reduce_conv1 = BaseConv(c4, c3, 1, 1)
        self.C3_p3 = CSPLayer(2 * c3, c3, round(3 * depth), shortcut=False, depthwise=depthwise)

        self.bu_conv2 = conv_cls(c3, c3, 3, 2)
        self.C3_n3 = CSPLayer(2 * c3, c4, round(3 * depth), shortcut=False, depthwise=depthwise)

        self.bu_conv1 = conv_cls(c4, c4, 3, 2)
        self.C3_n4 = CSPLayer(2 * c4, c5, round(3 * depth), shortcut=False, depthwise=depthwise)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x2, x1, x0 = self.backbone(x)

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = F.interpolate(fpn_out0, scale_factor=2.0, mode="nearest")
        f_out0 = torch.cat([f_out0, x1], dim=1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = F.interpolate(fpn_out1, scale_factor=2.0, mode="nearest")
        f_out1 = torch.cat([f_out1, x2], dim=1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], dim=1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], dim=1)
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    """Decoupled anchor-free YOLOX detection head."""

    def __init__(
        self,
        num_classes: int,
        width: float = 0.375,
        strides: tuple[int, int, int] = (8, 16, 32),
        in_channels: tuple[int, int, int] = (256, 512, 1024),
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        conv_cls = DWConv if depthwise else BaseConv
        feat_channels = int(256 * width)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for ch in in_channels:
            in_ch = int(ch * width)
            self.stems.append(BaseConv(in_ch, feat_channels, ksize=1, stride=1))
            self.cls_convs.append(
                nn.Sequential(
                    conv_cls(feat_channels, feat_channels, ksize=3, stride=1),
                    conv_cls(feat_channels, feat_channels, ksize=3, stride=1),
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    conv_cls(feat_channels, feat_channels, ksize=3, stride=1),
                    conv_cls(feat_channels, feat_channels, ksize=3, stride=1),
                )
            )
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, kernel_size=1, stride=1))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, kernel_size=1, stride=1))
            self.obj_preds.append(nn.Conv2d(feat_channels, 1, kernel_size=1, stride=1))

        self._initialize_biases(prior_prob=1e-2)

    def _initialize_biases(self, prior_prob: float) -> None:
        bias_val = -math.log((1.0 - prior_prob) / prior_prob)
        for conv in self.cls_preds:
            nn.init.constant_(conv.bias, bias_val)
        for conv in self.obj_preds:
            nn.init.constant_(conv.bias, bias_val)

    def forward_raw(self, feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return concatenated raw predictions [B, N_anchors, 5 + num_classes] in [cx, cy, w, h, obj, cls...]."""
        outputs: list[torch.Tensor] = []
        for idx, x in enumerate(feats):
            stem = self.stems[idx](x)
            cls_feat = self.cls_convs[idx](stem)
            reg_feat = self.reg_convs[idx](stem)

            cls_out = self.cls_preds[idx](cls_feat)
            reg_out = self.reg_preds[idx](reg_feat)
            obj_out = self.obj_preds[idx](reg_feat)

            # [B, 4 + 1 + C, H, W] -> [B, H*W, 5 + C]
            level_out = torch.cat([reg_out, obj_out, cls_out], dim=1)
            bsz, ch, h, w = level_out.shape
            outputs.append(level_out.permute(0, 2, 3, 1).reshape(bsz, h * w, ch))
        return torch.cat(outputs, dim=1)


def _iou_cxcywh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between [N, 4] and [M, 4] cxcywh boxes."""
    px1 = pred[:, 0:1] - pred[:, 2:3] * 0.5
    py1 = pred[:, 1:2] - pred[:, 3:4] * 0.5
    px2 = pred[:, 0:1] + pred[:, 2:3] * 0.5
    py2 = pred[:, 1:2] + pred[:, 3:4] * 0.5

    tx1 = target[:, 0:1].T - target[:, 2:3].T * 0.5
    ty1 = target[:, 1:2].T - target[:, 3:4].T * 0.5
    tx2 = target[:, 0:1].T + target[:, 2:3].T * 0.5
    ty2 = target[:, 1:2].T + target[:, 3:4].T * 0.5

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)

    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = torch.clamp(area_p + area_t - inter, min=1e-7)
    return inter / union


def _iou_loss_aligned(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Element-wise 1 - IoU^2 loss for matched [K, 4] cxcywh pairs."""
    px1, py1 = pred[:, 0] - pred[:, 2] * 0.5, pred[:, 1] - pred[:, 3] * 0.5
    px2, py2 = pred[:, 0] + pred[:, 2] * 0.5, pred[:, 1] + pred[:, 3] * 0.5
    tx1, ty1 = target[:, 0] - target[:, 2] * 0.5, target[:, 1] - target[:, 3] * 0.5
    tx2, ty2 = target[:, 0] + target[:, 2] * 0.5, target[:, 1] + target[:, 3] * 0.5

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    area_p = torch.clamp(px2 - px1, min=0.0) * torch.clamp(py2 - py1, min=0.0)
    area_t = torch.clamp(tx2 - tx1, min=0.0) * torch.clamp(ty2 - ty1, min=0.0)
    union = torch.clamp(area_p + area_t - inter, min=1e-7)
    iou = inter / union
    return 1.0 - iou * iou


class YOLOXModel(nn.Module):
    """YOLOX detector exposing raw head outputs for external decode/NMS and SimOTA training loss."""

    def __init__(
        self,
        num_classes: int = 6,
        depth: float = 0.33,
        width: float = 0.375,
        strides: tuple[int, int, int] = (8, 16, 32),
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.backbone = YOLOPAFPN(depth=depth, width=width, depthwise=depthwise)
        self.head = YOLOXHead(
            num_classes=num_classes,
            width=width,
            strides=strides,
            depthwise=depthwise,
        )

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning un-decoded head tensor [B, N_anchors, 5 + num_classes]."""
        feats = self.backbone(x)
        return self.head.forward_raw(feats)

    def build_grid_and_strides(
        self,
        input_h: int,
        input_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct flat grid center offsets [1, A, 2] and stride values [1, A, 1]."""
        grids: list[torch.Tensor] = []
        strides_list: list[torch.Tensor] = []
        for stride in self.strides:
            fh = input_h // stride
            fw = input_w // stride
            yv, xv = torch.meshgrid(
                torch.arange(fh, device=device, dtype=dtype),
                torch.arange(fw, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((xv, yv), dim=2).reshape(1, fh * fw, 2)
            grids.append(grid)
            strides_list.append(torch.full((1, fh * fw, 1), float(stride), device=device, dtype=dtype))
        return torch.cat(grids, dim=1), torch.cat(strides_list, dim=1)

    def decode_raw(self, raw: torch.Tensor, input_h: int, input_w: int) -> torch.Tensor:
        """Decode raw head tensor into [B, A, 5 + C] with cx,cy,w,h in input pixels and sigmoid probs."""
        grid, stride = self.build_grid_and_strides(input_h, input_w, raw.device, raw.dtype)
        xy = (raw[..., :2] + grid) * stride
        wh = torch.exp(torch.clamp(raw[..., 2:4], max=10.0)) * stride
        obj = torch.sigmoid(raw[..., 4:5])
        cls = torch.sigmoid(raw[..., 5:])
        return torch.cat([xy, wh, obj, cls], dim=-1)

    def compute_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        class_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute SimOTA-assigned YOLOX detection loss (IoU + objectness + classification)."""
        raw = self.forward_raw(images)
        _, _, h, w = images.shape
        grid, stride = self.build_grid_and_strides(h, w, raw.device, raw.dtype)

        decoded_xy = (raw[..., :2] + grid) * stride
        decoded_wh = torch.exp(torch.clamp(raw[..., 2:4], max=10.0)) * stride
        decoded_boxes = torch.cat([decoded_xy, decoded_wh], dim=-1)

        total_iou = raw.new_tensor(0.0)
        total_obj = raw.new_tensor(0.0)
        total_cls = raw.new_tensor(0.0)
        total_fg = 0

        grid_centers = (grid[0] + 0.5) * stride[0]  # [A, 2]

        for b_idx, tgt in enumerate(targets):
            gt_xyxy = tgt["boxes"].to(raw.device)
            gt_labels = tgt["labels"].to(raw.device)
            num_gt = gt_xyxy.shape[0]
            num_anchors = raw.shape[1]

            obj_target = raw.new_zeros((num_anchors, 1))
            if num_gt == 0:
                total_obj = total_obj + F.binary_cross_entropy_with_logits(
                    raw[b_idx, :, 4:5], obj_target, reduction="sum"
                )
                continue

            gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) * 0.5
            gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) * 0.5
            gt_w = torch.clamp(gt_xyxy[:, 2] - gt_xyxy[:, 0], min=1.0)
            gt_h = torch.clamp(gt_xyxy[:, 3] - gt_xyxy[:, 1], min=1.0)
            gt_cxcywh = torch.stack([gt_cx, gt_cy, gt_w, gt_h], dim=1)

            in_box = (
                (grid_centers[:, 0:1] >= gt_xyxy[:, 0:1].T)
                & (grid_centers[:, 0:1] <= gt_xyxy[:, 2:3].T)
                & (grid_centers[:, 1:2] >= gt_xyxy[:, 1:2].T)
                & (grid_centers[:, 1:2] <= gt_xyxy[:, 3:4].T)
            )
            center_radius = 2.5 * stride[0]
            in_center = (
                (torch.abs(grid_centers[:, 0:1] - gt_cx.unsqueeze(0)) <= center_radius)
                & (torch.abs(grid_centers[:, 1:2] - gt_cy.unsqueeze(0)) <= center_radius)
            )
            fg_mask = (in_box | in_center).any(dim=1)
            if not fg_mask.any():
                # Fallback: assign nearest grid center per GT.
                dists = torch.cdist(grid_centers, gt_cxcywh[:, :2])
                nearest_idx = torch.argmin(dists, dim=0)
                fg_mask[nearest_idx] = True

            fg_indices = torch.where(fg_mask)[0]
            pair_ious = _iou_cxcywh(decoded_boxes[b_idx, fg_indices].detach(), gt_cxcywh)
            cls_logits_fg = raw[b_idx, fg_indices, 5:]
            obj_logits_fg = raw[b_idx, fg_indices, 4:5]
            cls_probs_fg = torch.sigmoid(cls_logits_fg) * torch.sigmoid(obj_logits_fg)
            gt_onehot = F.one_hot(gt_labels, num_classes=self.num_classes).to(raw.dtype)

            # Pairwise classification + IoU cost for simplified SimOTA matching.
            cls_cost = F.binary_cross_entropy(
                torch.clamp(cls_probs_fg.unsqueeze(1).expand(-1, num_gt, -1), 1e-6, 1.0 - 1e-6),
                gt_onehot.unsqueeze(0).expand(fg_indices.shape[0], -1, -1),
                reduction="none",
            ).sum(dim=-1)
            iou_cost = -torch.log(pair_ious + 1e-8)
            in_both = in_box[fg_indices] & in_center[fg_indices]
            cost = cls_cost + 3.0 * iou_cost + (~in_both).to(raw.dtype) * 100.0

            # Dynamic k per GT based on top-k IoU sum.
            matching = torch.zeros_like(cost, dtype=torch.bool)
            topk_k = min(10, fg_indices.shape[0])
            top_ious, _ = torch.topk(pair_ious, k=topk_k, dim=0)
            dynamic_ks = torch.clamp(top_ious.sum(dim=0).long(), min=1)
            for g_idx in range(num_gt):
                k_g = min(int(dynamic_ks[g_idx].item()), fg_indices.shape[0])
                _, pos_rows = torch.topk(cost[:, g_idx], k=k_g, largest=False)
                matching[pos_rows, g_idx] = True

            # Resolve anchors matched to multiple GTs by minimum cost.
            multi = matching.sum(dim=1) > 1
            if multi.any():
                min_gt = torch.argmin(cost[multi], dim=1)
                matching[multi] = False
                matching[multi, min_gt] = True

            matched_anchor_rel, matched_gt_idx = torch.where(matching)
            if matched_anchor_rel.numel() == 0:
                best_row = torch.argmin(cost, dim=0)
                matched_anchor_rel = best_row
                matched_gt_idx = torch.arange(num_gt, device=raw.device)

            pos_anchors = fg_indices[matched_anchor_rel]
            obj_target[pos_anchors] = 1.0

            matched_pred_boxes = decoded_boxes[b_idx, pos_anchors]
            matched_gt_boxes = gt_cxcywh[matched_gt_idx]
            iou_loss = _iou_loss_aligned(matched_pred_boxes, matched_gt_boxes).sum()

            cls_targets = gt_onehot[matched_gt_idx] * pair_ious[matched_anchor_rel, matched_gt_idx].unsqueeze(-1)
            cls_loss_elem = F.binary_cross_entropy_with_logits(
                raw[b_idx, pos_anchors, 5:], cls_targets, reduction="none"
            )
            if class_weights is not None:
                sample_w = class_weights.to(raw.device)[gt_labels[matched_gt_idx]].unsqueeze(-1)
                cls_loss_elem = cls_loss_elem * sample_w
            cls_loss = cls_loss_elem.sum()

            obj_loss = F.binary_cross_entropy_with_logits(raw[b_idx, :, 4:5], obj_target, reduction="sum")

            total_iou = total_iou + 5.0 * iou_loss
            total_cls = total_cls + cls_loss
            total_obj = total_obj + obj_loss
            total_fg += int(pos_anchors.numel())

        denom = float(max(total_fg, 1))
        loss_iou = total_iou / denom
        loss_obj = total_obj / denom
        loss_cls = total_cls / denom
        total = loss_iou + loss_obj + loss_cls
        return {
            "total_loss": total,
            "loss_iou": loss_iou,
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
        }


def load_megvii_yolox_pretrained(
    model: YOLOXModel,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Load backbone/neck/stem weights from an official Megvii YOLOX-Tiny checkpoint into our 6-class model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model_state = model.state_dict()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, val in state_dict.items():
        if key in model_state and model_state[key].shape == val.shape:
            model_state[key] = val
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)
    return {"loaded_count": len(loaded_keys), "skipped_count": len(skipped_keys), "skipped_keys": skipped_keys}
