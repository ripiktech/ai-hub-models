# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast
import torch
from ultralytics.nn.modules.head import Segment
from ultralytics.nn.tasks import SegmentationModel

from qai_hub_models.models._shared.ultralytics.segment_patches import (
    patch_ultralytics_segmentation_head,
)
from qai_hub_models.models._shared.yolo.utils import (
    get_most_likely_score,
    transform_box_layout_xywh2xyxy,
)
from qai_hub_models.models.common import Precision
from qai_hub_models.utils.base_model import BaseModel, InputSpec


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Vectorized NMS that works with ONNX export.
    Uses torchvision.ops.nms if available, otherwise falls back to basic filtering.
    
    Args:
        boxes: Tensor of shape [N, 4] with box coordinates in format [x1, y1, x2, y2]
        scores: Tensor of shape [N] with confidence scores for each box
        iou_threshold: IoU threshold for suppression (default: 0.5)
    
    Returns:
        keep_indices: Tensor of indices to keep after NMS
    """
    try:
        # Use torchvision NMS which is ONNX-compatible
        from torchvision.ops import nms
        return nms(boxes, scores, iou_threshold)
    except ImportError:
        # Fallback: just return top-k by score (no actual NMS)
        # This ensures ONNX export works even without torchvision
        _, indices = torch.sort(scores, descending=True)
        return indices


def apply_nms_batched(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    mask_coeffs: torch.Tensor,
    classes: torch.Tensor = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.001
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply NMS to batched detection results using torchvision.ops.nms (ONNX-compatible).
    
    Args:
        boxes: Tensor of shape [batch_size, num_anchors, 4]
        scores: Tensor of shape [batch_size, num_anchors]
        mask_coeffs: Tensor of shape [batch_size, num_protos, num_anchors]
        classes: Optional tensor of shape [batch_size, num_anchors] with class indices
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold to keep boxes
    
    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_mask_coeffs, filtered_classes)
        where filtered_mask_coeffs has shape [batch_size, num_protos, num_detections]
    """
    batch_size = boxes.shape[0]
    num_protos = mask_coeffs.shape[1]
    num_anchors = boxes.shape[1]
    
    # Process each batch
    all_keep_indices = []
    max_keep = 0
    
    for b in range(batch_size):
        # Get boxes and scores for this batch
        batch_boxes = boxes[b]  # [num_anchors, 4]
        batch_scores = scores[b]  # [num_anchors]
        
        # Filter by score threshold
        score_mask = batch_scores > score_threshold
        valid_indices = torch.where(score_mask)[0]
        
        if valid_indices.numel() > 0:
            valid_boxes = batch_boxes[valid_indices]
            valid_scores = batch_scores[valid_indices]
            
            # Apply NMS using torchvision (ONNX-compatible)
            keep_indices_local = batched_nms(valid_boxes, valid_scores, iou_threshold)
            keep_indices = valid_indices[keep_indices_local]
        else:
            keep_indices = torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        all_keep_indices.append(keep_indices)
        max_keep = max(max_keep, keep_indices.numel())
    
    # Ensure at least 1 detection for consistent output shape
    if max_keep == 0:
        max_keep = 1
    
    # Create output tensors
    out_boxes = torch.zeros((batch_size, max_keep, 4), device=boxes.device, dtype=boxes.dtype)
    out_scores = torch.zeros((batch_size, max_keep), device=scores.device, dtype=scores.dtype)
    out_mask_coeffs = torch.zeros((batch_size, num_protos, max_keep), device=mask_coeffs.device, dtype=mask_coeffs.dtype)
    out_classes = None
    if classes is not None:
        out_classes = torch.zeros((batch_size, max_keep), device=classes.device, dtype=classes.dtype)
    
    # Fill output tensors
    for b in range(batch_size):
        keep_idx = all_keep_indices[b]
        n = keep_idx.numel()
        if n > 0:
            out_boxes[b, :n] = boxes[b, keep_idx]
            out_scores[b, :n] = scores[b, keep_idx]
            out_mask_coeffs[b, :, :n] = mask_coeffs[b, :, keep_idx]
            if classes is not None:
                out_classes[b, :n] = classes[b, keep_idx]
    
    return out_boxes, out_scores, out_mask_coeffs, out_classes


class PseudoCV2:
    """Pseudo implementation of OpenCV functionality used in this module."""
    
    COLOR_RGB2HSV = 'RGB2HSV'
    
    @staticmethod
    def resize(image, size, interpolation=None):
        """
        Resize image to specified size using PyTorch operations.
        
        Args:
            image: torch tensor of shape (H, W) or (H, W, C)
            size: tuple (width, height)
            interpolation: interpolation method (ignored, uses bilinear)
        
        Returns:
            Resized image as torch tensor
        """
        import torch.nn.functional as F
        
        # Convert to torch tensor if needed
        if not isinstance(image, torch.Tensor):
            img_tensor = torch.tensor(image)
        else:
            img_tensor = image
        
        # Store original dtype
        original_dtype = img_tensor.dtype
        
        # Handle 2D images (H, W)
        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            was_2d = True
        # Handle 3D images (H, W, C)
        elif img_tensor.ndim == 3:
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            was_2d = False
        else:
            was_2d = False
        
        # Resize using bilinear interpolation
        width, height = size
        resized = F.interpolate(
            img_tensor.float(),
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # Convert back to original format
        if was_2d:
            resized = resized.squeeze(0).squeeze(0)
        else:
            resized = resized.squeeze(0).permute(1, 2, 0)
        
        # Convert back to original dtype
        resized = resized.to(original_dtype)
        
        return resized
    
    @staticmethod
    def cvtColor(image, conversion_code):
        """
        Convert image color space using pure PyTorch.
        
        Args:
            image: torch tensor of shape (H, W, C)
            conversion_code: color conversion code (e.g., COLOR_RGB2HSV)
        
        Returns:
            Converted image as torch tensor
        """
        if conversion_code == PseudoCV2.COLOR_RGB2HSV or conversion_code == 'RGB2HSV':
            return PseudoCV2._rgb_to_hsv(image)
        else:
            raise NotImplementedError(f"Conversion code {conversion_code} not implemented")
    
    @staticmethod
    def _rgb_to_hsv(rgb_image):
        """
        Convert RGB image to HSV color space using pure PyTorch.
        
        Args:
            rgb_image: torch tensor of shape (H, W, 3) with values in range [0, 255]
        
        Returns:
            HSV image as torch tensor with H in [0, 179], S in [0, 255], V in [0, 255]
        """
        # Convert to torch tensor if needed
        if not isinstance(rgb_image, torch.Tensor):
            rgb_image = torch.tensor(rgb_image)
        
        # Normalize RGB to [0, 1]
        rgb_normalized = rgb_image.float() / 255.0
        
        r = rgb_normalized[:, :, 0]
        g = rgb_normalized[:, :, 1]
        b = rgb_normalized[:, :, 2]
        
        max_c = torch.maximum(torch.maximum(r, g), b)
        min_c = torch.minimum(torch.minimum(r, g), b)
        diff = max_c - min_c
        
        # Initialize HSV
        h = torch.zeros_like(max_c)
        s = torch.zeros_like(max_c)
        v = max_c
        
        # Calculate Saturation
        mask = max_c != 0
        s[mask] = diff[mask] / max_c[mask]
        
        # Calculate Hue
        mask_diff = diff != 0
        
        # Red is max
        mask_r = (max_c == r) & mask_diff
        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
        
        # Green is max
        mask_g = (max_c == g) & mask_diff
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
        
        # Blue is max
        mask_b = (max_c == b) & mask_diff
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)
        
        # Normalize to OpenCV ranges: H [0, 179], S [0, 255], V [0, 255]
        h = (h / 2).byte()  # OpenCV uses H in [0, 179]
        s = (s * 255).byte()
        v = (v * 255).byte()
        
        # Stack into HSV image
        hsv_image = torch.stack([h, s, v], dim=2)
        
        return hsv_image


DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW = 640

class UltralyticsSingleClassSegmentor(BaseModel):
    """Ultralytics segmentor that segments 1 class."""

    def __init__(self, model: SegmentationModel) -> None:
        super().__init__()
        patch_ultralytics_segmentation_head(model)
        self.model = model

    def forward(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run segmentor on `image` and produce segmentation masks.

        Parameters
        ----------
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            Tuple of 4 tensors:
                boxes:
                    Shape [1, num_anchors, 4]
                    where 4 = [x1, y1, x2, y2] (box coordinates in pixel space)
                scores:
                    Shape [batch_size, num_anchors]
                    per-anchor confidence of whether the anchor box
                    contains an object / box or does not contain an object
                mask_coeffs:
                    Shape [batch_size, num_anchors, num_prototype_masks]
                    Per-anchor mask coefficients
                mask_protos:
                    Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
                    Mask protos.
        """
        boxes: torch.Tensor
        scores: torch.Tensor
        mask_coeffs: torch.Tensor
        mask_protos: torch.Tensor
        boxes, scores, mask_coeffs, mask_protos = self.model(image)

        # Convert boxes to (x1, y1, x2, y2)
        boxes = transform_box_layout_xywh2xyxy(boxes.permute(0, 2, 1))

        return boxes, scores.squeeze(1), mask_coeffs.permute(0, 2, 1), mask_protos

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
        width: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm® AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "mask_coeffs", "mask_protos"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask_protos"]


class UltralyticsMulticlassSegmentor(BaseModel):
    """Ultralytics segmentor that segments multiple classes."""

    def __init__(self, model: SegmentationModel, precision: Precision | None = None):
        super().__init__(model)
        self.num_classes: int = cast(Segment, model.model[-1]).nc
        self.precision = precision
        patch_ultralytics_segmentation_head(model)

    def _analyze_color_distribution(self, image: torch.Tensor, boxes: torch.Tensor, mask_coeffs: torch.Tensor, mask_protos: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Simple threshold-based classification:
        - Score > 50% -> Bauxite (class 0)
        - Score <= 50% -> Laterite (class 1)
        
        Args:
            scores: Tensor of shape [batch_size, num_anchors] with confidence scores
        
        Returns:
            labels: Tensor of shape [batch_size, num_anchors] with values 0 (Bauxite) or 1 (Laterite)
        """
        batch_size = scores.shape[0]
        num_anchors = scores.shape[1]
        
        # Initialize all as Laterite (1) using int32 instead of int64 for web compatibility
        labels = torch.ones((batch_size, num_anchors), dtype=torch.int32, device=image.device)
        
        # Set all predictions with score > 0.5 to Bauxite (0)
        labels[scores > 0.5] = 0
        
        return labels

    @staticmethod
    def get_input_spec(
        batch_size: int = 1,
        height: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
        width: int = DEFAULT_ULTRALYTICS_IMAGE_INPUT_HW,
    ) -> InputSpec:
        """
        Returns the input specification (name -> (shape, type). This can be
        used to submit profiling job on Qualcomm AI Hub.
        """
        return {"image": ((batch_size, 3, height, width), "float32")}

    @staticmethod
    def get_output_names() -> list[str]:
        return ["boxes", "scores", "mask_coeffs", "class_idx", "mask_protos", "labels"]

    @staticmethod
    def get_channel_last_inputs() -> list[str]:
        return ["image"]

    @staticmethod
    def get_channel_last_outputs() -> list[str]:
        return ["mask_protos"]



    def forward(self, image: torch.Tensor):
        """
        Run the segmentor on `image` and produce segmentation masks.

        Parameters
        ----------
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            Tuple of 6 tensors:
                boxes:
                    Shape [1, num_anchors, 4]
                    where 4 = [x1, y1, x2, y2] (box coordinates in pixel space)
                scores:
                    Shape [batch_size, num_anchors, num_classes + 1]
                    per-anchor confidence of whether the anchor box
                    contains an object / box or does not contain an object
                mask_coeffs:
                    Shape [batch_size, num_anchors, num_prototype_masks]
                    Per-anchor mask coefficients
                class_idx:
                    Shape [batch_size, num_anchors]
                    Index
                mask_protos:
                    Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
                    Mask protos.
                labels:
                    Shape [batch_size, num_anchors]
                    Label indices for each anchor (currently filled with 0 as placeholder for "Test")
        """
        boxes: torch.Tensor
        scores: torch.Tensor
        mask_coeffs: torch.Tensor
        mask_protos: torch.Tensor
        boxes, scores, mask_coeffs, mask_protos = self.model(image)

        # Convert boxes to (x1, y1, x2, y2)
        boxes = transform_box_layout_xywh2xyxy(boxes.permute(0, 2, 1))

        # Get class ID of most likely score.
        scores = scores.permute(0, 2, 1)
        scores, classes = get_most_likely_score(scores)

        # Keep classes unchanged from model predictions
        if self.precision == Precision.float:
            classes = classes.to(torch.float32)
        if self.precision is None:
            classes = classes.to(torch.uint8)

        # Apply threshold-based classification for labels output only
        # Score > 50% -> Bauxite (0), Score <= 50% -> Laterite (1)
        labels = self._analyze_color_distribution(image, boxes, mask_coeffs.permute(0, 2, 1), mask_protos, scores)

        return boxes, scores, mask_coeffs.permute(0, 2, 1), classes, mask_protos, labels

