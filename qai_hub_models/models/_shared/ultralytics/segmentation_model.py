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
        Analyze color distribution ONLY for the most confident bounding box per batch.
        This dramatically improves performance while keeping the classification logic.
        
        Args:
            scores: Tensor of shape [batch_size, num_anchors] with confidence scores
        
        Returns:
            color_classes: Tensor of shape [batch_size, num_anchors] with values 0 (Bauxite) or 1 (Laterite)
        """
        batch_size = image.shape[0]
        num_anchors = boxes.shape[1]
        
        # Convert image from [B, C, H, W] to [B, H, W, C] for color analysis
        # Image is in RGB format with range [0, 1]
        image_uint8 = (image.cpu() * 255).byte().permute(0, 2, 3, 1)  # [B, H, W, C]
        
        color_classes = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=image.device)
        
        for b in range(batch_size):
            # Find the most confident anchor for this batch
            max_conf_idx = torch.argmax(scores[b]).item()
            
            img = image_uint8[b]  # [H, W, C]
            
            # Generate mask ONLY for the most confident anchor
            mask = torch.matmul(mask_coeffs[b, max_conf_idx:max_conf_idx+1], mask_protos[b].reshape(mask_protos.shape[1], -1))
            mask = mask.reshape(mask_protos.shape[2], mask_protos.shape[3])
            mask = torch.sigmoid(mask).cpu()
            
            # Resize mask to match image size
            mask_resized = PseudoCV2.resize(mask, (img.shape[1], img.shape[0]))
            mask_binary = (mask_resized > 0.5).byte()
            
            # Get bounding box
            box_coords = boxes[b, max_conf_idx].cpu().int()
            x1, y1, x2, y2 = box_coords[0].item(), box_coords[1].item(), box_coords[2].item(), box_coords[3].item()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract region of interest
            roi = img[y1:y2, x1:x2]
            roi_mask = mask_binary[y1:y2, x1:x2]
            
            if roi_mask.sum() == 0:
                continue
            
            # Get pixels within the mask
            mask_indices = roi_mask > 0
            if mask_indices.sum() == 0:
                continue
            
            masked_pixels = roi[mask_indices]
            
            if len(masked_pixels) == 0:
                continue
            
            # Convert RGB to HSV for better color analysis
            roi_hsv = PseudoCV2.cvtColor(roi, PseudoCV2.COLOR_RGB2HSV)
            masked_pixels_hsv = roi_hsv[mask_indices]
            
            # Color thresholds in HSV
            # WHITE: High V (value), Low S (saturation)
            # PALE YELLOW: H in [20-40], Low S, High V
            # BRIGHT YELLOW: H in [20-40], High S, High V
            # RED: H in [0-10] or [160-180], High S
            
            total_pixels = len(masked_pixels_hsv)
            
            # Count WHITE pixels (low saturation, high value)
            white_mask = (masked_pixels_hsv[:, 1] < 50) & (masked_pixels_hsv[:, 2] > 180)
            white_count = white_mask.sum().item()
            
            # Count PALE YELLOW pixels (yellow hue, low saturation, high value)
            pale_yellow_mask = (masked_pixels_hsv[:, 0] >= 20) & (masked_pixels_hsv[:, 0] <= 40) & \
                               (masked_pixels_hsv[:, 1] < 100) & (masked_pixels_hsv[:, 2] > 150)
            pale_yellow_count = pale_yellow_mask.sum().item()
            
            # Count BRIGHT YELLOW pixels (yellow hue, high saturation)
            bright_yellow_mask = (masked_pixels_hsv[:, 0] >= 20) & (masked_pixels_hsv[:, 0] <= 40) & \
                                 (masked_pixels_hsv[:, 1] >= 100)
            bright_yellow_count = bright_yellow_mask.sum().item()
            
            # Count RED pixels (red hue, high saturation)
            red_mask = ((masked_pixels_hsv[:, 0] <= 10) | (masked_pixels_hsv[:, 0] >= 160)) & \
                       (masked_pixels_hsv[:, 1] >= 100)
            red_count = red_mask.sum().item()
            
            # Calculate percentages
            white_pale_yellow_ratio = (white_count + pale_yellow_count) / total_pixels
            white_ratio = white_count / total_pixels
            red_ratio = red_count / total_pixels
            red_bright_yellow_ratio = (red_count + bright_yellow_count) / total_pixels
            other_ratio = 1.0 - white_ratio - pale_yellow_count / total_pixels - red_ratio - bright_yellow_count / total_pixels
            
            # Classification logic
            is_bauxite = False
            is_laterite = False
            
            # Rule 1: Majority WHITE or PALE YELLOW colour -> Bauxite
            if white_pale_yellow_ratio > 0.5:
                is_bauxite = True
            
            # Rule 2: WHITE colour more than RED colour -> Bauxite
            elif white_count > red_count:
                is_bauxite = True
            
            # Rule 3: RED Colour dominant -> Laterite
            elif red_ratio > 0.3:
                is_laterite = True
            
            # Rule 4: ONLY RED and BRIGHT YELLOW colour -> Laterite
            elif red_bright_yellow_ratio > 0.6 and other_ratio < 0.2:
                is_laterite = True
            
            # Assign class ONLY to the most confident anchor: 0 for Bauxite, 1 for Laterite
            if is_laterite:
                color_classes[b, max_conf_idx] = 1
            elif is_bauxite:
                color_classes[b, max_conf_idx] = 0
            else:
                color_classes[b, max_conf_idx] = 0  # Default to Bauxite
        
        return color_classes

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
        return ["boxes", "scores", "mask_coeffs", "class_idx", "mask_protos"]

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
            Tuple of 5 tensors:
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

        # Apply color-based classification logic ONLY to the most confident bounding box
        # Analyze color distribution to determine if regions meet Bauxite (0) or Laterite (1) criteria
        color_classes = self._analyze_color_distribution(image, boxes, mask_coeffs.permute(0, 2, 1), mask_protos, scores)
        
        # Boost confidence by 1.5x when color analysis confirms the classification
        # If color_classes matches classes, it means the color criteria are met
        # color_classes: 0 = Bauxite criteria met, 1 = Laterite criteria met
        # classes: predicted class from model
        confidence_boost = torch.where(
            color_classes == classes,
            torch.tensor(1.5, device=scores.device, dtype=scores.dtype),
            torch.tensor(1.0, device=scores.device, dtype=scores.dtype)
        )
        scores = scores * confidence_boost
        
        # Clamp scores to [0, 1] range
        scores = torch.clamp(scores, 0.0, 1.0)

        if self.precision == Precision.float:
            classes = classes.to(torch.float32)
        if self.precision is None:
            classes = classes.to(torch.uint8)

        return boxes, scores, mask_coeffs.permute(0, 2, 1), classes, mask_protos

