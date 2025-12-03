# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Segment
from ultralytics.nn.tasks import SegmentationModel
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

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
        
        # Initialize bauxite/laterite classification model (hardcoded)
        self.classifier_model = None
        self.use_mobilenet = False  # Set to True for MobileNet, False for ResNet
        
        if self.use_mobilenet:
            self.classifier_model_path = '/home/gautam/diy_hanoon/Research/hindalco-samari/test/classification/best_bauxite_laterite_mobilenet.pth'
        else:
            self.classifier_model_path = '/home/gautam/diy_hanoon/Research/hindalco-samari/test/classification/best_bauxite_laterite_resnet50.pth'
        
        # Hardcode normalization values for ONNX compatibility
        self.classifier_img_size = 224  # Standard size for both models
        self.classifier_crop_size = 640  # Size of centered crop
        self.classifier_norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.classifier_norm_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._load_classifier_model()

    def _load_classifier_model(self):
        """Load the bauxite/laterite classification model."""
        try:
            device = next(self.model.parameters()).device
            
            if self.use_mobilenet:
                # Load MobileNetV3-Small
                self.classifier_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
                # Replace final classifier layer for binary classification
                num_ftrs = self.classifier_model.classifier[3].in_features
                self.classifier_model.classifier[3] = nn.Linear(num_ftrs, 2)  # 0 = bauxite, 1 = laterite
            else:
                # Load ResNet-50
                self.classifier_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                # Replace final FC layer for binary classification
                num_ftrs = self.classifier_model.fc.in_features
                self.classifier_model.fc = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(num_ftrs, 2)  # 0 = bauxite, 1 = laterite
                )
            
            self.classifier_model.load_state_dict(torch.load(self.classifier_model_path, map_location=device))
            self.classifier_model.to(device).eval()
            print(f"✓ Loaded {'MobileNetV3-Small' if self.use_mobilenet else 'ResNet-50'} classifier from {self.classifier_model_path}")
        except Exception as e:
            print(f"Warning: Could not load classification model: {e}")
            self.classifier_model = None

    def _create_masked_crop(self, image: torch.Tensor, boxes: torch.Tensor, mask_coeffs: torch.Tensor, mask_protos: torch.Tensor) -> torch.Tensor:
        """
        Create masked crops similar to training data: 640x640 centered crop with black background.
        
        Args:
            image: Tensor of shape [batch_size, 3, H, W] with values in range [0, 1]
            boxes: Tensor of shape [batch_size, 1, 4] with box coordinates [x1, y1, x2, y2]
            mask_coeffs: Tensor of shape [batch_size, 1, num_protos]
            mask_protos: Tensor of shape [batch_size, num_protos, mask_h, mask_w]
        
        Returns:
            crops: Tensor of shape [batch_size, 3, 640, 640] with masked and centered crops
        """
        batch_size = image.shape[0]
        device = image.device
        _, _, img_h, img_w = image.shape
        
        # Generate mask from coefficients and prototypes
        # mask_coeffs: [batch_size, 1, num_protos], mask_protos: [batch_size, num_protos, mask_h, mask_w]
        masks = torch.einsum('bnp,bphw->bnhw', mask_coeffs, mask_protos)  # [batch_size, 1, mask_h, mask_w]
        masks = torch.sigmoid(masks)  # Apply sigmoid to get [0, 1] range
        
        # Resize masks to image size
        masks_resized = F.interpolate(masks, size=(img_h, img_w), mode='bilinear', align_corners=False)
        masks_resized = (masks_resized > 0.5).float()  # Binarize
        
        # Create black canvas for crops
        crops = torch.zeros((batch_size, 3, self.classifier_crop_size, self.classifier_crop_size), 
                           device=device, dtype=image.dtype)
        
        for b in range(batch_size):
            # Get bounding box
            x1, y1, x2, y2 = boxes[b, 0]
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            
            # Extract ROI and apply mask
            roi = image[b:b+1, :, y1:y2, x1:x2]
            roi_mask = masks_resized[b:b+1, 0:1, y1:y2, x1:x2]
            masked_roi = roi * roi_mask
            
            # Get ROI dimensions
            roi_h, roi_w = y2 - y1, x2 - x1
            
            # Resize if larger than crop size while maintaining aspect ratio
            if roi_h > self.classifier_crop_size or roi_w > self.classifier_crop_size:
                scale = min(self.classifier_crop_size / roi_h, self.classifier_crop_size / roi_w)
                new_h = int(roi_h * scale)
                new_w = int(roi_w * scale)
                masked_roi = F.interpolate(masked_roi, size=(new_h, new_w), mode='bilinear', align_corners=False)
                roi_h, roi_w = new_h, new_w
            
            # Center the ROI on the black canvas
            start_y = (self.classifier_crop_size - roi_h) // 2
            start_x = (self.classifier_crop_size - roi_w) // 2
            crops[b, :, start_y:start_y+roi_h, start_x:start_x+roi_w] = masked_roi.squeeze(0)
        
        return crops
    
    def _classify_material(self, image: torch.Tensor, boxes: torch.Tensor, mask_coeffs: torch.Tensor, mask_protos: torch.Tensor) -> torch.Tensor:
        """
        Classify segmented regions as bauxite (0) or laterite (1).
        
        Args:
            image: Tensor of shape [batch_size, 3, H, W] with values in range [0, 1]
            boxes: Tensor of shape [batch_size, 1, 4] with box coordinates
            mask_coeffs: Tensor of shape [batch_size, 1, num_protos]
            mask_protos: Tensor of shape [batch_size, num_protos, mask_h, mask_w]
        
        Returns:
            labels: Tensor of shape [batch_size, 1] with values 0 (Bauxite) or 1 (Laterite)
        """
        if self.classifier_model is None:
            # If classifier model not loaded, return 0 (bauxite)
            batch_size = image.shape[0]
            return torch.zeros((batch_size, 1), dtype=torch.int32, device=image.device)
        
        batch_size = image.shape[0]
        device = image.device
        
        # Create masked crops (640x640 with black background)
        crops = self._create_masked_crop(image, boxes, mask_coeffs, mask_protos)
        
        # Resize to classifier input size (224x224)
        resized = F.interpolate(crops, size=(self.classifier_img_size, self.classifier_img_size), 
                               mode='bilinear', align_corners=False)
        
        # Normalize using ImageNet stats
        mean = self.classifier_norm_mean.to(device)
        std = self.classifier_norm_std.to(device)
        normalized = (resized - mean) / std
        
        # Run classification
        pred = self.classifier_model(normalized).argmax(1)  # Shape: [batch_size]
        
        # pred == 0 means bauxite, pred == 1 means laterite
        labels = pred.unsqueeze(1).to(torch.int32)  # Shape: [batch_size, 1]
        
        return labels

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
        Returns ONLY the highest score prediction.

        Parameters
        ----------
            image: Pixel values pre-processed for encoder consumption.
                   Range: float[0, 1]
                   3-channel Color Space: RGB
        Returns:
            Tuple of 6 tensors:
                boxes:
                    Shape [1, 1, 4]
                    where 4 = [x1, y1, x2, y2] (box coordinates in pixel space)
                scores:
                    Shape [batch_size, 1]
                    per-anchor confidence of whether the anchor box
                    contains an object / box or does not contain an object
                mask_coeffs:
                    Shape [batch_size, 1, num_prototype_masks]
                    Per-anchor mask coefficients
                class_idx:
                    Shape [batch_size, 1]
                    Index
                mask_protos:
                    Shape [batch_size, num_prototype_masks, mask_x_size, mask_y_size]
                    Mask protos.
                labels:
                    Shape [batch_size, 1]
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

        # ONLY keep the highest score prediction
        batch_size = scores.shape[0]
        
        # Find the index of the highest score for each batch
        max_score_idx = torch.argmax(scores, dim=1, keepdim=True)  # Shape: [batch_size, 1]
        
        # Gather the highest score prediction for each output
        # boxes: [batch_size, num_anchors, 4] -> [batch_size, 1, 4]
        boxes = torch.gather(boxes, 1, max_score_idx.unsqueeze(-1).expand(-1, -1, 4))
        
        # scores: [batch_size, num_anchors] -> [batch_size, 1]
        scores_filtered = torch.gather(scores, 1, max_score_idx)
        
        # mask_coeffs: [batch_size, num_anchors, num_protos] -> [batch_size, 1, num_protos]
        mask_coeffs_permuted = mask_coeffs.permute(0, 2, 1)
        num_protos = mask_coeffs_permuted.shape[2]
        mask_coeffs_filtered = torch.gather(mask_coeffs_permuted, 1, max_score_idx.unsqueeze(-1).expand(-1, -1, num_protos))
        
        # classes: [batch_size, num_anchors] -> [batch_size, 1]
        classes = torch.gather(classes, 1, max_score_idx)
        
        # Apply material classification: returns 0 for bauxite, 1 for laterite
        labels = self._classify_material(image, boxes, mask_coeffs_filtered, mask_protos)
        
        # mask_protos remains unchanged as it's shared across all predictions
        return boxes, scores_filtered, mask_coeffs_filtered, classes, mask_protos, labels

