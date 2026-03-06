"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Multi-Task Model Module

This module implements a multi-task network for simultaneous:
- Hand detection (bounding box regression)
- Hand segmentation (pixel-wise classification)
- Gesture classification (10-class classification)

COMPLIANCE NOTE:
- We use torchvision.models.resnet18/resnet34 as BACKBONE FEATURE EXTRACTORS
- This is ALLOWED: ResNet is a classification backbone, not a detection framework
- PROHIBITED items (NOT used): YOLO, Detectron2, MMDetection,
  segmentation_models_pytorch, torchvision.models.detection.* (Mask R-CNN, etc.)
- All detection, segmentation, and classification heads are CUSTOM implemented
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict


class MultiTaskResNet(nn.Module):
    """
    Multi-task network with ResNet backbone for joint hand detection,
    segmentation, and gesture classification.

    Architecture:
        - Shared encoder: ResNet-18/34 (modified for 4-channel RGB-D input)
        - Detection head: GAP -> FC -> 4 (normalised bbox coordinates)
        - Classification head: GAP -> FC -> num_classes
        - Segmentation decoder: U-Net style with skip connections

    Args:
        backbone: ResNet variant ('resnet18' or 'resnet34')
        num_classes: Number of gesture classes (default: 10)
    """

    def __init__(self, backbone: str = 'resnet34', num_classes: int = 10):
        super().__init__()

        if backbone == 'resnet18':
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feat_dim = 512
        elif backbone == 'resnet34':
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feat_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify first conv layer to accept 4-channel input (RGB + Depth)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3] = base.conv1.weight
            self.conv1.weight[:, 3] = base.conv1.weight[:, 0]

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        # Detection head (bounding box regression)
        self.det_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 4), nn.Sigmoid()
        )

        # Classification head (gesture recognition)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Segmentation decoder (U-Net style with skip connections)
        self.seg_up1 = nn.ConvTranspose2d(feat_dim, 256, 2, stride=2)
        self.seg_conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU()
        )
        self.seg_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.seg_conv2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU()
        )
        self.seg_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.seg_conv3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.seg_up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_conv4 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.seg_up5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.seg_out = nn.Conv2d(16, 1, 1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            rgb: RGB input tensor [B, 3, H, W]
            depth: Depth input tensor [B, 1, H, W]

        Returns:
            Dictionary with keys 'bbox', 'mask', 'class_logits'
        """
        x = torch.cat([rgb, depth], dim=1)

        # Encoder
        c1 = self.relu(self.bn1(self.conv1(x)))   # 64 x H/2 x W/2
        c2 = self.layer1(self.maxpool(c1))          # 64 x H/4 x W/4
        c3 = self.layer2(c2)                        # 128 x H/8 x W/8
        c4 = self.layer3(c3)                        # 256 x H/16 x W/16
        c5 = self.layer4(c4)                        # 512 x H/32 x W/32

        # Detection head
        bbox = self.det_head(c5)

        # Classification head
        cls_logits = self.cls_head(c5)

        # Segmentation decoder with skip connections
        d = self.seg_up1(c5)
        d = self.seg_conv1(torch.cat([d, c4], dim=1))
        d = self.seg_up2(d)
        d = self.seg_conv2(torch.cat([d, c3], dim=1))
        d = self.seg_up3(d)
        d = self.seg_conv3(torch.cat([d, c2], dim=1))
        d = self.seg_up4(d)
        d = self.seg_conv4(torch.cat([d, c1], dim=1))
        d = self.seg_up5(d)
        mask = self.seg_out(d)

        return {
            'bbox': bbox,
            'mask': mask,
            'class_logits': cls_logits
        }
