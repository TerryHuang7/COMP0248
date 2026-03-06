"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Utility Functions
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List, Tuple


def compute_bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        box1: [4] array [x1, y1, x2, y2]
        box2: [4] array [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.

    Args:
        mask1: [H, W] binary mask
        mask2: [H, W] binary mask

    Returns:
        IoU value
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union < 1e-6:
        return 0.0

    return intersection / union


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Args:
        mask1: [H, W] binary mask
        mask2: [H, W] binary mask

    Returns:
        Dice coefficient
    """
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()

    if total < 1e-6:
        return 0.0

    return 2.0 * intersection / total


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_path: str,
    config: Dict = None
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }
    if config:
        checkpoint['config'] = config
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None
) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
