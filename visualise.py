"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Visualization Module

Provides functions to visualize predictions, masks, bounding boxes, and training curves.
"""

import os
import argparse
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import HandGestureDataset, GESTURE_CLASSES
from model import MultiTaskResNet


def denormalize_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize RGB tensor for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img


def visualize_prediction(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    true_bbox: torch.Tensor,
    pred_bbox: torch.Tensor,
    true_label: int,
    pred_label: int,
    save_path: str = None
):
    """
    Visualize a single prediction with RGB, depth, masks, and bounding boxes.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # RGB image
    rgb_img = denormalize_rgb(rgb)
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')

    # Depth image
    depth_img = depth.squeeze().cpu().numpy()
    axes[0, 1].imshow(depth_img, cmap='viridis')
    axes[0, 1].set_title('Depth Image')
    axes[0, 1].axis('off')

    # RGB with true bbox
    axes[0, 2].imshow(rgb_img)
    h, w = rgb_img.shape[:2]
    x1, y1, x2, y2 = true_bbox.cpu().numpy()
    rect = patches.Rectangle(
        (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
        linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth'
    )
    axes[0, 2].add_patch(rect)
    x1, y1, x2, y2 = pred_bbox.cpu().numpy()
    rect = patches.Rectangle(
        (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
        linewidth=2, edgecolor='r', facecolor='none', label='Predicted'
    )
    axes[0, 2].add_patch(rect)
    axes[0, 2].legend()
    axes[0, 2].set_title('Bounding Boxes')
    axes[0, 2].axis('off')

    # True mask
    axes[1, 0].imshow(true_mask.squeeze().cpu().numpy(), cmap='gray')
    axes[1, 0].set_title('Ground Truth Mask')
    axes[1, 0].axis('off')

    # Predicted mask (apply sigmoid since model outputs logits)
    pred_mask_vis = torch.sigmoid(pred_mask).squeeze().cpu().numpy()
    axes[1, 1].imshow(pred_mask_vis, cmap='gray')
    axes[1, 1].set_title('Predicted Mask')
    axes[1, 1].axis('off')

    # Overlay
    axes[1, 2].imshow(rgb_img)
    pred_binary = (pred_mask_vis > 0.5).astype(np.float32)
    overlay = np.zeros((*pred_binary.shape, 4))
    overlay[..., 0] = 1.0
    overlay[..., 3] = pred_binary * 0.4
    axes[1, 2].imshow(overlay)
    true_label_name = GESTURE_CLASSES[true_label] if true_label < len(GESTURE_CLASSES) else str(true_label)
    pred_label_name = GESTURE_CLASSES[pred_label] if pred_label < len(GESTURE_CLASSES) else str(pred_label)
    color = 'green' if true_label == pred_label else 'red'
    axes[1, 2].set_title(f'True: {true_label_name} | Pred: {pred_label_name}', color=color)
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_batch_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_samples: int = 8,
    save_dir: str = None
):
    """
    Visualize predictions on a batch of samples.
    """
    model.eval()

    batch = next(iter(data_loader))

    rgb = batch['rgb'].to(device)
    depth = batch['depth'].to(device)
    true_masks = batch['mask']
    true_bboxes = batch['bbox']
    true_labels = batch['label']

    with torch.no_grad():
        outputs = model(rgb, depth)

    pred_masks = outputs['mask']
    pred_bboxes = outputs['bbox']
    pred_labels = outputs['class_logits'].argmax(dim=1)

    num_samples = min(num_samples, rgb.size(0))

    for i in range(num_samples):
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sample_{i:03d}.png')

        visualize_prediction(
            rgb[i],
            depth[i],
            true_masks[i],
            pred_masks[i],
            true_bboxes[i],
            pred_bboxes[i],
            true_labels[i].item(),
            pred_labels[i].item(),
            save_path
        )

    print(f"Visualized {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description='Visualize Model Predictions')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='../results/visualizations')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskResNet(
        backbone=args.backbone,
        num_classes=len(GESTURE_CLASSES)
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    from dataloader import get_data_loaders
    _, val_loader = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.num_samples,
        target_size=args.image_size,
        num_workers=0
    )

    visualize_batch_predictions(
        model, val_loader, device,
        num_samples=args.num_samples,
        save_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
