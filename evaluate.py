"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Evaluation Script

Computes all required metrics:
- Detection: accuracy@0.5 IoU, accuracy@0.75 IoU, mean bbox IoU
- Segmentation: mean IoU, Dice coefficient
- Classification: top-1 accuracy, macro-averaged F1 score, confusion matrix
- M2B post-processing: Mask-to-BBox heuristic for improved detection
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import json
from typing import Dict, List, Tuple, Set
import numpy as np
import random
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataloader import get_data_loaders, get_student_split, HandGestureDataset, GESTURE_CLASSES
from model import MultiTaskResNet


def compute_bbox_iou_batch(
    pred_bboxes: torch.Tensor,
    true_bboxes: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU for batches of bounding boxes.

    Args:
        pred_bboxes: [N, 4] predicted boxes (x1, y1, x2, y2) normalised
        true_bboxes: [N, 4] ground truth boxes (x1, y1, x2, y2) normalised

    Returns:
        [N] IoU values
    """
    pred_bboxes = pred_bboxes.clamp(0, 1)
    true_bboxes = true_bboxes.clamp(0, 1)

    x1 = torch.max(pred_bboxes[:, 0], true_bboxes[:, 0])
    y1 = torch.max(pred_bboxes[:, 1], true_bboxes[:, 1])
    x2 = torch.min(pred_bboxes[:, 2], true_bboxes[:, 2])
    y2 = torch.min(pred_bboxes[:, 3], true_bboxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
    true_area = (true_bboxes[:, 2] - true_bboxes[:, 0]) * (true_bboxes[:, 3] - true_bboxes[:, 1])
    union_area = pred_area + true_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_mask_iou_batch(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor
) -> torch.Tensor:
    """
    Compute IoU for batches of binary masks.

    Args:
        pred_masks: [N, 1, H, W] predicted binary masks
        true_masks: [N, 1, H, W] ground truth binary masks

    Returns:
        [N] IoU values
    """
    intersection = (pred_masks * true_masks).sum(dim=[1, 2, 3])
    union = ((pred_masks + true_masks) > 0).float().sum(dim=[1, 2, 3])
    iou = intersection / (union + 1e-6)
    return iou


def compute_dice_batch(
    pred_masks: torch.Tensor,
    true_masks: torch.Tensor
) -> torch.Tensor:
    """
    Compute Dice coefficient for batches of binary masks.

    Args:
        pred_masks: [N, 1, H, W] predicted binary masks
        true_masks: [N, 1, H, W] ground truth binary masks

    Returns:
        [N] Dice coefficients
    """
    intersection = (pred_masks * true_masks).sum(dim=[1, 2, 3])
    pred_sum = pred_masks.sum(dim=[1, 2, 3])
    true_sum = true_masks.sum(dim=[1, 2, 3])
    dice = (2.0 * intersection) / (pred_sum + true_sum + 1e-6)
    return dice


def mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
    """
    Derive bounding box from binary mask (M2B heuristic).

    Args:
        mask: [1, H, W] binary mask tensor

    Returns:
        [4] normalised bbox tensor [x1, y1, x2, y2]
    """
    mask_2d = mask.squeeze()
    coords = torch.where(mask_2d > 0.5)
    if len(coords[0]) > 0:
        h, w = mask_2d.shape
        y_min, y_max = coords[0].min().float(), coords[0].max().float()
        x_min, x_max = coords[1].min().float(), coords[1].max().float()
        return torch.tensor([x_min / w, y_min / h, x_max / w, y_max / h])
    return torch.zeros(4)


def get_test_loader(
    data_root: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    target_size: int = 320,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a test data loader using all students NOT in the train/val split.

    If no separate test students exist (i.e. all students are already in
    train+val), falls back to the validation set.

    Args:
        data_root: Root directory containing student folders
        batch_size: Batch size
        val_ratio: Fraction used for validation (to reconstruct the split)
        seed: Random seed
        target_size: Target image size
        num_workers: Number of data loading workers

    Returns:
        DataLoader for the test set (or val set as fallback)
    """
    train_students, val_students = get_student_split(data_root, val_ratio, seed)
    all_students = set(
        os.path.basename(f) for f in glob.glob(os.path.join(data_root, '*'))
        if os.path.isdir(f)
    )
    test_students = all_students - train_students - val_students

    if len(test_students) == 0:
        print("  No separate test students found. Using validation set.")
        test_students = val_students

    print(f"  Test students ({len(test_students)}): {sorted(test_students)}")

    test_dataset = HandGestureDataset(
        data_root, test_students, target_size, augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return test_loader


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    use_m2b: bool = False
) -> Tuple[Dict, Dict]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        data_loader: DataLoader for evaluation
        device: Computation device
        use_m2b: Whether to use Mask-to-BBox heuristic for detection

    Returns:
        Tuple of (metrics_dict, raw_predictions_dict)
    """
    model.eval()

    all_predictions = {
        'labels': [],
        'pred_labels': [],
        'bbox_ious': [],
        'bbox_ious_m2b': [],
        'mask_ious': [],
        'dice_scores': []
    }

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            true_labels = batch['label'].to(device)
            true_bboxes = batch['bbox'].to(device)
            true_masks = batch['mask'].to(device)
            has_annotation = batch['has_annotation'].to(device)

            outputs = model(rgb, depth)

            # Classification
            pred_labels = outputs['class_logits'].argmax(dim=1)
            all_predictions['labels'].extend(true_labels.cpu().numpy())
            all_predictions['pred_labels'].extend(pred_labels.cpu().numpy())

            # Detection and segmentation (annotated samples only)
            if has_annotation.any():
                # BBox IoU (direct prediction)
                pred_bboxes = outputs['bbox'][has_annotation]
                true_bboxes_ann = true_bboxes[has_annotation]
                bbox_ious = compute_bbox_iou_batch(pred_bboxes, true_bboxes_ann)
                all_predictions['bbox_ious'].extend(bbox_ious.cpu().numpy())

                # Mask: model outputs logits, apply sigmoid then threshold
                pred_masks = (torch.sigmoid(outputs['mask'][has_annotation]) > 0.5).float()
                true_masks_ann = true_masks[has_annotation]

                mask_ious = compute_mask_iou_batch(pred_masks, true_masks_ann)
                dice_scores = compute_dice_batch(pred_masks, true_masks_ann)
                all_predictions['mask_ious'].extend(mask_ious.cpu().numpy())
                all_predictions['dice_scores'].extend(dice_scores.cpu().numpy())

                # M2B: derive bbox from predicted mask
                if use_m2b:
                    for pm, tb in zip(pred_masks, true_bboxes_ann):
                        m2b_bbox = mask_to_bbox(pm).to(device)
                        m2b_iou = compute_bbox_iou_batch(
                            m2b_bbox.unsqueeze(0), tb.unsqueeze(0)
                        )
                        all_predictions['bbox_ious_m2b'].extend(m2b_iou.cpu().numpy())

    # Compute aggregate metrics
    labels = np.array(all_predictions['labels'])
    pred_labels = np.array(all_predictions['pred_labels'])

    metrics = {}

    # Classification metrics
    metrics['classification'] = {
        'accuracy': 100.0 * accuracy_score(labels, pred_labels),
        'macro_f1': 100.0 * f1_score(labels, pred_labels, average='macro'),
        'per_class_f1': (100.0 * f1_score(labels, pred_labels, average=None)).tolist(),
        'confusion_matrix': confusion_matrix(labels, pred_labels).tolist()
    }

    # Detection metrics (raw)
    bbox_ious = np.array(all_predictions['bbox_ious'])
    if len(bbox_ious) > 0:
        metrics['detection'] = {
            'mean_iou': float(np.mean(bbox_ious)),
            'accuracy_at_0.5': float(np.mean(bbox_ious >= 0.5) * 100),
            'accuracy_at_0.75': float(np.mean(bbox_ious >= 0.75) * 100)
        }
        # M2B detection metrics
        if use_m2b and len(all_predictions['bbox_ious_m2b']) > 0:
            m2b_ious = np.array(all_predictions['bbox_ious_m2b'])
            metrics['detection']['m2b_mean_iou'] = float(np.mean(m2b_ious))
            metrics['detection']['m2b_accuracy_at_0.5'] = float(np.mean(m2b_ious >= 0.5) * 100)
            metrics['detection']['m2b_accuracy_at_0.75'] = float(np.mean(m2b_ious >= 0.75) * 100)
    else:
        metrics['detection'] = {
            'mean_iou': 0.0, 'accuracy_at_0.5': 0.0, 'accuracy_at_0.75': 0.0
        }

    # Segmentation metrics
    mask_ious = np.array(all_predictions['mask_ious'])
    dice_scores = np.array(all_predictions['dice_scores'])
    if len(mask_ious) > 0:
        metrics['segmentation'] = {
            'mean_iou': float(np.mean(mask_ious)),
            'mean_dice': float(np.mean(dice_scores))
        }
    else:
        metrics['segmentation'] = {'mean_iou': 0.0, 'mean_dice': 0.0}

    return metrics, all_predictions


def print_metrics(metrics: Dict, split: str = 'VAL'):
    """Print formatted evaluation metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {split} SET RESULTS")
    print(f"{'=' * 60}")

    cls = metrics['classification']
    print(f"\n  Classification:")
    print(f"    Accuracy:  {cls['accuracy']:.2f}%")
    print(f"    Macro F1:  {cls['macro_f1']:.2f}%")
    print(f"    Per-class F1:")
    for i, name in enumerate(GESTURE_CLASSES):
        print(f"      {name:>10s}: {cls['per_class_f1'][i]:.2f}%")

    det = metrics['detection']
    print(f"\n  Detection (Raw):")
    print(f"    Mean BBox IoU:    {det['mean_iou']:.4f}")
    print(f"    Acc@0.5 IoU:      {det['accuracy_at_0.5']:.2f}%")
    print(f"    Acc@0.75 IoU:     {det['accuracy_at_0.75']:.2f}%")
    if 'm2b_mean_iou' in det:
        print(f"\n  Detection (M2B Post-Processing):")
        print(f"    M2B Mean IoU:     {det['m2b_mean_iou']:.4f}")
        print(f"    M2B Acc@0.5 IoU:  {det['m2b_accuracy_at_0.5']:.2f}%")
        print(f"    M2B Acc@0.75 IoU: {det['m2b_accuracy_at_0.75']:.2f}%")

    seg = metrics['segmentation']
    print(f"\n  Segmentation:")
    print(f"    Mean Mask IoU:  {seg['mean_iou']:.4f}")
    print(f"    Mean Dice:      {seg['mean_dice']:.4f}")
    print(f"{'=' * 60}")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hand Gesture Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--use_m2b', action='store_true', default=False,
                        help='Use Mask-to-BBox heuristic for detection')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = MultiTaskResNet(
        backbone=args.backbone,
        num_classes=len(GESTURE_CLASSES)
    )
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load data
    print("Loading data...")
    if args.split == 'test':
        data_loader = get_test_loader(
            data_root=args.data_root,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            target_size=args.image_size,
            num_workers=0
        )
    else:
        train_loader, val_loader = get_data_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            target_size=args.image_size,
            num_workers=0
        )
        data_loader = val_loader if args.split == 'val' else train_loader

    print(f"  Evaluating on {args.split} set ({len(data_loader.dataset)} samples)")

    # Evaluate
    print("Evaluating model...")
    metrics, predictions = evaluate_model(model, data_loader, device, use_m2b=args.use_m2b)

    # Print results
    print_metrics(metrics, split=args.split.upper())

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Confusion matrix
    cm = np.array(metrics['classification']['confusion_matrix'])
    plot_confusion_matrix(cm, GESTURE_CLASSES,
                          os.path.join(args.output_dir, f'confusion_matrix_{args.split}.png'))

    # Save metrics JSON
    with open(os.path.join(args.output_dir, f'metrics_{args.split}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
