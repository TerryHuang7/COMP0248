"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Training Script

Trains a multi-task ResNet model for joint hand detection, segmentation,
and gesture classification. Supports TensorBoard logging, early stopping,
and cosine annealing learning rate scheduling.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import json
import random
import argparse
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score

from dataloader import get_data_loaders, GESTURE_CLASSES
from model import MultiTaskResNet


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining detection, segmentation, and classification.

    Loss weights are critical for balancing gradient magnitudes across tasks:
    - Detection (SmoothL1): operates on normalised coords, typical ~0.01-0.05
    - Segmentation (BCE): operates on full-resolution masks, typical ~0.3-0.8
    - Classification (CE): operates on 10-class logits, typical ~0.5-2.0

    Without reweighting, segmentation dominates and detection fails to learn.
    """

    def __init__(
        self,
        w_det: float = 20.0,
        w_seg: float = 2.0,
        w_cls: float = 1.0
    ):
        super().__init__()
        self.w_det = w_det
        self.w_seg = w_seg
        self.w_cls = w_cls
        self.bbox_loss = nn.SmoothL1Loss()
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        has_annotation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            predictions: Model outputs with keys 'bbox', 'mask', 'class_logits'
            targets: Ground truth with keys 'bbox', 'mask', 'label'
            has_annotation: Boolean mask indicating annotated samples

        Returns:
            Dictionary of individual and total losses
        """
        cls_loss = self.cls_loss(predictions['class_logits'], targets['label'])

        if has_annotation.any():
            det_loss = self.bbox_loss(
                predictions['bbox'][has_annotation],
                targets['bbox'][has_annotation]
            )
            seg_loss = self.seg_loss(
                predictions['mask'][has_annotation],
                targets['mask'][has_annotation]
            )
        else:
            det_loss = torch.tensor(0.0, device=cls_loss.device)
            seg_loss = torch.tensor(0.0, device=cls_loss.device)

        total = (self.w_det * det_loss +
                 self.w_seg * seg_loss +
                 self.w_cls * cls_loss)

        return {
            'total': total,
            'detection': det_loss,
            'segmentation': seg_loss,
            'classification': cls_loss
        }


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    writer: SummaryWriter
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Returns:
        Tuple of (average_loss, classification_accuracy)
    """
    model.train()
    total_loss = 0.0
    det_loss_sum = 0.0
    seg_loss_sum = 0.0
    cls_loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    for batch_idx, batch in enumerate(pbar):
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        target = {
            'bbox': batch['bbox'].to(device),
            'mask': batch['mask'].to(device),
            'label': batch['label'].to(device)
        }
        has_ann = batch['has_annotation'].to(device)

        optimizer.zero_grad()
        pred = model(rgb, depth)
        losses = criterion(pred, target, has_ann)
        losses['total'].backward()
        optimizer.step()

        total_loss += losses['total'].item()
        det_loss_sum += losses['detection'].item()
        seg_loss_sum += losses['segmentation'].item()
        cls_loss_sum += losses['classification'].item()

        pred_labels = pred['class_logits'].argmax(dim=1)
        correct += (pred_labels == target['label']).sum().item()
        total += target['label'].size(0)

        avg_loss = total_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        pbar.set_postfix_str(f'Loss: {avg_loss:.3f}, Acc: {acc:.1f}%')

        global_step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % 100 == 0:
            writer.add_scalar('train/batch_loss', losses['total'].item(), global_step)
            writer.add_scalar('train/batch_acc', acc, global_step)

    n = len(loader)
    avg_loss = total_loss / n
    avg_acc = 100.0 * correct / total

    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/detection_loss', det_loss_sum / n, epoch)
    writer.add_scalar('train/segmentation_loss', seg_loss_sum / n, epoch)
    writer.add_scalar('train/classification_loss', cls_loss_sum / n, epoch)
    writer.add_scalar('train/accuracy', avg_acc, epoch)
    writer.add_scalars('Loss/total', {'train': avg_loss}, epoch)
    writer.add_scalars('Accuracy/classification', {'train': avg_acc}, epoch)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Returns:
        Dictionary of validation metrics including loss, accuracy, and macro-F1.
    """
    model.eval()
    total_loss = 0.0
    det_loss_sum = 0.0
    seg_loss_sum = 0.0
    cls_loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        target = {
            'bbox': batch['bbox'].to(device),
            'mask': batch['mask'].to(device),
            'label': batch['label'].to(device)
        }
        has_ann = batch['has_annotation'].to(device)

        pred = model(rgb, depth)
        losses = criterion(pred, target, has_ann)

        total_loss += losses['total'].item()
        det_loss_sum += losses['detection'].item()
        seg_loss_sum += losses['segmentation'].item()
        cls_loss_sum += losses['classification'].item()

        pred_labels = pred['class_logits'].argmax(dim=1)
        correct += (pred_labels == target['label']).sum().item()
        total += target['label'].size(0)

        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(target['label'].cpu().numpy())

    n = len(loader)
    avg_loss = total_loss / n
    avg_acc = 100.0 * correct / total
    macro_f1 = 100.0 * f1_score(all_labels, all_preds, average='macro')

    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/detection_loss', det_loss_sum / n, epoch)
    writer.add_scalar('val/segmentation_loss', seg_loss_sum / n, epoch)
    writer.add_scalar('val/classification_loss', cls_loss_sum / n, epoch)
    writer.add_scalar('val/accuracy', avg_acc, epoch)
    writer.add_scalar('val/macro_f1', macro_f1, epoch)
    writer.add_scalars('Loss/total', {'val': avg_loss}, epoch)
    writer.add_scalars('Accuracy/classification', {'val': avg_acc}, epoch)

    return {
        'loss': avg_loss,
        'cls_acc': avg_acc,
        'macro_f1': macro_f1
    }


def main():
    parser = argparse.ArgumentParser(description='Train Hand Gesture Multi-Task Model')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset root directory')
    parser.add_argument('--save_dir', type=str, default='../weights',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--backbone', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--w_detection', type=float, default=20.0)
    parser.add_argument('--w_segmentation', type=float, default=2.0)
    parser.add_argument('--w_classification', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Experiment naming
    if args.exp_name is None:
        args.exp_name = f"{args.backbone}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(args.log_dir, args.exp_name)

    # Save config
    config = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Config: {args.backbone} | {args.image_size}px | {args.epochs}ep | "
          f"Det={args.w_detection}x | Patience={args.early_stop_patience}")

    # Data
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_size=args.image_size,
        num_workers=args.num_workers
    )
    print(f"  Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)}")

    # Model
    model = MultiTaskResNet(
        backbone=args.backbone,
        num_classes=len(GESTURE_CLASSES)
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.1f}M")

    # Loss, optimizer, scheduler
    criterion = MultiTaskLoss(
        w_det=args.w_detection,
        w_seg=args.w_segmentation,
        w_cls=args.w_classification
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # TensorBoard
    writer = SummaryWriter(log_dir)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, args.epochs, writer
        )

        val = validate(model, val_loader, criterion, device, epoch, writer)

        scheduler.step()

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (args.epochs - epoch)
        lr = optimizer.param_groups[0]['lr']

        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch}/{args.epochs} | {format_time(epoch_time)} | "
              f"Total: {format_time(elapsed)} | Remaining: {format_time(remaining)} | "
              f"LR: {lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   - Loss: {val['loss']:.4f} | Acc: {val['cls_acc']:.1f}% | "
              f"Macro-F1: {val['macro_f1']:.2f}%")

        # Save best loss model
        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  Saved best loss model: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            print(f"\n  Early stopping triggered! No improvement for "
                  f"{args.early_stop_patience} epochs.")
            print(f"  Loading best model (val_loss={best_val_loss:.4f})")
            best_ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'))
            model.load_state_dict(best_ckpt['model_state_dict'])
            writer.add_text(
                'early_stopping',
                f'Triggered at epoch {epoch}, best epoch {best_ckpt["epoch"]}',
                epoch
            )
            break

    writer.close()
    total_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()
