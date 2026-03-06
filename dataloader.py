"""
COMP0248 Coursework 1: Hand Gesture Detection, Segmentation & Classification
Data Loader Module

This module provides PyTorch Dataset and DataLoader for RGB-D hand gesture data.
Student-level train/val split ensures no data leakage between splits.
"""

import os
import random
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Dict, List, Set


GESTURE_CLASSES = [
    'call', 'dislike', 'like', 'ok', 'one',
    'palm', 'peace', 'rock', 'stop', 'three'
]


def get_student_split(
    data_root: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Split students into train and validation sets.

    Args:
        data_root: Root directory containing student folders
        val_ratio: Fraction of students for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_students, val_students) as sets of student names
    """
    all_students = sorted([
        os.path.basename(f) for f in glob.glob(os.path.join(data_root, '*'))
        if os.path.isdir(f)
    ])
    rng = random.Random(seed)
    rng.shuffle(all_students)
    split_idx = int(len(all_students) * (1 - val_ratio))
    train_students = set(all_students[:split_idx])
    val_students = set(all_students[split_idx:])
    return train_students, val_students


class HandGestureDataset(Dataset):
    """
    Dataset for hand gesture detection, segmentation and classification.

    Uses student-level splitting to prevent data leakage.
    Loads depth from raw .npy files (16-bit) when available, clipped to 2m.

    Args:
        data_root: Root directory containing student folders
        student_set: Set of student folder names to include
        target_size: Target image size (square)
        augment: Whether to apply data augmentation (train only)
    """

    def __init__(
        self,
        data_root: str,
        student_set: Set[str],
        target_size: int = 320,
        augment: bool = False
    ):
        self.target_size = target_size
        self.augment = augment
        self.samples = self._collect_samples(data_root, student_set)

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if augment:
            self.color_jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            )

    def _collect_samples(self, data_root: str, student_set: Set[str]) -> List[Dict]:
        """Scan directory tree and collect all valid samples."""
        samples = []
        for student_name in sorted(student_set):
            student_dir = os.path.join(data_root, student_name)
            if not os.path.isdir(student_dir):
                continue
            for gi, gesture in enumerate(GESTURE_CLASSES):
                gesture_dir = os.path.join(student_dir, gesture)
                if not os.path.isdir(gesture_dir):
                    continue
                for clip in sorted(glob.glob(os.path.join(gesture_dir, '*'))):
                    if not os.path.isdir(clip):
                        continue
                    rgb_folder = os.path.join(clip, 'rgb')
                    ann_folder = os.path.join(clip, 'annotation')
                    if not os.path.isdir(rgb_folder):
                        continue
                    ann_frames = set()
                    if os.path.isdir(ann_folder):
                        ann_frames = {
                            os.path.basename(f)
                            for f in glob.glob(os.path.join(ann_folder, '*.png'))
                        }
                    for rp in sorted(glob.glob(os.path.join(rgb_folder, '*.png'))):
                        fn = os.path.basename(rp)
                        samples.append({
                            'rgb_path': rp,
                            'depth_raw_path': os.path.join(
                                clip, 'depth_raw', fn.replace('.png', '.npy')
                            ),
                            'depth_png_path': os.path.join(clip, 'depth', fn),
                            'annotation_path': os.path.join(ann_folder, fn) if fn in ann_frames else None,
                            'gesture_label': gi,
                            'has_annotation': fn in ann_frames,
                            'student': student_name
                        })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _mask_to_bbox(self, mask_arr: np.ndarray) -> List[float]:
        """Derive normalised bounding box [x1, y1, x2, y2] from binary mask."""
        coords = np.where(mask_arr > 0.5)
        if len(coords[0]) > 0:
            h, w = mask_arr.shape
            return [
                coords[1].min() / w, coords[0].min() / h,
                coords[1].max() / w, coords[0].max() / h
            ]
        return [0.0, 0.0, 0.0, 0.0]

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        # Load RGB
        try:
            rgb = Image.open(s['rgb_path']).convert('RGB').resize(
                (self.target_size, self.target_size), Image.BILINEAR
            )
        except Exception:
            rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))

        # Load mask
        try:
            if s['annotation_path'] and os.path.exists(s['annotation_path']):
                mask = Image.open(s['annotation_path']).convert('L').resize(
                    (self.target_size, self.target_size), Image.NEAREST
                )
            else:
                mask = Image.new('L', (self.target_size, self.target_size), 0)
        except Exception:
            mask = Image.new('L', (self.target_size, self.target_size), 0)

        # Load depth (prefer raw .npy for 16-bit precision)
        try:
            if os.path.exists(s['depth_raw_path']):
                depth_arr = np.load(s['depth_raw_path']).astype(np.float32)
                depth_arr = np.clip(depth_arr, 0, 2000) / 2000.0
                depth_img = Image.fromarray((depth_arr * 255).astype(np.uint8)).resize(
                    (self.target_size, self.target_size), Image.BILINEAR
                )
            elif os.path.exists(s['depth_png_path']):
                depth_img = Image.open(s['depth_png_path']).convert('L').resize(
                    (self.target_size, self.target_size), Image.BILINEAR
                )
            else:
                depth_img = Image.new('L', (self.target_size, self.target_size), 0)
        except Exception:
            depth_img = Image.new('L', (self.target_size, self.target_size), 0)

        # Synchronised augmentation (spatial transforms applied to all)
        if self.augment:
            rgb = self.color_jitter(rgb)
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Derive bbox from mask
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        bbox = self._mask_to_bbox(mask_arr)

        # Convert to tensors
        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = torch.from_numpy(
            np.array(depth_img, dtype=np.float32) / 255.0
        ).unsqueeze(0)
        mask_tensor = torch.from_numpy(
            (mask_arr > 0.5).astype(np.float32)
        ).unsqueeze(0)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        label_tensor = torch.tensor(s['gesture_label'], dtype=torch.long)

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'mask': mask_tensor,
            'bbox': bbox_tensor,
            'label': label_tensor,
            'has_annotation': torch.tensor(s['has_annotation'], dtype=torch.bool),
            'rgb_path': s['rgb_path']
        }


def get_data_loaders(
    data_root: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    seed: int = 42,
    target_size: int = 320,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders with student-level splitting.

    Args:
        data_root: Root directory containing student folders
        batch_size: Batch size
        val_ratio: Fraction of students for validation
        seed: Random seed
        target_size: Target image size (square)
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_students, val_students = get_student_split(data_root, val_ratio, seed)

    train_dataset = HandGestureDataset(
        data_root, train_students, target_size, augment=True
    )
    val_dataset = HandGestureDataset(
        data_root, val_students, target_size, augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
