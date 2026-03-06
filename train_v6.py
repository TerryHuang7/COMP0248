"""
COMP0248 Coursework 1 - Final Sprint Version (v6)
终极冲刺版本 - 整合所有优化

改进点 (vs v5):
  1. [数据增强] 恢复随机旋转和缩放 (v4/v5不小心删掉了)
  2. [AMP混合精度] FP16训练，速度提升1.5x，显存减半
  3. [Backbone升级] ResNet34 -> ResNet50
  4. [Batch Size] 32 -> 64 (配合AMP)
  5. 保持v5的所有修复 (学生划分, 深度图[-1,1], 检测权重20x等)

⚠️ 重要: 此版本保存到单独的weights子文件夹，不覆盖v5！

运行: D:\\anaconda\\python.exe train_v6.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import json
import random
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler  # 🔑 AMP混合精度
from tqdm import tqdm
from PIL import Image
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==================== 可修改的参数 ====================
EPOCHS = 100             # 更多epochs，配合更强增强
BATCH_SIZE = 32          # AMP允许更大batch
IMAGE_SIZE = 320
BACKBONE = 'resnet50'    # 🔑 升级backbone
LEARNING_RATE = 0.001
W_DETECTION = 20.0
W_SEGMENTATION = 2.0
W_CLASSIFICATION = 1.0
VAL_STUDENT_RATIO = 0.2
EARLY_STOP_PATIENCE = 15  # 稍微放宽，因为增强后收敛更慢
SEED = 42

USE_AMP = True           # 🔑 开启混合精度
USE_ROTATION = True      # 🔑 恢复旋转增强
USE_SCALE = True         # 🔑 添加缩放增强
# ====================================================

DATA_ROOT = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\full_data"
SAVE_DIR = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\weights_v6"  # ⚠️ 单独文件夹！

GESTURE_CLASSES = ['call', 'dislike', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'three']

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h{int((seconds%3600)//60)}m"


def get_student_split(data_root, val_ratio=0.2, seed=42):
    """按学生划分train/val"""
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


# ==================== Dataset: 恢复旋转增强 ====================

class SprintDataset(Dataset):
    """v6数据集: 恢复旋转+缩放增强"""
    def __init__(self, data_root, student_set, target_size=320, augment=False, seed=42):
        self.target_size = target_size
        self.augment = augment
        self.student_set = student_set
        
        self.samples = self._build_samples(data_root, student_set)
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        
        if augment:
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        
        print(f"  {'Train' if augment else 'Val'} samples: {len(self.samples)} from {len(student_set)} students")
    
    def _build_samples(self, data_root, student_set):
        samples = []
        for student_name in student_set:
            sf = os.path.join(data_root, student_name)
            if not os.path.isdir(sf):
                continue
            for gi, gn in enumerate(GESTURE_CLASSES):
                gf = os.path.join(sf, f'G{gi+1:02d}_{gn}')
                if not os.path.exists(gf):
                    continue
                for clip in glob.glob(os.path.join(gf, 'clip*')):
                    ann_folder = os.path.join(clip, 'annotation')
                    ann_frames = set()
                    if os.path.exists(ann_folder):
                        ann_frames = {os.path.basename(f) for f in glob.glob(os.path.join(ann_folder, '*.png'))}
                    rgb_folder = os.path.join(clip, 'rgb')
                    if not os.path.exists(rgb_folder):
                        continue
                    for rp in sorted(glob.glob(os.path.join(rgb_folder, '*.png'))):
                        fn = os.path.basename(rp)
                        samples.append({
                            'rgb_path': rp,
                            'depth_raw_path': os.path.join(clip, 'depth_raw', fn.replace('.png', '.npy')),
                            'depth_png_path': os.path.join(clip, 'depth', fn),
                            'annotation_path': os.path.join(ann_folder, fn) if fn in ann_frames else None,
                            'gesture_label': gi,
                            'has_annotation': fn in ann_frames,
                            'student': student_name
                        })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. 🔑 带有容错机制的 RGB 加载
        try:
            rgb = Image.open(s['rgb_path']).convert('RGB').resize((self.target_size, self.target_size), Image.BILINEAR)
        except Exception as e:
            print(f"\n[警告] 无法读取图像 (已用黑图替代): {s['rgb_path']} | 错误: {e}")
            rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # 2. 🔑 带有容错机制的 Mask 加载
        try:
            if s['annotation_path'] and os.path.exists(s['annotation_path']):
                mask = Image.open(s['annotation_path']).convert('L').resize((self.target_size, self.target_size), Image.NEAREST)
            else:
                mask = Image.new('L', (self.target_size, self.target_size), 0)
        except Exception as e:
            mask = Image.new('L', (self.target_size, self.target_size), 0)
        
        # 3. 🔑 带有容错机制的 Depth 加载
        try:
            if os.path.exists(s['depth_raw_path']):
                d_arr = np.load(s['depth_raw_path']).astype(np.float32)
                d_arr = np.clip(d_arr, 0, 2000) / 2000.0  # 归一化到 [0, 1]
                depth_img = Image.fromarray(d_arr, mode='F').resize((self.target_size, self.target_size), Image.NEAREST)
            elif os.path.exists(s['depth_png_path']):
                d_img = Image.open(s['depth_png_path']).convert('L').resize((self.target_size, self.target_size), Image.NEAREST)
                d_arr = np.array(d_img, dtype=np.float32) / 255.0
                depth_img = Image.fromarray(d_arr, mode='F')
            else:
                depth_img = Image.new('F', (self.target_size, self.target_size), 0.0)
        except Exception as e:
            depth_img = Image.new('F', (self.target_size, self.target_size), 0.0)
        
        # 4. 🔑 同步数据增强 (RGB, Mask, Depth 必须经历完全相同的空间变换！)
        if self.augment:
            # 水平翻转
            if random.random() > 0.5:
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 颜色抖动 (仅限RGB)
            rgb = self.color_jitter(rgb)
            
            # 随机旋转
            if USE_ROTATION and random.random() > 0.5:
                angle = random.uniform(-15, 15)
                rgb = rgb.rotate(angle, resample=Image.BILINEAR, fillcolor=(0,0,0))
                mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                depth_img = depth_img.rotate(angle, resample=Image.NEAREST, fillcolor=0.0)
            
            # 随机缩放与裁剪/填充
            if USE_SCALE and random.random() > 0.5:
                scale = random.uniform(0.9, 1.1)
                new_size = int(self.target_size * scale)
                
                rgb = rgb.resize((new_size, new_size), Image.BILINEAR)
                mask = mask.resize((new_size, new_size), Image.NEAREST)
                depth_img = depth_img.resize((new_size, new_size), Image.NEAREST)
                
                if new_size > self.target_size:
                    # 随机裁剪
                    left = random.randint(0, new_size - self.target_size)
                    top = random.randint(0, new_size - self.target_size)
                    rgb = rgb.crop((left, top, left + self.target_size, top + self.target_size))
                    mask = mask.crop((left, top, left + self.target_size, top + self.target_size))
                    depth_img = depth_img.crop((left, top, left + self.target_size, top + self.target_size))
                else:
                    # 填充 (Padding)
                    pad_left = (self.target_size - new_size) // 2
                    pad_top = (self.target_size - new_size) // 2
                    
                    new_rgb = Image.new('RGB', (self.target_size, self.target_size), (0,0,0))
                    new_rgb.paste(rgb, (pad_left, pad_top))
                    rgb = new_rgb
                    
                    new_mask = Image.new('L', (self.target_size, self.target_size), 0)
                    new_mask.paste(mask, (pad_left, pad_top))
                    mask = new_mask
                    
                    new_depth = Image.new('F', (self.target_size, self.target_size), 0.0)
                    new_depth.paste(depth_img, (pad_left, pad_top))
                    depth_img = new_depth
        
        # 5. 转换为 Tensors
        rgb_t = self.normalize(transforms.ToTensor()(rgb))
        
        mask_arr = (np.array(mask, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        
        depth_arr = np.array(depth_img, dtype=np.float32)
        depth_t = torch.from_numpy(depth_arr).unsqueeze(0)
        depth_t = (depth_t - 0.5) / 0.5  # 归一化到 [-1, 1]
        
        # 6. 🔑 生成 BBox (在所有空间变换完成之后统一生成！)
        if s['has_annotation'] and np.any(mask_arr > 0.5):
            bbox = self._mask_to_bbox(mask_arr)
            # 过滤掉非法的bbox (面积为0等情况)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]
        bbox_t = torch.tensor(bbox, dtype=torch.float32)
        
        return {
            'rgb': rgb_t,
            'depth': depth_t,
            'mask': mask_t,
            'bbox': bbox_t,
            'label': torch.tensor(s['gesture_label'], dtype=torch.long),
            'has_annotation': torch.tensor(s['has_annotation'], dtype=torch.bool)
        }
    
    def _mask_to_bbox(self, mask_arr):
        coords = np.where(mask_arr > 0.5)
        if len(coords[0]) > 0:
            h, w = mask_arr.shape
            return [coords[1].min()/w, coords[0].min()/h, coords[1].max()/w, coords[0].max()/h]
        return [0.0, 0.0, 0.0, 0.0]


# ==================== Model: ResNet50 ====================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.conv1 = ConvBlock(out_ch + skip_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class HandGestureNetV6(nn.Module):
    """v6模型: ResNet50 backbone"""
    def __init__(self, num_classes=10, input_channels=4, backbone='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        
        # Load ResNet50
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.resnet50(weights=weights)
            self.encoder_channels = [64, 256, 512, 1024, 2048]  # Res50通道不同
        else:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone_model = models.resnet34(weights=weights)
            self.encoder_channels = [64, 64, 128, 256, 512]
        
        # Modify first conv for RGB-D
        if input_channels != 3:
            orig_conv = backbone_model.conv1
            self.encoder_conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                self.encoder_conv1.weight[:, :3] = orig_conv.weight
                self.encoder_conv1.weight[:, 3:] = orig_conv.weight.mean(dim=1, keepdim=True)
        else:
            self.encoder_conv1 = backbone_model.conv1
        
        self.encoder_bn1 = backbone_model.bn1
        self.encoder_relu = backbone_model.relu
        self.encoder_maxpool = backbone_model.maxpool
        
        self.encoder_layer1 = backbone_model.layer1
        self.encoder_layer2 = backbone_model.layer2
        self.encoder_layer3 = backbone_model.layer3
        self.encoder_layer4 = backbone_model.layer4
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder_channels[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 4),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.encoder_channels[4], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Segmentation decoder (适配ResNet50通道)
        if backbone == 'resnet50':
            self.segmentation_decoder4 = DecoderBlock(2048, 1024, 512)
            self.segmentation_decoder3 = DecoderBlock(512, 512, 256)
            self.segmentation_decoder2 = DecoderBlock(256, 256, 128)
            self.segmentation_decoder1 = DecoderBlock(128, 64, 64)
        else:
            self.segmentation_decoder4 = DecoderBlock(512, 256, 256)
            self.segmentation_decoder3 = DecoderBlock(256, 128, 128)
            self.segmentation_decoder2 = DecoderBlock(128, 64, 64)
            self.segmentation_decoder1 = DecoderBlock(64, 64, 64)
        
        self.segmentation_final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        
        x = self.encoder_conv1(x)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x0 = self.encoder_maxpool(x)
        
        x1 = self.encoder_layer1(x0)
        x2 = self.encoder_layer2(x1)
        x3 = self.encoder_layer3(x2)
        x4 = self.encoder_layer4(x3)
        
        bbox = self.detection_head(x4)
        class_logits = self.classification_head(x4)
        
        s = self.segmentation_decoder4(x4, x3)
        s = self.segmentation_decoder3(s, x2)
        s = self.segmentation_decoder2(s, x1)
        s = self.segmentation_decoder1(s, x0)
        mask = self.segmentation_final(s)
        
        input_size = rgb.shape[2:]
        mask = F.interpolate(mask, size=input_size, mode='bilinear', align_corners=False)
        mask = torch.clamp(mask, 0.0, 1.0)  
        
        return {'bbox': bbox, 'mask': mask, 'class_logits': class_logits}


# ==================== Loss ====================

class MultiTaskLoss(nn.Module):
    def __init__(self, w_det=20.0, w_seg=2.0, w_cls=1.0):
        super().__init__()
        self.w_det = w_det
        self.w_seg = w_seg
        self.w_cls = w_cls
        self.bbox_loss = nn.SmoothL1Loss()
        self.seg_loss = nn.BCELoss()
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, preds, targets, has_ann):
        cls_loss = self.cls_loss(preds['class_logits'], targets['label'])
        
        if has_ann.any():
            det_loss = self.bbox_loss(preds['bbox'][has_ann], targets['bbox'][has_ann])
            seg_loss = self.seg_loss(preds['mask'][has_ann], targets['mask'][has_ann])
        else:
            det_loss = torch.tensor(0.0, device=preds['bbox'].device)
            seg_loss = torch.tensor(0.0, device=preds['mask'].device)
        
        total = self.w_det * det_loss + self.w_seg * seg_loss + self.w_cls * cls_loss
        return {'total': total, 'detection': det_loss, 'segmentation': seg_loss, 'classification': cls_loss}


# ==================== Training Functions ====================

def train_one_epoch_amp(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    """AMP混合精度训练"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}',
                bar_format='{desc} |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
                ncols=100)
    
    for batch in pbar:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        targets = {
            'bbox': batch['bbox'].to(device),
            'mask': batch['mask'].to(device),
            'label': batch['label'].to(device)
        }
        has_ann = batch['has_annotation'].to(device)
        
        optimizer.zero_grad()
        
        # 🔑 AMP混合精度前向
        # 🔑 AMP混合精度前向
        with torch.autocast('cuda', enabled=USE_AMP):
            preds = model(rgb, depth)
            
        # 🔑 退出 autocast 上下文，将预测转回 FP32，完美绕过 BCELoss 限制
        preds_fp32 = {k: v.float() for k, v in preds.items()}
        losses = criterion(preds_fp32, targets, has_ann)
        
        # 🔑 AMP反向
        scaler.scale(losses['total']).backward()
        
        # 梯度裁剪 (先unscale)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += losses['total'].item()
        pred_labels = preds['class_logits'].argmax(dim=1)
        correct += (pred_labels == targets['label']).sum().item()
        total += targets['label'].size(0)
        
        avg_loss = total_loss / (pbar.n + 1)
        acc = 100.0 * correct / total
        pbar.set_postfix_str(f'Loss: {avg_loss:.3f}, Acc: {acc:.1f}%')
    
    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """验证 + 生成混淆矩阵数据"""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in loader:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        targets = {
            'bbox': batch['bbox'].to(device),
            'mask': batch['mask'].to(device),
            'label': batch['label'].to(device)
        }
        has_ann = batch['has_annotation'].to(device)
        
        preds = model(rgb, depth)
        losses = criterion(preds, targets, has_ann)
        
        total_loss += losses['total'].item()
        pred_labels = preds['class_logits'].argmax(dim=1)
        
        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(targets['label'].cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
    
    # 计算macro F1
    f1_scores = []
    for i in range(10):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        if tp + fp > 0 and tp + fn > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall > 0:
                f1_scores.append(2 * precision * recall / (precision + recall))
    
    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return total_loss / len(loader), accuracy * 100, macro_f1 * 100, cm


def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Epoch)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==================== Main ====================

def main():
    print("="*70)
    print("COMP0248 Coursework 1 - Final Sprint (v6)")
    print("="*70)
    print(f"\n🔥 冲刺配置:")
    print(f"  Backbone: {BACKBONE} (vs ResNet34 in v5)")
    print(f"  Batch Size: {BATCH_SIZE} (AMP enabled: {USE_AMP})")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Data Aug: Flip + Color + Rotation({USE_ROTATION}) + Scale({USE_SCALE})")
    print(f"  Detection Weight: {W_DETECTION}x")
    print(f"\n⚠️  保存到: {SAVE_DIR} (不会覆盖v5!)")
    print("="*70)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  AMP混合精度: {'开启' if USE_AMP else '关闭'}")
    
    # 学生划分
    print("\n划分train/val (按学生)...")
    train_students, val_students = get_student_split(DATA_ROOT, VAL_STUDENT_RATIO, SEED)
    print(f"  Train: {len(train_students)} students")
    print(f"  Val: {len(val_students)} students")
    
    # 数据加载
    print("\n加载数据...")
    train_dataset = SprintDataset(DATA_ROOT, train_students, IMAGE_SIZE, augment=True, seed=SEED)
    val_dataset = SprintDataset(DATA_ROOT, val_students, IMAGE_SIZE, augment=False, seed=SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    # 模型
    print(f"\n创建模型: HandGestureNetV6 ({BACKBONE})")
    model = HandGestureNetV6(num_classes=10, input_channels=4, backbone=BACKBONE, pretrained=True)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 优化器、损失、调度器
    criterion = MultiTaskLoss(W_DETECTION, W_SEGMENTATION, W_CLASSIFICATION)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 🔑 AMP Scaler
    scaler = GradScaler() if USE_AMP else None
    
    # 保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)
    exp_name = f"v6_{BACKBONE}_{EPOCHS}ep_{datetime.now().strftime('%m%d_%H%M')}"
    save_dir = os.path.join(SAVE_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    config = {
        'version': 'v6_sprint',
        'backbone': BACKBONE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'use_amp': USE_AMP,
        'use_rotation': USE_ROTATION,
        'use_scale': USE_SCALE,
        'image_size': IMAGE_SIZE,
        'lr': LEARNING_RATE,
        'w_det': W_DETECTION,
        'w_seg': W_SEGMENTATION,
        'w_cls': W_CLASSIFICATION,
        'train_students': len(train_students),
        'val_students': len(val_students),
        'seed': SEED
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n保存目录: {save_dir}")
    print("="*70)
    print("开始训练！")
    print("="*70)
    
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch_amp(
            model, train_loader, criterion, optimizer, scaler, device, epoch, EPOCHS
        )
        
        # 验证
        val_loss, val_acc, val_f1, cm = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = epoch_time * (EPOCHS - epoch)
        
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Time: {format_time(epoch_time)} | Elapsed: {format_time(elapsed)} | ETA: {format_time(remaining)}")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Macro-F1: {val_f1:.2f}%")
        
        # 保存最佳模型 (按val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_f1': val_f1,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  💾 Saved best model (val_loss: {best_val_loss:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 保存最佳F1模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': best_val_f1,
                'config': config
            }, os.path.join(save_dir, 'best_f1_model.pth'))
        
        # 定期保存
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
            # 保存混淆矩阵
            plot_confusion_matrix(cm, os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️ Early stopping triggered (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break
        
        print('='*70)
    
    # 最终保存
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': config
    }, os.path.join(save_dir, 'final_model.pth'))
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("训练完成!")
    print(f"总用时: {format_time(total_time)}")
    print(f"最佳模型: {save_dir}/best_model.pth")
    print(f"最佳F1模型: {save_dir}/best_f1_model.pth")
    print(f"\n⚠️  这是v6冲刺版本，保存到单独的weights_v6文件夹")
    print(f"   v5模型在: D:\\OneDrive - University College London\\Robotics & AI\\Objectdetection\\weights/")
    print(f"   不会互相覆盖！")
    print('='*70)

if __name__ == '__main__':
    main()
