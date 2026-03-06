"""
训练脚本 v5 - 最终版
修复:
  v4全部修复 +
  1. 混淆矩阵生成并保存为图片
  2. 深度图归一化到[-1,1] (与RGB分布对齐)
  3. 学生划分逻辑移到main() (避免全局seed副作用)
  4. Early Stopping (patience=10)

运行: D:\\anaconda\\python.exe train_v5.py
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
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import glob
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 可修改的参数 ====================
EPOCHS = 80
BATCH_SIZE = 32
IMAGE_SIZE = 320
BACKBONE = 'resnet34'
LEARNING_RATE = 0.001
W_DETECTION = 20.0
W_SEGMENTATION = 2.0
W_CLASSIFICATION = 1.0
VAL_STUDENT_RATIO = 0.2
EARLY_STOP_PATIENCE = 10  # 连续10个验证epoch不改善就停止
SEED = 42
# ====================================================

DATA_ROOT = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\full_data"
SAVE_DIR = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\weights"

GESTURE_CLASSES = ['call', 'dislike', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'three']

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h{int((seconds%3600)//60)}m"


def get_student_split(data_root, val_ratio=0.2, seed=42):
    """在main()中划分学生，避免Dataset内部重置全局seed"""
    all_students = sorted([
        os.path.basename(f) for f in glob.glob(os.path.join(data_root, '*'))
        if os.path.isdir(f)
    ])
    rng = random.Random(seed)  # 局部Random，不影响全局
    rng.shuffle(all_students)
    split_idx = int(len(all_students) * (1 - val_ratio))
    train_students = set(all_students[:split_idx])
    val_students = set(all_students[split_idx:])
    return train_students, val_students


def plot_confusion_matrix(cm, class_names, save_path):
    """生成并保存混淆矩阵图片"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存: {save_path}")


# ==================== Dataset ====================

class StudentSplitDataset(Dataset):
    """按学生划分，allowed_students从外部传入"""
    def __init__(self, data_root, allowed_students, target_size=320, augment=False):
        self.target_size = target_size
        self.augment = augment
        self.samples = self._build_samples(data_root, allowed_students)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    
    def _build_samples(self, data_root, allowed_students):
        samples = []
        for sf in glob.glob(os.path.join(data_root, '*')):
            if not os.path.isdir(sf):
                continue
            student_name = os.path.basename(sf)
            if student_name not in allowed_students:
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
    
    def _load_depth(self, sample):
        """读取depth_raw .npy, torch interpolate NEAREST, 归一化到[-1,1]"""
        try:
            if os.path.exists(sample['depth_raw_path']):
                depth_arr = np.load(sample['depth_raw_path']).astype(np.float32)
                depth_arr = np.clip(depth_arr, 0, 2000) / 2000.0  # [0, 1]
                depth_t = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0)
                depth_t = F.interpolate(depth_t, size=(self.target_size, self.target_size), mode='nearest')
                depth_t = depth_t.squeeze(0)  # [1, H, W]
                # 归一化到[-1, 1]，与RGB分布对齐
                depth_t = (depth_t - 0.5) / 0.5
                return depth_t
            elif os.path.exists(sample['depth_png_path']):
                depth = Image.open(sample['depth_png_path']).convert('L')
                depth = depth.resize((self.target_size, self.target_size), Image.NEAREST)
                depth_t = torch.from_numpy(np.array(depth, dtype=np.float32) / 255.0).unsqueeze(0)
                depth_t = (depth_t - 0.5) / 0.5
                return depth_t
        except Exception:
            pass
        return torch.zeros((1, self.target_size, self.target_size))
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        try:
            rgb = Image.open(s['rgb_path']).convert('RGB')
        except:
            rgb = Image.new('RGB', (self.target_size, self.target_size))
        
        try:
            if s['annotation_path'] and os.path.exists(s['annotation_path']):
                mask = Image.open(s['annotation_path']).convert('L')
            else:
                mask = Image.new('L', rgb.size, 0)
        except:
            mask = Image.new('L', rgb.size, 0)
        
        flip = False
        if self.augment:
            if random.random() > 0.5:
                flip = True
                rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            rgb = self.color_jitter(rgb)
        
        rgb = rgb.resize((self.target_size, self.target_size), Image.BILINEAR)
        mask = mask.resize((self.target_size, self.target_size), Image.NEAREST)
        
        depth_t = self._load_depth(s)
        if flip:
            depth_t = depth_t.flip(-1)
        
        rgb_t = self.normalize(transforms.ToTensor()(rgb))
        
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        mask_arr = (mask_arr > 0.5).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        
        coords = np.where(mask_arr > 0.5)
        if len(coords[0]) > 0:
            h, w = mask_arr.shape
            x1 = np.clip(coords[1].min() / w, 0.0, 1.0)
            y1 = np.clip(coords[0].min() / h, 0.0, 1.0)
            x2 = np.clip(coords[1].max() / w, 0.0, 1.0)
            y2 = np.clip(coords[0].max() / h, 0.0, 1.0)
            bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32) if (x2 > x1 and y2 > y1) else torch.zeros(4)
        else:
            bbox = torch.zeros(4)
        
        return {
            'rgb': rgb_t, 'depth': depth_t, 'mask': mask_t, 'bbox': bbox,
            'label': torch.tensor(s['gesture_label'], dtype=torch.long),
            'has_annotation': torch.tensor(s['has_annotation'], dtype=torch.bool)
        }


# ==================== 模型 ====================
from model import HandGestureNet


# ==================== 损失 ====================

class BalancedLoss(nn.Module):
    def __init__(self, w_det=20.0, w_seg=2.0, w_cls=1.0):
        super().__init__()
        self.w_det = w_det
        self.w_seg = w_seg
        self.w_cls = w_cls
        self.bbox_loss = nn.SmoothL1Loss()
        self.cls_loss = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target):
        smooth = 1.0
        intersection = (pred.reshape(-1) * target.reshape(-1)).sum()
        return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    def forward(self, predictions, targets, has_annotation):
        cls_loss = self.cls_loss(predictions['class_logits'], targets['label'])
        if has_annotation.any():
            det_loss = self.bbox_loss(predictions['bbox'][has_annotation], targets['bbox'][has_annotation])
            pm = predictions['mask'][has_annotation]
            tm = targets['mask'][has_annotation]
            seg_loss = F.binary_cross_entropy(pm, tm) + self.dice_loss(pm, tm)
        else:
            det_loss = torch.tensor(0.0, device=predictions['bbox'].device)
            seg_loss = torch.tensor(0.0, device=predictions['mask'].device)
        total = self.w_det * det_loss + self.w_seg * seg_loss + self.w_cls * cls_loss
        return {'total': total, 'detection': det_loss, 'segmentation': seg_loss, 'classification': cls_loss}


# ==================== 训练 & 验证 ====================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0; correct = 0; total = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}',
                bar_format='{desc} |{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}', ncols=120)
    for batch in pbar:
        rgb = batch['rgb'].to(device); depth = batch['depth'].to(device)
        targets = {'bbox': batch['bbox'].to(device), 'mask': batch['mask'].to(device), 'label': batch['label'].to(device)}
        has_ann = batch['has_annotation'].to(device)
        optimizer.zero_grad()
        preds = model(rgb, depth)
        losses = criterion(preds, targets, has_ann)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += losses['total'].item()
        correct += (preds['class_logits'].argmax(1) == targets['label']).sum().item()
        total += targets['label'].size(0)
        pbar.set_postfix_str(f'Loss:{total_loss/(pbar.n+1):.3f} Acc:{100.*correct/total:.1f}% Det:{losses["detection"].item():.4f}')
    return total_loss / len(loader), 100.0 * correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0; correct = 0; total = 0
    bbox_ious = []; mask_ious = []; dice_scores = []
    all_preds = []; all_labels = []
    
    for batch in loader:
        rgb = batch['rgb'].to(device); depth = batch['depth'].to(device)
        targets = {'bbox': batch['bbox'].to(device), 'mask': batch['mask'].to(device), 'label': batch['label'].to(device)}
        has_ann = batch['has_annotation'].to(device)
        preds = model(rgb, depth)
        losses = criterion(preds, targets, has_ann)
        total_loss += losses['total'].item()
        pl = preds['class_logits'].argmax(1)
        correct += (pl == targets['label']).sum().item()
        total += targets['label'].size(0)
        all_preds.extend(pl.cpu().numpy()); all_labels.extend(targets['label'].cpu().numpy())
        
        if has_ann.any():
            pb = preds['bbox'][has_ann]; tb = targets['bbox'][has_ann]
            x1 = torch.max(pb[:,0], tb[:,0]); y1 = torch.max(pb[:,1], tb[:,1])
            x2 = torch.min(pb[:,2], tb[:,2]); y2 = torch.min(pb[:,3], tb[:,3])
            inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
            pa = (pb[:,2]-pb[:,0]).clamp(min=0) * (pb[:,3]-pb[:,1]).clamp(min=0)
            ta = (tb[:,2]-tb[:,0]).clamp(min=0) * (tb[:,3]-tb[:,1]).clamp(min=0)
            iou = inter / (pa + ta - inter + 1e-6)
            bbox_ious.extend(iou.cpu().numpy())
            
            pm = (preds['mask'][has_ann] > 0.5).float(); tm = targets['mask'][has_ann]
            inter_m = (pm * tm).sum(dim=[1,2,3]); union_m = ((pm+tm)>0).float().sum(dim=[1,2,3])
            mask_ious.extend((inter_m / (union_m + 1e-6)).cpu().numpy())
            dice = (2.*inter_m) / (pm.sum(dim=[1,2,3]) + tm.sum(dim=[1,2,3]) + 1e-6)
            dice_scores.extend(dice.cpu().numpy())
    
    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    
    # Macro F1 (跳过空类)
    per_class_f1 = []
    for c in range(10):
        tp = ((all_preds==c) & (all_labels==c)).sum()
        fp = ((all_preds==c) & (all_labels!=c)).sum()
        fn = ((all_preds!=c) & (all_labels==c)).sum()
        if tp + fp + fn == 0:
            continue
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        per_class_f1.append(f1)
    macro_f1 = np.mean(per_class_f1) if per_class_f1 else 0.0
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(10))
    
    return {
        'loss': total_loss / len(loader),
        'acc': 100.0 * correct / total,
        'macro_f1': macro_f1 * 100,
        'bbox_iou': float(np.mean(bbox_ious)) if bbox_ious else 0.0,
        'mask_iou': float(np.mean(mask_ious)) if mask_ious else 0.0,
        'dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
        'bbox_acc50': float(np.mean(np.array(bbox_ious)>=0.5)*100) if bbox_ious else 0.0,
        'bbox_acc75': float(np.mean(np.array(bbox_ious)>=0.75)*100) if bbox_ious else 0.0,
        'confusion_matrix': cm
    }


# ==================== Main ====================

def main():
    print("="*70)
    print("COMP0248 - Training v5 (Final)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"\nv5 修复:")
    print(f"  [v4] 按学生划分 / depth_raw / NEAREST / 检测20x / F1边界")
    print(f"  [新] 混淆矩阵保存为图片")
    print(f"  [新] 深度图归一化到[-1,1]")
    print(f"  [新] 学生划分在main()中 (不污染全局seed)")
    print(f"  [新] Early Stopping (patience={EARLY_STOP_PATIENCE})")
    print(f"\n配置: {BACKBONE} | {IMAGE_SIZE}px | {EPOCHS}ep | Det={W_DETECTION}x")
    print("="*70)
    
    # 全局seed (可复现)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # 按学生划分 (在main中，不在Dataset内部)
    print("\n划分学生...")
    train_students, val_students = get_student_split(DATA_ROOT, VAL_STUDENT_RATIO, SEED)
    print(f"  训练学生: {len(train_students)} | 验证学生: {len(val_students)}")
    
    # 数据
    print("加载数据...")
    train_dataset = StudentSplitDataset(DATA_ROOT, allowed_students=train_students,
                                         target_size=IMAGE_SIZE, augment=True)
    val_dataset = StudentSplitDataset(DATA_ROOT, allowed_students=val_students,
                                       target_size=IMAGE_SIZE, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    
    print(f"  训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")
    raw_count = sum(1 for s in train_dataset.samples if os.path.exists(s['depth_raw_path']))
    print(f"  depth_raw可用: {raw_count}/{len(train_dataset.samples)} ({100*raw_count/len(train_dataset.samples):.1f}%)")
    
    # 模型
    model = HandGestureNet(num_classes=10, input_channels=4, backbone=BACKBONE, pretrained=True)
    model = model.to(device)
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失 & 优化器
    criterion = BalancedLoss(w_det=W_DETECTION, w_seg=W_SEGMENTATION, w_cls=W_CLASSIFICATION)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 保存目录
    exp_name = f"v5_{BACKBONE}_{EPOCHS}ep_{datetime.now().strftime('%m%d_%H%M')}"
    save_dir = os.path.join(SAVE_DIR, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        'version': 'v5',
        'fixes': ['student_split_in_main', 'depth_normalized_neg1_1', 'confusion_matrix',
                   'early_stopping', 'all_v4_fixes'],
        'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'image_size': IMAGE_SIZE,
        'backbone': BACKBONE, 'lr': LEARNING_RATE,
        'w_det': W_DETECTION, 'w_seg': W_SEGMENTATION, 'w_cls': W_CLASSIFICATION,
        'train_students': len(train_students), 'val_students': len(val_students),
        'seed': SEED
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*70}")
    print("开始训练！")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS)
        
        do_val = (epoch <= 3) or (epoch % 5 == 0) or (epoch > EPOCHS - 10)
        
        if do_val:
            val = validate(model, val_loader, criterion, device)
            
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = (elapsed / epoch) * (EPOCHS - epoch)
            lr = optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{EPOCHS} | {format_time(epoch_time)} | 总: {format_time(elapsed)} | 剩余: {format_time(remaining)} | LR: {lr:.6f}")
            print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
            print(f"  Val   - Loss: {val['loss']:.4f} | Acc: {val['acc']:.1f}% | F1: {val['macro_f1']:.1f}%")
            print(f"  Det   - BBox IoU: {val['bbox_iou']:.4f} | @0.5: {val['bbox_acc50']:.1f}% | @0.75: {val['bbox_acc75']:.1f}%")
            print(f"  Seg   - Mask IoU: {val['mask_iou']:.4f} | Dice: {val['dice']:.4f}")
            
            if val['loss'] < best_val_loss:
                best_val_loss = val['loss']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': {k: v for k, v in val.items() if k != 'confusion_matrix'},
                    'config': config
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"  💾 保存最佳模型")
            else:
                patience_counter += 1
                print(f"  ⏳ 无改善 ({patience_counter}/{EARLY_STOP_PATIENCE})")
            
            # Early stopping
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⛔ Early Stopping! 连续{EARLY_STOP_PATIENCE}次验证无改善")
                break
            
            print(f"{'='*70}")
        else:
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining = (elapsed / epoch) * (EPOCHS - epoch)
            print(f"  Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.3f} | Acc: {train_acc:.1f}% | {format_time(epoch_time)} | 剩余: {format_time(remaining)}")
        
        scheduler.step()
    
    # 最终评估
    print(f"\n{'='*70}")
    print("最终评估...")
    final = validate(model, val_loader, criterion, device)
    print(f"最终结果:")
    print(f"  Acc: {final['acc']:.2f}% | F1: {final['macro_f1']:.2f}%")
    print(f"  BBox IoU: {final['bbox_iou']:.4f} | @0.5: {final['bbox_acc50']:.1f}% | @0.75: {final['bbox_acc75']:.1f}%")
    print(f"  Mask IoU: {final['mask_iou']:.4f} | Dice: {final['dice']:.4f}")
    
    # 保存混淆矩阵
    plot_confusion_matrix(final['confusion_matrix'], GESTURE_CLASSES, os.path.join(save_dir, 'confusion_matrix.png'))
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_metrics': {k: v for k, v in final.items() if k != 'confusion_matrix'},
        'config': config
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # 保存per-class F1
    print(f"\n  Per-class F1:")
    all_preds = []; all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['rgb'].to(device); depth = batch['depth'].to(device)
            preds = model(rgb, depth)
            all_preds.extend(preds['class_logits'].argmax(1).cpu().numpy())
            all_labels.extend(batch['label'].numpy())
    all_preds = np.array(all_preds); all_labels = np.array(all_labels)
    for c, name in enumerate(GESTURE_CLASSES):
        tp = ((all_preds==c) & (all_labels==c)).sum()
        fp = ((all_preds==c) & (all_labels!=c)).sum()
        fn = ((all_preds!=c) & (all_labels==c)).sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        print(f"    {name:12s}: F1={f1*100:.1f}% (P={prec*100:.1f}% R={rec*100:.1f}%)")
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"训练完成! 总用时: {format_time(total_time)}")
    print(f"模型保存: {save_dir}")
    print(f"混淆矩阵: {os.path.join(save_dir, 'confusion_matrix.png')}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
