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
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

EPOCHS = 80
BATCH_SIZE = 32
IMAGE_SIZE = 320
BACKBONE = 'resnet34'
LEARNING_RATE = 0.001
W_DETECTION = 20.0
W_SEGMENTATION = 2.0
W_CLASSIFICATION = 1.0
VAL_STUDENT_RATIO = 0.2
EARLY_STOP_PATIENCE = 20
SEED = 42


DATA_ROOT = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\full_data"
SAVE_DIR = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\weights"
LOG_DIR = r"D:\OneDrive - University College London\Robotics & AI\Objectdetection\results\runs"

GESTURE_CLASSES = ['call', 'dislike', 'like', 'ok', 'one', 'palm', 'peace', 'rock', 'stop', 'three']

def format_time(seconds):
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m{int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h{int((seconds%3600)//60)}m"


def get_student_split(data_root, val_ratio=0.2, seed=42):
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


class StudentSplitDataset(Dataset):
    def __init__(self, data_root, student_set, target_size=320, augment=False):
        self.target_size = target_size
        self.augment = augment
        self.samples = self._collect_samples(data_root, student_set)

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ]) if augment else None

    def _collect_samples(self, data_root, student_set):
        samples = []
        for student_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
            student_name = os.path.basename(student_dir)
            if student_name not in student_set:
                continue
            for gi, gesture in enumerate(GESTURE_CLASSES):
                gesture_dir = os.path.join(student_dir, gesture)
                if not os.path.isdir(gesture_dir):
                    continue
                for clip in sorted(glob.glob(os.path.join(gesture_dir, 'clip_*'))):
                    rgb_folder = os.path.join(clip, 'rgb')
                    ann_folder = os.path.join(clip, 'annotations')
                    if not os.path.isdir(rgb_folder):
                        continue
                    ann_frames = set()
                    if os.path.isdir(ann_folder):
                        ann_frames = {os.path.basename(f) for f in glob.glob(os.path.join(ann_folder, '*.png'))}
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

    def _mask_to_bbox(self, mask_arr):
        coords = np.where(mask_arr > 0.5)
        if len(coords[0]) > 0:
            h, w = mask_arr.shape
            return [coords[1].min()/w, coords[0].min()/h, coords[1].max()/w, coords[0].max()/h]
        return [0.0, 0.0, 0.0, 0.0]

    def __getitem__(self, idx):
        s = self.samples[idx]

        try:
            rgb = Image.open(s['rgb_path']).convert('RGB').resize((self.target_size, self.target_size), Image.BILINEAR)
        except:
            rgb = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))

        try:
            if s['annotation_path'] and os.path.exists(s['annotation_path']):
                mask = Image.open(s['annotation_path']).convert('L').resize((self.target_size, self.target_size), Image.NEAREST)
            else:
                mask = Image.new('L', (self.target_size, self.target_size), 0)
        except:
            mask = Image.new('L', (self.target_size, self.target_size), 0)

        try:
            if os.path.exists(s['depth_raw_path']):
                d_arr = np.load(s['depth_raw_path']).astype(np.float32)
                d_arr = np.clip(d_arr, 0, 2000) / 2000.0
                depth_img = Image.fromarray(d_arr, mode='F').resize((self.target_size, self.target_size), Image.BILINEAR)
            elif os.path.exists(s['depth_png_path']):
                depth_img = Image.open(s['depth_png_path']).convert('I;16')
                d_arr = np.array(depth_img).astype(np.float32) / 65535.0
                depth_img = Image.fromarray(d_arr, mode='F').resize((self.target_size, self.target_size), Image.BILINEAR)
            else:
                depth_img = Image.new('F', (self.target_size, self.target_size), 0)
        except:
            depth_img = Image.new('F', (self.target_size, self.target_size), 0)

        if self.augment and self.aug_transform:
            seed_val = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed_val)
            random.seed(seed_val)
            rgb = self.aug_transform(rgb)
            torch.manual_seed(seed_val)
            random.seed(seed_val)
            if random.random() < 0.5:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                depth_img = depth_img.transpose(Image.FLIP_LEFT_RIGHT)

        rgb_tensor = self.rgb_transform(rgb)
        mask_arr = np.array(mask).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        depth_arr = np.array(depth_img).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_arr).unsqueeze(0)

        bbox = self._mask_to_bbox(mask_arr)
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)

        return {
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'bbox': bbox_tensor,
            'mask': mask_tensor,
            'label': s['gesture_label'],
            'has_annotation': 1 if s['has_annotation'] else 0
        }


class MultiTaskResNet(nn.Module):
    def __init__(self, backbone='resnet34', num_classes=10):
        super().__init__()
        import torchvision.models as models
        if backbone == 'resnet34':
            base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feat_dim = 512
        else:
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feat_dim = 2048

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

        self.det_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 4), nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.seg_up1 = nn.ConvTranspose2d(feat_dim, 256, 2, stride=2)
        self.seg_conv1 = nn.Sequential(nn.Conv2d(256 + 256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU())
        self.seg_up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.seg_conv2 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.seg_up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.seg_conv3 = nn.Sequential(nn.Conv2d(64 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.seg_up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.seg_conv4 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.seg_up5 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.seg_out = nn.Conv2d(16, 1, 1)

    def forward(self, rgb, depth):
        x = torch.cat([rgb, depth], dim=1)
        c1 = self.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(self.maxpool(c1))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        bbox = self.det_head(c5)
        cls_logits = self.cls_head(c5)

        d = self.seg_up1(c5)
        d = self.seg_conv1(torch.cat([d, c4], dim=1))
        d = self.seg_up2(d)
        d = self.seg_conv2(torch.cat([d, c3], dim=1))
        d = self.seg_up3(d)
        d = self.seg_conv3(torch.cat([d, c2], dim=1))
        d = self.seg_up4(d)
        d = self.seg_conv4(torch.cat([d, c1], dim=1))
        d = self.seg_up5(d)
        seg_logits = self.seg_out(d)

        return {'bbox': bbox, 'seg_logits': seg_logits, 'class_logits': cls_logits}


class MultiTaskLoss(nn.Module):
    def __init__(self, w_det=20.0, w_seg=2.0, w_cls=1.0):
        super().__init__()
        self.w_det = w_det
        self.w_seg = w_seg
        self.w_cls = w_cls

    def forward(self, pred, target, has_annotation):
        det_loss = F.smooth_l1_loss(pred['bbox'], target['bbox'])
        seg_logits = F.interpolate(pred['seg_logits'], size=target['mask'].shape[-2:], mode='bilinear', align_corners=False)
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, target['mask'])
        cls_loss = F.cross_entropy(pred['class_logits'], target['label'])
        total = self.w_det * det_loss + self.w_seg * seg_loss + self.w_cls * cls_loss
        return {'total': total, 'detection': det_loss, 'segmentation': seg_loss, 'classification': cls_loss}


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs, writer):
    model.train()
    total_loss = 0
    det_loss_sum = 0
    seg_loss_sum = 0
    cls_loss_sum = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs}')
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

    avg_loss = total_loss / len(loader)
    avg_det = det_loss_sum / len(loader)
    avg_seg = seg_loss_sum / len(loader)
    avg_cls = cls_loss_sum / len(loader)
    avg_acc = 100.0 * correct / total

    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/detection_loss', avg_det, epoch)
    writer.add_scalar('train/segmentation_loss', avg_seg, epoch)
    writer.add_scalar('train/classification_loss', avg_cls, epoch)
    writer.add_scalar('train/accuracy', avg_acc, epoch)
    writer.add_scalars('Loss/total', {'train': avg_loss}, epoch)
    writer.add_scalars('Loss/detection', {'train': avg_det}, epoch)
    writer.add_scalars('Loss/segmentation', {'train': avg_seg}, epoch)
    writer.add_scalars('Loss/classification', {'train': avg_cls}, epoch)
    writer.add_scalars('Accuracy/classification', {'train': avg_acc}, epoch)

    return avg_loss, avg_acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    det_loss_sum = 0
    seg_loss_sum = 0
    cls_loss_sum = 0
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

    avg_loss = total_loss / len(loader)
    avg_det = det_loss_sum / len(loader)
    avg_seg = seg_loss_sum / len(loader)
    avg_cls = cls_loss_sum / len(loader)
    avg_acc = 100.0 * correct / total

    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100.0

    writer.add_scalars('Loss/total', {'val': avg_loss}, epoch)
    writer.add_scalars('Loss/detection', {'val': avg_det}, epoch)
    writer.add_scalars('Loss/segmentation', {'val': avg_seg}, epoch)
    writer.add_scalars('Loss/classification', {'val': avg_cls}, epoch)
    writer.add_scalars('Accuracy/classification', {'val': avg_acc}, epoch)
    writer.add_scalar('Metrics/val_macro_f1', macro_f1, epoch)

    return {
        'loss': avg_loss, 'detection': avg_det, 'segmentation': avg_seg,
        'classification': avg_cls, 'cls_acc': avg_acc, 'macro_f1': macro_f1,
        'preds': all_preds, 'labels': all_labels
    }


def main():
    print("="*70)
    print("v5_with_logging - Identical to v5 + TensorBoard logging")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    print(f"\nConfig: {BACKBONE} | {IMAGE_SIZE}px | {EPOCHS}ep | Det={W_DETECTION}x")
    print("="*70)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    print("\nSplitting students...")
    train_students, val_students = get_student_split(DATA_ROOT, VAL_STUDENT_RATIO, SEED)
    print(f"  Train students: {len(train_students)} | Val students: {len(val_students)}")

    print("Loading data...")
    train_dataset = StudentSplitDataset(DATA_ROOT, train_students, IMAGE_SIZE, augment=True)
    val_dataset = StudentSplitDataset(DATA_ROOT, val_students, IMAGE_SIZE, augment=False)
    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = MultiTaskResNet(backbone=BACKBONE, num_classes=len(GESTURE_CLASSES)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params/1e6:.1f}M")

    criterion = MultiTaskLoss(w_det=W_DETECTION, w_seg=W_SEGMENTATION, w_cls=W_CLASSIFICATION)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(SAVE_DIR, f'v5_logging_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join(LOG_DIR, f'v5_logging_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  TensorBoard log dir: {log_dir}")

    config = {
        'backbone': BACKBONE, 'image_size': IMAGE_SIZE, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE, 'lr': LEARNING_RATE,
        'w_det': W_DETECTION, 'w_seg': W_SEGMENTATION, 'w_cls': W_CLASSIFICATION,
        'train_students': len(train_students), 'val_students': len(val_students),
        'train_samples': len(train_dataset), 'val_samples': len(val_dataset),
        'early_stop_patience': EARLY_STOP_PATIENCE
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    writer.add_text('config', json.dumps(config, indent=2), 0)

    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, EPOCHS, writer)

        val = validate(model, val_loader, criterion, device, epoch, writer)
        writer.add_scalars('Compare_Curve/Loss', {'train_loss': train_loss, 'val_loss': val['loss']}, epoch)
        writer.add_scalars('Compare_Curve/Accuracy', {'train_acc': train_acc, 'val_acc': val['cls_acc']}, epoch)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (elapsed / epoch) * (EPOCHS - epoch)
        lr = optimizer.param_groups[0]['lr']

        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{EPOCHS} | {format_time(epoch_time)} | Total: {format_time(elapsed)} | Remaining: {format_time(remaining)} | LR: {lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.1f}%")
        print(f"  Val   - Loss: {val['loss']:.4f} | Acc: {val['cls_acc']:.1f}% | Macro-F1: {val['macro_f1']:.2f}%")

        if val['loss'] < best_val_loss:
            best_val_loss = val['loss']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': best_val_loss, 'config': config}, os.path.join(save_dir, 'best_model.pth'))
            print(f"  Saved best loss model: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if val['macro_f1'] > best_val_f1:
            best_val_f1 = val['macro_f1']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_f1': best_val_f1, 'config': config}, os.path.join(save_dir, 'best_f1_model.pth'))
            print(f"  Saved best F1 model: {best_val_f1:.2f}%")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered! No improvement for {EARLY_STOP_PATIENCE} epochs.")
            print(f"  Loading best model (val_loss={best_val_loss:.4f})")
            best_ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'))
            model.load_state_dict(best_ckpt['model_state_dict'])
            writer.add_text('early_stopping', f'Triggered at epoch {epoch}, best epoch {best_ckpt["epoch"]}', epoch)
            break

        scheduler.step()

    writer.close()
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Total time: {format_time(total_time)}")
    print(f"Best model: {save_dir}/best_model.pth")
    print(f"TensorBoard: {log_dir}")
    print(f"{'='*70}")
    print(f"\nTo view training curves:")
    print(f"   tensorboard --logdir \"{LOG_DIR}\"")
    print(f"   Then visit: http://localhost:6006")

if __name__ == '__main__':
    main()
