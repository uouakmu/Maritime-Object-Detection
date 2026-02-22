import os
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.metrics import ap_per_class

# ============== Settings (Identical to 5ch) ==============
DATA_DIR = "/dais01/LJY/metaBTS/dataset"
OUTPUT_DIR = "/dais01/LJY/metaBTS/runs/yolov8s_3ch_stable"
EPOCHS = 20
BATCH_SIZE = 4
IMG_SIZE = 640
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT_SPRAY = True

# ============== 1. Dataset Class (Identical Logic, but RGB Only) ==============
class MaritimeDataset3ch(Dataset):
    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.img_root = self.data_dir / 'images'
        self.label_root = self.data_dir / 'annotations'
        
        if not self.img_root.exists():
            print(f"Error: {self.img_root} not found")
            self.sequences = []
        else:
            self.sequences = sorted([p.name for p in self.img_root.iterdir() if p.is_dir()])
            random.seed(42)
            random.shuffle(self.sequences)
            split_idx = int(len(self.sequences) * 0.8)
            self.target_seqs = self.sequences[:split_idx] if split == 'train' else self.sequences[split_idx:]
            print(f"[{split.upper()}] Using {len(self.target_seqs)} sequences")

        self.images_paths = []
        for seq in self.target_seqs:
            seq_dir = self.img_root / seq
            self.images_paths.extend(sorted(list(seq_dir.glob('*.jpg')) + list(seq_dir.glob('*.png'))))
                
        print(f"[{split.upper()}] Caching {len(self.images_paths)} images to RAM...")
        
        self.cache_imgs = []
        self.cache_labels = []
        from multiprocessing.pool import ThreadPool
        
        def load_data(idx):
            img_path = self.images_paths[idx]
            img = cv2.imread(str(img_path))
            if img is None: 
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.img_size, self.img_size))
            
            labels = []
            label_path = self.label_root / img_path.parent.name / f"{img_path.stem}.xml"
            if label_path.exists():
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.parse(str(label_path)).getroot()
                    size = root.find('size')
                    dw = int(size.find('width').text) if size is not None else 1920
                    dh = int(size.find('height').text) if size is not None else 1080
                    for obj in root.findall('object'):
                        b = obj.find('bndbox')
                        if b is not None:
                            xn, yn = float(b.find('xmin').text), float(b.find('ymin').text)
                            xx, yx = float(b.find('xmax').text), float(b.find('ymax').text)
                            labels.append([0, (xn+xx)/2/dw, (yn+yx)/2/dh, (xx-xn)/dw, (yx-yn)/dh])
                except: pass
            return img, np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

        with ThreadPool(8) as pool:
            results = list(tqdm(pool.imap(load_data, range(len(self.images_paths))), total=len(self.images_paths)))
            
        for img, lbl in results:
            self.cache_imgs.append(img)
            self.cache_labels.append(lbl)
        print(f"[{split.upper()}] RAM Cache Complete. Ready. 🚀")
        
        # [SPEED] Pre-compute Noise (CPU 병목 제거)
        print(f"[{split.upper()}] Pre-generating noise options...")
        self.noise_bank = self._precompute_noise_bank(n=100)

    def _precompute_noise_bank(self, n=50):
        bank = []
        for _ in range(n):
            h, w = self.img_size, self.img_size
            overlay = np.zeros((h, w, 3), dtype=np.float32)
            mask = np.zeros((h, w), dtype=np.uint8)
            num_blobs = random.randint(40, 100)
            
            for _ in range(num_blobs):
                cx, cy = random.randint(0, w), random.randint(0, h)
                size = random.randint(10, 100)
                axes = (size, random.randint(5, size//2))
                angle = random.randint(0, 180)
                gray = random.uniform(0.7, 1.0)
                color = (gray, gray, random.uniform(0.8, 1.0))
                cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1)
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
            
            gauss = np.random.normal(0, 0.1, (h, w, 3)).astype(np.float32)
            bank.append({'overlay': overlay, 'gauss': gauss})
        return bank

    def apply_water_spray_fast(self, img_float):
        noise = random.choice(self.noise_bank)
        alpha = random.uniform(0.4, 0.8)
        
        img_noisy = cv2.addWeighted(noise['overlay'], alpha, img_float, 1 - alpha, 0)
        img_noisy = np.clip(img_noisy + noise['gauss'], 0.0, 1.0)
        
        return img_noisy.transpose(2, 0, 1) # Ret HWC->CHW
    
    def __len__(self): return len(self.cache_imgs)
    
    def __getitem__(self, idx):
        img = self.cache_imgs[idx]
        labels = self.cache_labels[idx]
        
        img_float = img.astype(np.float32) / 255.0
        
        if AUGMENT_SPRAY: 
            img_final = self.apply_water_spray_fast(img_float) # Pass float img
        else:
            img_final = img_float.transpose(2, 0, 1)
            
        return {'img': torch.from_numpy(img_final), 'labels': torch.from_numpy(labels)}

def collate_fn(batch):
    imgs = torch.stack([b['img'] for b in batch])
    cls, bboxes, idx = [], [], []
    for i, b in enumerate(batch):
        if len(b['labels']) > 0:
            cls.append(b['labels'][:, 0:1])
            bboxes.append(b['labels'][:, 1:5])
            idx.append(torch.full((len(b['labels']),), i, dtype=torch.float32))
    if cls:
        cls, bboxes, idx = torch.cat(cls, 0), torch.cat(bboxes, 0), torch.cat(idx, 0)
    else:
        cls, bboxes, idx = torch.zeros(0, 1), torch.zeros(0, 4), torch.zeros(0)
    return {'img': imgs, 'cls': cls, 'bboxes': bboxes, 'batch_idx': idx}

# ============== 2. Model Setup (YOLOv8s Baseline) ==============
def get_v8s_model(num_classes=1):
    model = YOLO('yolov8s.pt').model.to(DEVICE)
    model.args = DEFAULT_CFG
    model.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, anchor_t=4.0)
    
    from ultralytics.nn.modules.head import Detect
    det = model.model[-1]
    if isinstance(det, Detect):
        det.nc = num_classes
        det.no = num_classes + det.reg_max * 4
        for i, seq in enumerate(det.cv3):
            old = seq[-1]
            new = nn.Conv2d(old.in_channels, num_classes, 1, 1).to(DEVICE)
            import math
            nn.init.normal_(new.weight, std=0.01)
            new.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            seq[-1] = new
    return model

# ============== 3. Training Loop (Identical Logic) ==============
def validate(model, loader):
    model.eval()
    stats = []
    for b in tqdm(loader, desc="Val"):
        img = b['img'].to(DEVICE)
        with torch.no_grad():
            preds = model(img)
            if isinstance(preds, (list, tuple)): preds = preds[0]
            preds = non_max_suppression(preds, 0.001, 0.7, max_det=300)
        for si, pred in enumerate(preds):
            mask = b['batch_idx'] == si
            tc = b['cls'][mask].view(-1)
            tb = xywh2xyxy(b['bboxes'][mask].view(-1, 4)) * 640
            if len(pred) == 0:
                if len(tc): stats.append((torch.zeros(0, 1, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tc.cpu().numpy()))
                continue
            from ultralytics.utils.metrics import box_iou
            iou = box_iou(pred[:, :4], tb.to(DEVICE))
            matches = iou.max(1)[0] > 0.5 if iou.numel() else torch.zeros(len(pred), dtype=torch.bool).to(DEVICE)
            stats.append((matches.unsqueeze(1).bool().cpu().numpy(), pred[:, 4].cpu().numpy(), pred[:, 5].cpu().numpy(), tc.cpu().numpy()))
    if stats:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if stats[0].any():
            ap = ap_per_class(*stats, plot=False, save_dir=OUTPUT_DIR, names={0:'ship'})[5]
            return ap[:, 0].mean() if len(ap.shape) > 1 else ap.mean()
    return 0.0

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = get_v8s_model(num_classes=1).to(DEVICE)
    loss_fn = v8DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_ds = MaritimeDataset3ch(DATA_DIR, 'train', IMG_SIZE)
    val_ds = MaritimeDataset3ch(DATA_DIR, 'val', IMG_SIZE)
    train_dl = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, BATCH_SIZE * 2, False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print("YOLOv8s 3ch Training Started (Identical Setup)")
    best_map = 0.0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}")
        for b in pbar:
            img = b['img'].to(DEVICE, non_blocking=True)
            batch_gpu = {'img': img, 'cls': b['cls'].to(DEVICE), 'bboxes': b['bboxes'].to(DEVICE), 'batch_idx': b['batch_idx'].to(DEVICE)}
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(img)
                loss, _ = loss_fn(preds, batch_gpu)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")
            
        mAP = validate(model, val_dl)
        print(f"Ep {epoch+1}: Loss={total_loss/len(train_dl):.4f}, mAP50={mAP:.3f}")
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best.pt"))
            print("Best Model Saved")

if __name__ == "__main__":
    train()
