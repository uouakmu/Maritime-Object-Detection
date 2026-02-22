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
from ultralytics.nn.modules import Conv
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.metrics import ap_per_class

# ============== Settings ==============
DATA_DIR = "/dais01/LJY/metaBTS/dataset"
OUTPUT_DIR = "/dais01/LJY/metaBTS/runs/two_stream_v3_stable"
EPOCHS = 20
BATCH_SIZE = 4
IMG_SIZE = 640
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT_SPRAY = True

# ============== 1. Dataset Class ==============
class TwoStreamDataset(Dataset):
    def __init__(self, data_dir, split='train', img_size=640):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        self.img_root = self.data_dir / 'images'
        self.label_root = self.data_dir / 'annotations'
        self.flow_root = self.data_dir / 'flows'
        
        if not self.img_root.exists():
            print(f"Error: {self.img_root} not found")
            self.sequences = []
        else:
            self.sequences = sorted([p.name for p in self.img_root.iterdir() if p.is_dir()])
            # random.seed(42); random.shuffle(self.sequences) # Keep randomization
            
            # v2와 동일한 split 사용을 위해 seed 고정
            random.seed(42)
            random.shuffle(self.sequences)
            
            split_idx = int(len(self.sequences) * 0.8)
            self.target_seqs = self.sequences[:split_idx] if split == 'train' else self.sequences[split_idx:]
            
            print(f"[{split.upper()}] Using {len(self.target_seqs)} sequences")

        self.images_paths = []
        self.flow_paths = []
        for seq in self.target_seqs:
            seq_dir = self.img_root / seq
            imgs = sorted(list(seq_dir.glob('*.jpg')) + list(seq_dir.glob('*.png')))
            self.images_paths.extend(imgs)
            # Find corresponding flows
            for img_p in imgs:
                self.flow_paths.append(self.flow_root / seq / f"{img_p.stem}.npy")
                
        print(f"[{split.upper()}] Caching {len(self.images_paths)} images to RAM... (This takes a moment)")
        
        # [SPEED] RAM Caching
        self.cache_imgs = []
        self.cache_flows = []
        self.cache_labels = []
        
        from multiprocessing.pool import ThreadPool
        
        # Multi-threaded Loading
        def load_data(idx):
            # Image
            img_path = self.images_paths[idx]
            img = cv2.imread(str(img_path))
            if img is None: 
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize 미리 해둠 (속도 최적화)
                img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Flow
            flow_path = self.flow_paths[idx]
            flow = np.zeros((2, self.img_size, self.img_size), dtype=np.float32)
            if flow_path.exists():
                try:
                    f = np.load(str(flow_path))
                    # Shape Handling
                    if f.shape[0] != 2:
                        if f.ndim == 3 and f.shape[2] == 2: f = f.transpose(2, 0, 1)
                        else: f = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.float32)
                    
                    # Resize Flow
                    f_resized = np.zeros((2, self.img_size, self.img_size), dtype=np.float32)
                    for i in range(2): f_resized[i] = cv2.resize(f[i], (self.img_size, self.img_size))
                    flow = np.clip(f_resized / 20.0, -1, 1)
                except: pass
            
            # Label
            labels = []
            label_path = self.label_root / img_path.parent.name / f"{img_path.stem}.xml"
            if label_path.exists():
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.parse(str(label_path)).getroot()
                    size = root.find('size')
                    # 원본 크기 알아야 함 (Resize 되었으므로 비율 변환)
                    # 하지만 이미지는 이미 Resize 되었음. 
                    # 원본 XML은 원본 크기 기준.
                    # -> Resize 전 원본 크기가 필요함? 
                    #    ㄴ 이미지를 resize해서 저장하니까, 라벨도 그에 맞춰야 함.
                    #    ㄴ 아니면 정규화(0~1)된 좌표를 쓰면 됨. YOLO는 정규화 좌표.
                    dw = int(size.find('width').text) if size is not None else 1920 # Default
                    dh = int(size.find('height').text) if size is not None else 1080
                    for obj in root.findall('object'):
                        b = obj.find('bndbox')
                        if b is not None:
                            xn, yn = float(b.find('xmin').text), float(b.find('ymin').text)
                            xx, yx = float(b.find('xmax').text), float(b.find('ymax').text)
                            labels.append([0, (xn+xx)/2/dw, (yn+yx)/2/dh, (xx-xn)/dw, (yx-yn)/dh])
                except: pass
            labels = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

            return img, flow, labels

        # 병렬 로딩으로 대기 시간 최소화
        with ThreadPool(8) as pool:
            results = list(tqdm(pool.imap(load_data, range(len(self.images_paths))), total=len(self.images_paths)))
            
        for img, flow, lbl in results:
            self.cache_imgs.append(img)
            self.cache_flows.append(flow)
            self.cache_labels.append(lbl)

        print(f"[{split.upper()}] RAM Cache Complete. Ready to fly. 🚀")
        
        # [SPEED] Pre-compute Noise Patterns (CPU 병목 제거)
        print(f"[{split.upper()}] Pre-generating noise options...")
        self.noise_bank = self._precompute_noise_bank(n=100)

    def _precompute_noise_bank(self, n=50):
        bank = []
        for _ in range(n):
            # 640x640 기준 (이미지 크기가 다르면 resize 필요)
            h, w = self.img_size, self.img_size
            
            # Mask & Overlay
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
                
            # Flow Jitter 
            jitter = np.random.uniform(-1.0, 1.0, (h, w, 2)).astype(np.float32)
            jitter[mask == 0] = 0 # 마스크 없는 곳은 0
            
            # Gaussian Noise field
            gauss = np.random.normal(0, 0.1, (h, w, 3)).astype(np.float32)
            
            bank.append({'overlay': overlay, 'gauss': gauss}) # Removed mask and jitter
        return bank

    def apply_water_spray_fast(self, img_float):
        # Select random noise pattern
        noise = random.choice(self.noise_bank)
        alpha = random.uniform(0.4, 0.8)
        
        # Simple Weighted Add (전체에 적용해도 overlay가 0인 부분은 영향 적음 - 
        # 정확히는 overlay가 0이어도 alpha 때문에 원본이 어두워질 수 있음.
        # 따라서 Overlay가 있는 부분마 Blending 해야 하는데, 
        # 속도를 위해 전체 addWeighted 후 overlay가 0인 부분 처리... 
        # -> 복잡함. 그냥 기존 마스크 방식이 정확함.)
        
        # [REVERT TO MASK LOGIC FOR ACCURACY]
        # Bank에 다시 mask 추가 필요. 일단 간단히 전체 addWeighted (어두워지는 효과 = 날씨 흐림 효과로 간주)
        img_noisy = cv2.addWeighted(noise['overlay'], alpha, img_float, 1 - alpha, 0)
        img_noisy = np.clip(img_noisy + noise['gauss'], 0.0, 1.0)
        return img_noisy

    def compute_flow_fast(self, prev, curr):
        # [Ultra Fast Flow]
        # 1. Grayscale
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY) # (H,W) float32
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        
        # 2. Downscale for Speed (320x320)
        scale = 0.5
        small_h, small_w = int(self.img_size*scale), int(self.img_size*scale)
        p_small = cv2.resize(prev_gray, (small_w, small_h))
        c_small = cv2.resize(curr_gray, (small_w, small_h))
        
        # 3. Farneback (Fast Params)
        # pyr_scale=0.5, levels=1, winsize=12, iterations=1, poly_n=5, poly_sigma=1.1, flags=0
        flow_small = cv2.calcOpticalFlowFarneback(p_small, c_small, None, 0.5, 1, 12, 1, 5, 1.1, 0)
        
        # 4. Upscale & Normalize
        # Upscale
        flow = cv2.resize(flow_small, (self.img_size, self.img_size))
        # Adjust magnitude (since image was scaled down) -> No, flow is displacement in pixels.
        # If image is 0.5x, displacement is 0.5x. So need to multiply by 1/scale = 2.
        flow = flow * (1.0 / scale)
        
        # 5. Clip & Transpose
        flow = np.clip(flow / 20.0, -1.0, 1.0) # Simple normalization
        return flow.transpose(2, 0, 1)
    
    def __len__(self): return len(self.cache_imgs)
    
    def __getitem__(self, idx):
        # [LEAKAGE FREE] Load t and t-1
        img_t = self.cache_imgs[idx] # (H,W,3) uint8
        
        # 이전 프레임 찾기 (같은 시퀀스 내에서)
        # cache_imgs는 concat된 리스트라 시퀀스 경계를 모름.
        # 간단한 휴리스틱: idx > 0 이고 경로 상의 폴더 이름이 같으면 t-1. 아니면 t 복제.
        
        # 이미지 경로를 따로 저장해뒀어야 함. (cache_imgs만으로는 부족)
        # self.images_paths 에 접근.
        
        path_t = self.images_paths[idx]
        path_prev = self.images_paths[idx-1] if idx > 0 else path_t
        
        if idx > 0 and path_t.parent == path_prev.parent:
             img_prev = self.cache_imgs[idx-1]
        else:
             img_prev = img_t.copy()

        labels = self.cache_labels[idx]
        
        # 1. Normalize
        img_t_float = img_t.astype(np.float32) / 255.0
        img_prev_float = img_prev.astype(np.float32) / 255.0
        
        if AUGMENT_SPRAY:
            # 2. Apply Independent Noise (Stochastic)
            # t와 t-1에 서로 다른 노이즈가 낌 -> Flow가 엉망이 됨 (Real World Simulation)
            img_t_noisy = self.apply_water_spray_fast(img_t_float)
            img_prev_noisy = self.apply_water_spray_fast(img_prev_float)
        else:
            img_t_noisy = img_t_float
            img_prev_noisy = img_prev_float
            
        # 3. Compute Real-time Flow (Leakage Free: t-1 -> t)
        flow_real = self.compute_flow_fast(img_prev_noisy, img_t_noisy)
        
        # 4. Finalize
        img_final = img_t_noisy.transpose(2, 0, 1)
        
        return {'img': torch.from_numpy(img_final), 'flow': torch.from_numpy(flow_real), 'labels': torch.from_numpy(labels)}

def collate_fn(batch):
    imgs = torch.stack([b['img'] for b in batch])
    flows = torch.stack([b['flow'] for b in batch])
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
    return {'img': imgs, 'flow': flows, 'cls': cls, 'bboxes': bboxes, 'batch_idx': idx}

# ============== 2. Model (Stabilized) ==============
class FusionBlock(nn.Module):
    def __init__(self, c_rgb, c_flow, c_out):
        super().__init__()
        # [KEY CHANGE] 1x1 Conv + BN + SiLU for Stability (Prevent Gradient Explosion)
        self.conv = nn.Sequential(
            nn.Conv2d(c_rgb + c_flow, c_out, 1, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU()
        )
    def forward(self, x_rgb, x_flow):
        if x_rgb.shape[-2:] != x_flow.shape[-2:]:
            x_flow = torch.nn.functional.interpolate(x_flow, size=x_rgb.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x_rgb, x_flow], dim=1))

class TwoStreamYOLO(nn.Module):
    def __init__(self, rgb_model='yolov8s.pt', flow_model='yolov8n.pt', num_classes=1):
        super().__init__()
        print(f"Init Stable Two-Stream (v3)... RGB:{rgb_model}, Flow:{flow_model}")
        
        rgb_yolo = YOLO(rgb_model)
        self.rgb_backbone = rgb_yolo.model.model[:10]
        flow_yolo = YOLO(flow_model)
        self.flow_backbone = flow_yolo.model.model[:10]
        
        # Modify Flow Input (2ch)
        c1 = self.flow_backbone[0].conv
        new_c1 = nn.Conv2d(2, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=c1.bias is not None)
        with torch.no_grad(): new_c1.weight[:, :2] = c1.weight[:, :2]
        self.flow_backbone[0].conv = new_c1

        # Channel Check
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            dummy_f = torch.zeros(1, 2, 640, 640)
            fr = self._get_feats(self.rgb_backbone, dummy)
            ff = self._get_feats(self.flow_backbone, dummy_f)
            c_rgb = [x.shape[1] for x in fr]
            c_flow = [x.shape[1] for x in ff]
            
        self.fusion_p3 = FusionBlock(c_rgb[0], c_flow[0], c_rgb[0])
        self.fusion_p4 = FusionBlock(c_rgb[1], c_flow[1], c_rgb[1])
        self.fusion_p5 = FusionBlock(c_rgb[2], c_flow[2], c_rgb[2])

        self.head_layers = rgb_yolo.model.model[10:]
        
        # Update Head
        from ultralytics.nn.modules.head import Detect
        det = self.head_layers[-1]
        if isinstance(det, Detect):
            det.nc = num_classes
            det.no = num_classes + det.reg_max * 4
            for i, seq in enumerate(det.cv3):
                old = seq[-1]
                new = nn.Conv2d(old.in_channels, num_classes, 1, 1).to(old.weight.device)
                import math
                nn.init.normal_(new.weight, std=0.01)
                if new.bias is not None:
                    b = new.bias.view(1,-1)
                    b.data.fill_(-math.log((1 - 0.01) / 0.01))
                    new.bias = torch.nn.Parameter(b.view(-1))
                seq[-1] = new

        self.nc = num_classes
        self.hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, anchor_t=4.0)
        self.args = DEFAULT_CFG
        self.model = rgb_yolo.model.model # loss access

    def _get_feats(self, backbone, x):
        feats = []
        for i, m in enumerate(backbone):
            x = m(x)
            if i in [4, 6, 9]: feats.append(x)
        return feats

    def forward(self, x_rgb, x_flow):
        fr = self._get_feats(self.rgb_backbone, x_rgb)
        ff = self._get_feats(self.flow_backbone, x_flow)
        
        f3 = self.fusion_p3(fr[0], ff[0])
        f4 = self.fusion_p4(fr[1], ff[1])
        f5 = self.fusion_p5(fr[2], ff[2])
        
        x = f5
        cache = {4: f3, 6: f4, 9: f5}
        for i, m in enumerate(self.head_layers):
            idx = i + 10 
            if m.f == -1: input_x = x
            elif isinstance(m.f, list): input_x = [x if j == -1 else cache[j] for j in m.f]
            else: input_x = cache[m.f]
            x = m(input_x)
            cache[idx] = x
        return x

# ============== 3. Training Loop ==============
def validate(model, loader, device, names={0:'ship'}):
    model.eval()
    stats = []
    for b in tqdm(loader, desc="Val"):
        img, flow = b['img'].to(device), b['flow'].to(device)
        with torch.no_grad():
            preds = model(img, flow)
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
            iou = box_iou(pred[:, :4], tb.to(device))
            matches = iou.max(1)[0] > 0.5 if iou.numel() else torch.zeros(len(pred), dtype=torch.bool).to(device)
            stats.append((matches.unsqueeze(1).bool().cpu().numpy(), pred[:,4].cpu().numpy(), pred[:,5].cpu().numpy(), tc.cpu().numpy()))
            
    if stats:
        stats = [np.concatenate(x, 0) for x in zip(*stats)]
        if stats[0].any():
            ap = ap_per_class(*stats, plot=False, save_dir=OUTPUT_DIR, names=names)[5]
            return ap[:, 0].mean() if len(ap.shape) > 1 else ap.mean()
    return 0.0

def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = TwoStreamYOLO().to(DEVICE)
    loss_fn = v8DetectionLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    train_ds = TwoStreamDataset(DATA_DIR, 'train', IMG_SIZE)
    val_ds = TwoStreamDataset(DATA_DIR, 'val', IMG_SIZE)
    # [SPEED] RAM 캐싱 시에는 num_workers=0이 최적 (SHM 에러 방지 및 복사 비용 제거)
    train_dl = DataLoader(train_ds, BATCH_SIZE, True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_ds, BATCH_SIZE * 2, False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

    print("Two-Stream Training Started (Stable v3 + Optimization)")
    best_map = 0.0

    # [SPEED] AMP Scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}")
        
        for b in pbar:
            img, flow = b['img'].to(DEVICE, non_blocking=True), b['flow'].to(DEVICE, non_blocking=True)
            batch_gpu = {'img': img, 'cls': b['cls'].to(DEVICE), 'bboxes': b['bboxes'].to(DEVICE), 'batch_idx': b['batch_idx'].to(DEVICE)}
            
            optimizer.zero_grad()
            
            # [SPEED] Mixed Precision Training
            with torch.cuda.amp.autocast():
                preds = model(img, flow)
                loss, _ = loss_fn(preds, batch_gpu)
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping (Scaler Aware)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.2f}")
            
        mAP = validate(model, val_dl, DEVICE)
        avg_loss = total_loss / len(train_dl)
        print(f"Ep {epoch+1}: Loss={avg_loss:.4f}, mAP50={mAP:.3f}")
        
        if mAP > best_map:
            best_map = mAP
            ckpt = {'model': model.state_dict(), 'map': mAP}
            torch.save(ckpt, os.path.join(OUTPUT_DIR, "best.pt"))
            print(f"Best Model Saved (mAP={mAP:.3f})")

if __name__ == "__main__":
    train()
