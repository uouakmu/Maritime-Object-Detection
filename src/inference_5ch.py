import torch
import torch.nn as nn
import cv2
import numpy as np
import time
import random
import sys
import os
import os
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from types import SimpleNamespace

# ============== Settings ==============
WEIGHTS_PATH = "/dais01/LJY/metaBTS/runs/two_stream_v3_stable/best.pt"
SOURCE_DIR = "/dais01/LJY/metaBTS/dataset/images"
OUTPUT_VIDEO = "output_5ch_two_stream.mp4"
IMG_SIZE = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRES = 0.5

# ============== Model Definition (Must Match Training) ==============
class FusionBlock(nn.Module):
    def __init__(self, c_rgb, c_flow, c_out):
        super().__init__()
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
        print("Rebuilding TwoStream Model...")
        rgb_yolo = YOLO(rgb_model)
        self.rgb_backbone = rgb_yolo.model.model[:10]
        flow_yolo = YOLO(flow_model)
        self.flow_backbone = flow_yolo.model.model[:10]
        
        # Modify Flow Input (2ch)
        c1 = self.flow_backbone[0].conv
        new_c1 = nn.Conv2d(2, c1.out_channels, c1.kernel_size, c1.stride, c1.padding, bias=c1.bias is not None)
        self.flow_backbone[0].conv = new_c1

        # Channel Check (Mock)
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
        
        # Head Update logic (simplified for inference reconstruction, assuming weights match)
        # Note: We just need structure. Weights load will handle values.
        # But we need to ensure layer shapes match.
        from ultralytics.nn.modules.head import Detect
        det = self.head_layers[-1]
        if isinstance(det, Detect):
            det.nc = num_classes
            det.no = num_classes + det.reg_max * 4
            for i, seq in enumerate(det.cv3):
                 old = seq[-1]
                 new = nn.Conv2d(old.in_channels, num_classes, 1, 1)
                 seq[-1] = new

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

class InferencePipeline:
    def __init__(self):
        # Pre-compute noise
        print("Pre-generating noise bank...")
        self.noise_bank = self._precompute_noise_bank()
        
    def _precompute_noise_bank(self, n=100):
        bank = []
        for _ in range(n):
            h, w = IMG_SIZE, IMG_SIZE
            overlay = np.zeros((h, w, 3), dtype=np.float32)
            num_blobs = random.randint(40, 100)
            for _ in range(num_blobs):
                cx, cy = random.randint(0, w), random.randint(0, h)
                size = random.randint(10, 100)
                axes = (size, random.randint(5, size//2))
                angle = random.randint(0, 180)
                gray = random.uniform(0.7, 1.0)
                color = (gray, gray, random.uniform(0.8, 1.0))
                cv2.ellipse(overlay, (cx, cy), axes, angle, 0, 360, color, -1)
            gauss = np.random.normal(0, 0.1, (h, w, 3)).astype(np.float32)
            bank.append({'overlay': overlay, 'gauss': gauss})
        return bank

    def apply_noise_fast(self, img_bgr):
        img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_float = img_resized.astype(np.float32) / 255.0
        # img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB) # Torch uses RGB usually, but let's stick to consistency
        # Actually YOLO usually expects RGB. OpenCV is BGR.
        # My training script used cv2.imread -> cv2.cvtColor(..., BGR2RGB).
        img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        
        noise = random.choice(self.noise_bank)
        alpha = random.uniform(0.4, 0.8)
        
        # Add noise
        img_noisy = cv2.addWeighted(noise['overlay'], alpha, img_rgb, 1 - alpha, 0)
        img_noisy = np.clip(img_noisy + noise['gauss'], 0.0, 1.0)
        
        return img_noisy # (H,W,3) RGB float32

    def compute_flow_fast(self, prev_rgb_float, curr_rgb_float):
        # prev, curr: (H,W,3) RGB float32
        p_gray = cv2.cvtColor(prev_rgb_float, cv2.COLOR_RGB2GRAY)
        c_gray = cv2.cvtColor(curr_rgb_float, cv2.COLOR_RGB2GRAY)
        
        # Downscale 0.5x
        scale = 0.5
        h, w = p_gray.shape
        sh, sw = int(h*scale), int(w*scale)
        p_small = cv2.resize(p_gray, (sw, sh))
        c_small = cv2.resize(c_gray, (sw, sh))
        
        # Farneback
        flow_small = cv2.calcOpticalFlowFarneback(p_small, c_small, None, 0.5, 1, 12, 1, 5, 1.1, 0)
        
        # Upscale
        flow = cv2.resize(flow_small, (w, h))
        flow = flow * (1.0/scale) # Adjust magnitude
        
        # Clip & Transpose
        flow = np.clip(flow / 20.0, -1.0, 1.0)
        return flow.transpose(2, 0, 1) # (2,H,W)

def main():
    # 1. Load Model
    model = TwoStreamYOLO().to(DEVICE)
    try:
        # Debugging Load
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        
        if isinstance(ckpt, dict) and 'model' in ckpt:
            sd = ckpt['model']
            state_dict = sd.state_dict() if hasattr(sd, 'state_dict') else sd
        else:
            state_dict = ckpt
            
        # [CLEANUP] Remove redundant 'model.x' keys from training script
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('model.')}
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Weights Loaded (Strict=False)")
        if missing: print(f"⚠️ Missing Keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected: print(f"⚠️ Unexpected Keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        
        # If crucial parts are missing, we should stop
        if any('fusion' in k for k in missing):
             print("❌ CRITICAL: Fusion layers are missing in weights! This is likely a 3ch model file.")
             # return # For debugging, let's proceed to see what happens, or stop. Ideally stop.
             return

    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return
    model.eval()

    # 2. Data Setup
    pipeline = InferencePipeline()
    img_paths = sorted(list(Path(SOURCE_DIR).rglob("*.jpg")) + list(Path(SOURCE_DIR).rglob("*.png")))
    test_images = img_paths[:500] 
    print(f"Processing {len(test_images)} frames...")

    # 3. Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (IMG_SIZE, IMG_SIZE))

    # 4. Loop
    prev_rgb_noisy = None
    frame_count = 0
    start_time = time.time()
    
    for i, p in enumerate(test_images):
        img_orig = cv2.imread(str(p))
        if img_orig is None: continue
        
        # A. Apply Noise (Independent for each frame, but usually sequential video)
        # To simulate leakage-free, we compute flow between (Noise(t-1), Noise(t))
        curr_rgb_noisy = pipeline.apply_noise_fast(img_orig)
        
        # Handle first frame
        if prev_rgb_noisy is None:
            prev_rgb_noisy = curr_rgb_noisy.copy()
            
        # B. Compute Flow (t-1 -> t)
        t_start = time.time()
        
        flow_tensor = pipeline.compute_flow_fast(prev_rgb_noisy, curr_rgb_noisy)
        flow_tensor = torch.from_numpy(flow_tensor).unsqueeze(0).to(DEVICE) # (1,2,H,W)
        
        # C. Prepare Image Tensor
        img_tensor = torch.from_numpy(curr_rgb_noisy.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE) # (1,3,H,W)
        
        # D. Inference
        with torch.no_grad():
            preds = model(img_tensor, flow_tensor)
            # 영상 출력 시에는 가독성을 위해 CONF_THRES(0.5) 이상만 표시
            preds = non_max_suppression(preds, CONF_THRES, 0.5, max_det=300)
            
        t_end = time.time()
        
        # E. Visualization
        # Convert RGB float back to BGR uint8 for OpenCV
        vis_img = cv2.cvtColor(curr_rgb_noisy, cv2.COLOR_RGB2BGR)
        vis_img = (vis_img * 255).astype(np.uint8)
        
        # Draw Boxes
        for det in preds:
            if len(det):
                det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], vis_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(vis_img, c1, c2, (0, 255, 0), 2)
                    label = f"Ship {conf:.2f}"
                    cv2.putText(vis_img, label, (c1[0], c1[1] - 2), 0, 0.5, (0, 255, 0), 1)

        fps = 1.0 / (t_end - t_start)
        cv2.putText(vis_img, f"FPS: {fps:.1f} (Two-Stream)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(vis_img)
        prev_rgb_noisy = curr_rgb_noisy
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames, Current FPS: {fps:.1f}")

    out.release()
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f"\nDone! Output saved to {OUTPUT_VIDEO}")
    print(f"Average FPS: {avg_fps:.2f}")

    # 5. Full Evaluation (Generate PNGs)
    print("\nStarting Full Evaluation for Metrics (mAP, Confusion Matrix)...")
    from ultralytics.utils.metrics import ap_per_class
    from ultralytics.utils.ops import xywh2xyxy
    
    # Need to re-instantiate Dataset/Loader logic from training to get labels properly
    # For simplicity, we assume we just need to run the validation loop used in training
    # We will copy the critical parts of validation logic here
    
    # Define simple dataset class wrapper if not importing from train script
    # To save time, we will assume train_5ch.py is importable or just copy validate logic
    # Let's copy the validate logic self-contained to avoid import errors
    
    def validate_and_plot(model, data_dir, device):
        # We need a proper DataLoader that returns labels
        # Re-using the logic from train_5ch.py's validation is best way
        # But we need to make sure we don't break dry-run
        pass # Placeholder for actual implementation in next block

    # Ideally we import from train_5ch.py but it might run training code on import if not careful.
    # train_5ch.py has if __name__ == "__main__", so it sits safe.
    try:
        # Add 'two_stream' folder to path, as train scripts are there
        script_dir = os.path.dirname(os.path.abspath(__file__))
        two_stream_dir = os.path.join(script_dir, "two_stream")
        if os.path.exists(two_stream_dir):
            sys.path.append(two_stream_dir)
        else:
             sys.path.append(os.path.join(os.getcwd(), "two_stream"))

        from train_5ch import TwoStreamDataset, collate_fn, validate
        from torch.utils.data import DataLoader
        
        val_ds = TwoStreamDataset(f"/dais01/LJY/metaBTS/dataset", 'val', IMG_SIZE)
        val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)
        
        print("Running Validation...")
        # Note: The validate function in train_5ch.py returns mAP but might not save all plots unless configured.
        # We might need to enhance it or write a custom one here.
        # Ultralytics ap_per_class saves plots if save_dir is provided.
        # Let's use a custom output dir for inference results
        
        save_dir = Path("inference_results_5ch")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model.eval()
        stats = []
        for b in val_dl:
            img, flow = b['img'].to(DEVICE), b['flow'].to(DEVICE)
            with torch.no_grad():
                preds = model(img, flow)
                preds = non_max_suppression(preds, 0.001, 0.5)
            
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
                stats.append((matches.unsqueeze(1).bool().cpu().numpy(), pred[:,4].cpu().numpy(), pred[:,5].cpu().numpy(), tc.cpu().numpy()))
        
        if stats:
            stats = [np.concatenate(x, 0) for x in zip(*stats)]
            if stats[0].any():
                # save_dir must be Path object
                ap_per_class(*stats, plot=True, save_dir=save_dir, names={0:'ship'})
                print(f"✅ Metrics plots saved to {save_dir}")
                
    except ImportError:
        print("⚠️  Could not import validation logic from train_5ch.py. Make sure it is in the same directory.")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")

    # Call the validation
    validate_and_plot(model, "", DEVICE)

if __name__ == "__main__":
    main()
