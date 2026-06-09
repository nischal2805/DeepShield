# How To Run — DeepShield

Full instructions for setup, training, inference, and debugging.

---

## 1. Environment Setup (Windows + RTX 4060)

### 1.1 Install PyTorch with CUDA 12.1

```powershell
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX 4060
```

### 1.2 Install remaining dependencies

```powershell
pip install -r requirements.txt
```

**If insightface fails on Windows:**
```powershell
# Install Visual C++ Build Tools first from:
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
pip install insightface --no-build-isolation
```

**If insightface still fails** → facenet-pytorch is the fallback (already in requirements.txt):
```powershell
pip install facenet-pytorch
```
The detector auto-detects which one to use.

### 1.3 Verify imports

```python
python -c "from models.sbi_branch import SBIBranch; print('SBI OK')"
python -c "from models.wavelet_clip import WaveletCLIP; print('WaveletCLIP OK')"
python -c "from models.retinaface_detector import FaceDetector; print('Detector OK')"
```

---

## 2. Download Datasets

Create the data directory:
```powershell
mkdir D:\deepshield_data\ffhq\thumbnails128x128
mkdir D:\deepshield_data\celeb_df_v2
mkdir D:\deepshield_data\wilddeepfake
```

| Dataset | Instructions |
|---|---|
| **FFHQ thumbnails** | Run `python -m gdown --folder` with the FFHQ thumbnails128x128 gdrive ID, or use the official downloader from https://github.com/NVlabs/ffhq-dataset |
| **Celeb-DF v2** | Download from https://github.com/yuezunli/celeb-deepfakeforensics (Google Form) |
| **WildDeepfake** | Download from https://github.com/deepfakeinthewild/deepfake-in-the-wild |

---

## 3. Phase 1 — Preprocess Celeb-DF v2 (Extract Faces)

Before SBI training, extract face crops from Celeb-DF v2 for validation:

```powershell
python -m data.preprocess.extract_faces `
    --input  D:/deepshield_data/celeb_df_v2 `
    --output D:/deepshield_data/celeb_df_v2/faces `
    --fps 1 `
    --size 224 `
    --device cuda
```

Expected output structure:
```
D:/deepshield_data/celeb_df_v2/faces/
├── Celeb-real/videos/*.jpg       (real face crops)
└── Celeb-synthesis/videos/*.jpg  (fake face crops)
```

Rename subdirs to `real/` and `fake/` or update `celeb_df_faces_dir` in `sbi_config.yaml` to point to the parent with those subdirs.

---

## 4. Phase 2 — Train SBI Branch (Local, RTX 4060)

```powershell
python train/train_sbi.py --config configs/sbi_config.yaml
```

**What it does:**
- Epochs 1-5: trains head only (backbone frozen)
- Epochs 6-30: full fine-tune (backbone lr=1e-5, head lr=1e-4)
- fp16 mixed precision + gradient accumulation (effective batch=32)
- Saves best checkpoint to `D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth`

**Expected time:** 6-8 hours on RTX 4060

**OOM troubleshooting:**
```yaml
# In configs/sbi_config.yaml — reduce memory:
training:
  batch_size: 8          # was 16
  grad_accum_steps: 4    # was 2 — effective batch stays 32
  img_size: 192          # was 224 — last resort
```

**Monitor training:**
```powershell
# Enable wandb (set API key first):
# setx WANDB_API_KEY "your-key"
# Then in sbi_config.yaml: use_wandb: true
```

---

## 5. Phase 3 — Train Wavelet-CLIP (Kaggle Notebook)

### 5.1 Upload repo to GitHub

```powershell
cd E:\wic\deepshield
git remote add origin https://github.com/YOUR_USERNAME/deepshield.git
git push -u origin main
```

### 5.2 Set up Kaggle datasets

1. Upload your FF++ download as a private Kaggle dataset named `faceforensics-pp`
2. Upload your Celeb-DF v2 as a private Kaggle dataset named `celeb-df-v2`
3. The DFDC dataset is already public on Kaggle: `deepfake-detection-challenge`

### 5.3 Run the notebook

1. Open `notebooks/train_wavelet_clip_kaggle.ipynb` on Kaggle
2. Mount all three datasets (Settings > Data > Add Data)
3. Enable GPU T4×2 or P100 (Settings > Accelerator)
4. Replace `YOUR_USERNAME` in Cell 2 with your GitHub username
5. Run all cells in order

**Kaggle space management:**
- CLIP weights: ~1.7 GB (auto-downloaded)
- Face crops: ~8 GB (deleted after training if needed)
- Checkpoints: ~2 GB
- Total working dir: stays under 20 GB

**If Kaggle runs out of disk:**
- Reduce `MAX_DFDC_VIDEOS` from 3000 to 1500 in Cell 5
- The script checks free space every 100 videos and stops automatically

### 5.4 Download checkpoint

Cell 12 generates a download link. Save to:
```
D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth
```

---

## 6. Phase 4 — Fit Ensemble Weights

```powershell
python train/train_ensemble.py `
    --sbi-ckpt    D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth `
    --wclip-ckpt  D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth `
    --celeb-faces D:/deepshield_data/celeb_df_v2/faces `
    --output      checkpoints/ensemble_weights.json
```

Takes ~2-5 minutes. Prints fitted weights and ensemble AUC.

---

## 7. Phase 5 — Evaluate on WildDeepfake

First extract WildDeepfake faces:
```powershell
python -m data.preprocess.extract_faces `
    --input  D:/deepshield_data/wilddeepfake `
    --output D:/deepshield_data/wilddeepfake/faces `
    --fps 1 --device cuda
```

Rename subdirs to `val/real/` and `val/fake/`, then:

```powershell
python evaluate/eval.py `
    --sbi-ckpt    D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth `
    --wclip-ckpt  D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth `
    --weights     checkpoints/ensemble_weights.json `
    --test-dir    D:/deepshield_data/wilddeepfake/faces `
    --output      eval_results.json
```

Target metrics: AUC > 0.80, FNR < 0.15, latency < 200ms.

---

## 8. Run the Streamlit App

```powershell
streamlit run app.py
```

Opens at `http://localhost:8501`

**Override checkpoint paths via env vars:**
```powershell
$env:SBI_CKPT   = "D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth"
$env:WCLIP_CKPT = "D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth"
$env:WEIGHTS    = "checkpoints/ensemble_weights.json"
streamlit run app.py
```

---

## 9. CLI Single-Image Inference

```powershell
python inference/pipeline.py --image path/to/image.jpg

# Override paths if checkpoints not at default locations:
python inference/pipeline.py `
    --image       path/to/image.jpg `
    --sbi-ckpt    D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth `
    --wclip-ckpt  D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth `
    --threshold   0.5
```

Output:
```json
{
  "verdict": "FAKE",
  "confidence": 0.8723,
  "score_sbi": 0.821,
  "score_wclip": 0.908,
  "face_bbox": [120, 80, 380, 340],
  "track": 1,
  "latency_ms": 87.3
}
```

Heatmap saved as `<image_name>_heatmap.png` if pytorch-grad-cam is installed.

---

## 10. Train Wavelet-CLIP Locally (alternative to Kaggle)

If you have ≥12 GB VRAM locally:

First extract faces from FF++ and DFDC to `D:/deepshield_data/faces/`:
```powershell
python -m data.preprocess.extract_faces --input D:/deepshield_data/faceforensics --output D:/deepshield_data/faces/train --fps 1 --device cuda
```

Then train:
```powershell
python train/train_wavelet_clip.py --config configs/wavelet_clip_config.yaml
```

---

## 11. Debugging Guide

### "CUDA out of memory" during SBI training
```yaml
# sbi_config.yaml
training:
  batch_size: 8
  grad_accum_steps: 4
  img_size: 192
```

### "No face detected" for every image
- Check that insightface or facenet-pytorch is installed
- Test detector: `python -c "from models.retinaface_detector import FaceDetector; d=FaceDetector(); print(d.backend)"`
- If insightface backend: check onnxruntime-gpu is installed

### DataLoader worker crash on Windows
- Set `num_workers: 0` in config for debugging (slower but no multiprocessing)
- Ensure `persistent_workers: false` in config (required on Windows)
- Ensure `if __name__ == '__main__':` guard is present (it is, in all train scripts)

### CLIP download fails on Kaggle
- The model downloads from HuggingFace on first run (~1.7 GB)
- If network blocked, pre-download and upload as a Kaggle dataset:
  ```python
  from transformers import CLIPModel
  CLIPModel.from_pretrained("openai/clip-vit-large-patch14").save_pretrained("./clip_model")
  ```
  Then upload `./clip_model` as a Kaggle dataset and modify `clip_model_name` in `WaveletCLIP.__init__` to point to `/kaggle/input/clip-vit-l14/`.

### "Module not found" errors when running scripts
All scripts add the project root to `sys.path` automatically. Run from the `deepshield/` directory:
```powershell
cd E:\wic\deepshield
python train/train_sbi.py --config configs/sbi_config.yaml
```

### Wavelet-CLIP NaN loss during training
- Already handled: FocalLoss casts result to float32 before `.mean()`
- If NaN persists: disable mixed precision (`mixed_precision: false` in config)
- Check that CLIP normalization is correct — mismatched norms cause NaN silently

### Track 2 (SynthID) returns "API key not set"
Set environment variables:
```powershell
$env:SYNTHID_API_KEY = "your-google-api-key"
$env:SYNTHID_PROJECT = "your-gcp-project-id"
```
C2PA check works without API keys (reads file metadata only).
