# DeepShield

Dual-model deepfake and AI-generated image detection system.

- **Track 1:** Face forgery detection — Wavelet-CLIP + SBI ensemble
- **Track 2:** AI-generated image detection — C2PA metadata + SynthID
- **Output:** REAL / FAKE / AI-GENERATED verdict + confidence + Grad-CAM heatmap

## Architecture

```
Image
  │
  ├─ Face detected? ─── YES ──► Track 1
  │                               ├── SBI Branch (EfficientNet-B4, trained locally)
  │                               ├── Wavelet-CLIP Branch (trained on Kaggle)
  │                               └── Weighted ensemble → verdict + Grad-CAM
  │
  └─────────────────── NO ───► Track 2
                                 ├── C2PA metadata check
                                 └── Google SynthID watermark check
```

## Quick Start

```bash
# 1. Install PyTorch (CUDA 12.1)
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Run Streamlit app (requires trained checkpoints)
streamlit run app.py

# 4. Single-image CLI inference
python inference/pipeline.py --image path/to/image.jpg
```

## Training Order

1. **Phase 1** — SBI branch, local Windows RTX 4060 (~6-8h)
2. **Phase 2** — Wavelet-CLIP, Kaggle notebook (~4-6h on T4)
3. **Phase 3** — Ensemble weights, local (~5 min)
4. **Phase 4** — Evaluation on WildDeepfake

See `HOW_TO_RUN.md` for full instructions.

## Datasets Required

| Dataset | Purpose | Where |
|---|---|---|
| FFHQ thumbnails 128×128 | SBI real faces | https://github.com/NVlabs/ffhq-dataset |
| FaceForensics++ c23 | Wavelet-CLIP training | Request form (see FF++ repo) |
| Celeb-DF v2 | Validation | https://github.com/yuezunli/celeb-deepfakeforensics |
| DFDC subset | Wavelet-CLIP training | Kaggle: deepfake-detection-challenge |
| WildDeepfake | Final eval only (never train) | https://github.com/deepfakeinthewild/deepfake-in-the-wild |

## File Structure

```
deepshield/
├── app.py                          # Streamlit inference app
├── configs/                        # Training YAML configs
├── data/preprocess/                # Face extraction + dataset classes
├── models/                         # SBIBranch, WaveletCLIP, Ensemble
├── train/                          # Training scripts
├── inference/                      # Pipeline, GradCAM, Track2
├── evaluate/                       # Eval script + metrics
├── notebooks/train_wavelet_clip_kaggle.ipynb
└── checkpoints/                    # Place trained .pth files here
```

## Target Metrics

| Metric | Target |
|---|---|
| Cross-dataset AUC | > 0.85 |
| WildDeepfake AUC | > 0.80 |
| FNR @ 0.5 | < 0.15 |
| Inference latency | < 200 ms |
