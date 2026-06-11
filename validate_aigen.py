"""
Validate AIGenDetector on the labeled HEIC Conversions folder.
Computes AUC(Stylegan vs Real) and picks a threshold.

CRITICAL import order (must match batch_test.py / Windows 0xC0000005 rule):
"""
import os
import sys

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2                                                    # noqa
import torch                                                  # noqa
from inference.pipeline import DeepShieldPipeline, _TRANSFORM  # noqa

# Only import models.* AFTER inference.pipeline to avoid segfault
from models.aigen_branch import AIGenDetector               # noqa

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CKPT_NEW   = 'D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth'
FC_PATH    = 'D:/deepshield_data/checkpoints/aigen/fc_weights.pth'
VAL_DIR    = r'C:\Users\nisch\OneDrive\Pictures\HEIC Conversions'
EXTS       = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

# ── Load pipeline (once) — to get the wclip model and its CLIP handle ─────────
print("Loading pipeline …")
pipe = DeepShieldPipeline(wclip_ckpt=CKPT_NEW)

wclip_model  = pipe.ensemble.wclip                       # WaveletCLIP
clip_handle  = wclip_model.clip_projection.clip          # transformers CLIPModel

# ── Try both normalization variants, report both, keep better one ──────────────
results_raw   = {}   # filename -> (label, prob_raw)
results_l2    = {}   # filename -> (label, prob_l2)

detector_raw = AIGenDetector(
    clip_model=clip_handle,
    to_clip_norm_fn=wclip_model._to_clip_norm,
    device=pipe.device,
    fc_path=FC_PATH,
    threshold=0.5,
    use_l2_norm=False,
)
detector_l2 = AIGenDetector(
    clip_model=clip_handle,
    to_clip_norm_fn=wclip_model._to_clip_norm,
    device=pipe.device,
    fc_path=FC_PATH,
    threshold=0.5,
    use_l2_norm=True,
)

print("\nRunning validation …\n")

images = []
for f in sorted(os.listdir(VAL_DIR)):
    if f.lower().endswith(EXTS):
        images.append(f)

rows = []
for fname in images:
    name_upper = fname.upper()
    # Determine ground-truth label
    if name_upper.startswith('REAL') or fname == 'DP.jpg':
        label = 'REAL'
    elif name_upper.startswith('STYLEGAN'):
        label = 'AIGEN'
    elif name_upper.startswith('FACE SWAP'):
        label = 'FACESWAP'   # ignore for AUC
    else:
        label = 'UNKNOWN'

    fpath = os.path.join(VAL_DIR, fname)
    img_bgr = cv2.imread(fpath)
    if img_bgr is None:
        print(f"  SKIP (cannot read): {fname}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Face-crop (same as flask_app.py pipeline)
    faces = pipe.detector.detect(img_rgb)
    if not faces:
        print(f"  NOFACE: {fname}")
        rows.append((fname, label, None, None))
        continue

    crop = pipe.detector.align_and_crop(img_rgb, faces[0], size=224)
    t    = _TRANSFORM(crop).unsqueeze(0).to(pipe.device)

    with torch.no_grad():
        p_raw = detector_raw.predict(t)
        p_l2  = detector_l2.predict(t)

    rows.append((fname, label, p_raw, p_l2))

# ── Print table ───────────────────────────────────────────────────────────────
print(f"{'Filename':<36} {'Label':<10} {'P_raw':>8} {'P_l2':>8}")
print('-' * 66)
for fname, label, p_raw, p_l2 in rows:
    pr_str = f"{p_raw:.4f}" if p_raw is not None else "  NOFACE"
    pl_str = f"{p_l2:.4f}"  if p_l2  is not None else "  NOFACE"
    print(f"{fname[:35]:<36} {label:<10} {pr_str:>8} {pl_str:>8}")

# ── Compute AUC for Stylegan vs Real (ignore FACESWAP / UNKNOWN / NOFACE) ────
from sklearn.metrics import roc_auc_score

def compute_auc_and_threshold(rows, score_idx):
    y_true, y_score = [], []
    for _, label, p_raw, p_l2 in rows:
        if label not in ('REAL', 'AIGEN'):
            continue
        p = p_raw if score_idx == 0 else p_l2
        if p is None:
            continue
        y_true.append(1 if label == 'AIGEN' else 0)
        y_score.append(p)
    if len(set(y_true)) < 2:
        return None, None, None
    auc = roc_auc_score(y_true, y_score)

    # pick threshold: maximize F1 on this small set
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.0, 1.0, 201):
        preds = [1 if s >= t else 0 for s in y_score]
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    return auc, best_t, best_f1

auc_raw, thr_raw, f1_raw = compute_auc_and_threshold(rows, 0)
auc_l2,  thr_l2,  f1_l2  = compute_auc_and_threshold(rows, 1)

print()
print("=" * 66)
print(f"RAW feats   AUC={auc_raw:.4f}  best_thresh={thr_raw:.3f}  F1={f1_raw:.4f}")
print(f"L2-norm     AUC={auc_l2:.4f}  best_thresh={thr_l2:.3f}  F1={f1_l2:.4f}")
print("=" * 66)

# Decide which variant to use
if auc_l2 >= auc_raw:
    chosen_variant   = 'l2_norm'
    chosen_threshold = round(float(thr_l2), 3)
    chosen_auc       = auc_l2
else:
    chosen_variant   = 'raw'
    chosen_threshold = round(float(thr_raw), 3)
    chosen_auc       = auc_raw

print(f"\nCHOSEN: {chosen_variant}  threshold={chosen_threshold}  AUC={chosen_auc:.4f}")
print(f"\nFinal threshold to hard-code in flask_app.py: {chosen_threshold}")
print(f"use_l2_norm = {'True' if chosen_variant == 'l2_norm' else 'False'}")
