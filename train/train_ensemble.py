"""
Fit ensemble fusion weights (logistic regression on val-set scores).
Run after both SBI and Wavelet-CLIP training is complete.

Usage:
    python train/train_ensemble.py \\
        --sbi-ckpt     D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth \\
        --wclip-ckpt   D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth \\
        --celeb-faces  D:/deepshield_data/celeb_df_v2/faces \\
        --output       checkpoints/ensemble_weights.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess.prepare_datasets import SBIDataset
from models.sbi_branch import SBIBranch
from models.wavelet_clip import WaveletCLIP


@torch.no_grad()
def collect_scores(sbi_model, wclip_model, loader, device):
    sbi_model.eval()
    wclip_model.eval()
    all_sbi, all_wclip, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc="Collecting scores"):
        imgs = imgs.to(device, non_blocking=True)
        s_sbi   = torch.sigmoid(sbi_model(imgs)).cpu().numpy().flatten()
        s_wclip = torch.sigmoid(wclip_model(imgs)).cpu().numpy().flatten()
        all_sbi.extend(s_sbi.tolist())
        all_wclip.extend(s_wclip.tolist())
        all_labels.extend(labels.numpy().flatten().tolist())

    return np.array(all_sbi), np.array(all_wclip), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sbi-ckpt',    required=True,  type=Path)
    parser.add_argument('--wclip-ckpt',  required=True,  type=Path)
    parser.add_argument('--celeb-faces', required=True,  type=Path,
                        help="Pre-extracted Celeb-DF val faces dir (must have real/ and fake/ subdirs)")
    parser.add_argument('--output',      default='checkpoints/ensemble_weights.json', type=Path)
    parser.add_argument('--batch-size',  type=int, default=32)
    parser.add_argument('--device',      default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Load models ──
    print("Loading SBI model...")
    sbi_model = SBIBranch(pretrained=False).to(device)
    ckpt = torch.load(args.sbi_ckpt, map_location=device)
    sbi_model.load_state_dict(ckpt['model_state'])

    print("Loading Wavelet-CLIP model...")
    wclip_model = WaveletCLIP().to(device)
    ckpt = torch.load(args.wclip_ckpt, map_location=device)
    wclip_model.load_state_dict(ckpt['model_state'])

    # ── Val dataset ──
    val_ds = SBIDataset(val_faces_dir=str(args.celeb_faces), is_train=False)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, persistent_workers=False)

    # ── Collect scores ──
    scores_sbi, scores_wclip, labels = collect_scores(sbi_model, wclip_model, loader, device)
    print(f"Collected {len(labels)} samples. Positive rate: {labels.mean():.2%}")

    X = np.stack([scores_sbi, scores_wclip], axis=1)

    # ── Fit logistic regression ──
    lr = LogisticRegression(C=1.0, max_iter=1000, fit_intercept=False)
    lr.fit(X, labels)
    coef = lr.coef_[0]

    # Normalize to sum to 1 (keep as weights for weighted sum)
    w = np.abs(coef)
    w = w / w.sum()
    w_sbi, w_wclip = float(w[0]), float(w[1])

    final_scores = w_sbi * scores_sbi + w_wclip * scores_wclip
    auc = roc_auc_score(labels, final_scores)
    print(f"Ensemble weights — SBI: {w_sbi:.4f}  WCLIP: {w_wclip:.4f}")
    print(f"Ensemble val AUC: {auc:.4f}")

    result = {'w_sbi': w_sbi, 'w_wclip': w_wclip, 'val_auc': auc}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
