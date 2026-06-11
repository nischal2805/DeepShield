"""
Post-training calibration for DeepShield models.

Two outputs, both written to a JSON the pipeline can load:
  1. temperature T  — divides the logit before sigmoid so the score is a
     real probability (fixes the "razor-thin margin / overconfident" problem).
  2. threshold      — decision cut chosen for HIGH RECALL on fakes, because for
     NCII a missed fake (failing a victim) is worse than a false alarm.

Run AFTER training, on a labeled validation set of face crops:
    python -m evaluate.calibrate --model sbi \
        --ckpt D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth \
        --faces-dir D:/deepshield_data/val_faces \
        --target-recall 0.95 \
        --out D:/deepshield_data/checkpoints/sbi/calibration.json

faces-dir layout:  <dir>/real/*.jpg   <dir>/fake/*.jpg
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ── Core calibration math (model-agnostic; operate on logits + labels) ──────────
def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lr: float = 0.01, iters: int = 500) -> float:
    """
    Fit scalar temperature T>0 minimizing binary NLL of sigmoid(logit / T).
    Plain numpy gradient descent on log T (keeps T positive). No torch needed.
    """
    logits = logits.astype(np.float64).ravel()
    labels = labels.astype(np.float64).ravel()
    log_t = 0.0  # T = exp(log_t), start at T=1
    for _ in range(iters):
        T = np.exp(log_t)
        z = logits / T
        p = 1.0 / (1.0 + np.exp(-z))
        # dNLL/dz = (p - y);  dz/dlog_t = -logits / T  →  chain rule
        grad = np.mean((p - labels) * (-logits / T))
        log_t -= lr * grad
    return float(np.exp(log_t))


def select_threshold(scores: np.ndarray, labels: np.ndarray,
                     target_recall: float = 0.95) -> dict:
    """
    Pick the highest probability threshold that still catches >= target_recall
    of the fakes (label==1). Highest such threshold = fewest false positives
    while honoring the recall floor. Returns chosen threshold + metrics there.
    """
    scores = scores.ravel()
    labels = labels.ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0:
        return {'threshold': 0.5, 'note': 'no positive (fake) samples'}

    # candidate thresholds = sorted unique fake scores
    best_thr = 0.0
    for thr in np.unique(np.concatenate([pos, [0.0, 1.0]])):
        recall = (pos >= thr).mean()
        if recall >= target_recall:
            best_thr = max(best_thr, thr)
    thr = float(best_thr)
    tp = float((pos >= thr).mean())
    fp = float((neg >= thr).mean()) if len(neg) else 0.0
    return {
        'threshold':   thr,
        'recall_fake': round(tp, 4),
        'fpr_real':    round(fp, 4),
        'target_recall': target_recall,
    }


def calibrate(logits: np.ndarray, labels: np.ndarray,
              target_recall: float = 0.95) -> dict:
    """Full calibration: temperature + threshold on the temperature-scaled scores."""
    T = fit_temperature(logits, labels)
    scaled = 1.0 / (1.0 + np.exp(-(logits.ravel() / T)))
    thr_info = select_threshold(scaled, labels, target_recall)
    auc = _auc(labels.ravel(), scaled)
    return {'temperature': round(T, 4), 'val_auc': round(auc, 4), **thr_info}


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC, no sklearn dependency."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    sum_pos = ranks[labels == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ── CLI: compute logits from a checkpoint over a labeled faces dir ──────────────
def _collect_logits(model_name: str, ckpt: str, faces_dir: str, img_size: int = 224):
    import cv2
    import torch
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'sbi':
        from models.sbi_branch import SBIBranch
        model = SBIBranch(pretrained=False).to(device)
    elif model_name == 'wclip':
        from models.wavelet_clip import WaveletCLIP
        model = WaveletCLIP(pretrained=False).to(device)
    else:
        raise ValueError("model must be 'sbi' or 'wclip'")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state['model_state'])
    model.eval()

    logits, labels = [], []
    base = Path(faces_dir)
    for label, sub in [(0, 'real'), (1, 'fake')]:
        d = base / sub
        files = sorted(d.rglob('*.jpg')) + sorted(d.rglob('*.png')) if d.exists() else []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                z = model(t)
            logits.append(float(z.flatten()[0]))
            labels.append(label)
    return np.array(logits), np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['sbi', 'wclip'])
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--faces-dir', required=True, help='dir with real/ and fake/ subdirs')
    ap.add_argument('--target-recall', type=float, default=0.95)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f"Scoring {args.model} on {args.faces_dir} ...")
    logits, labels = _collect_logits(args.model, args.ckpt, args.faces_dir, args.img_size)
    print(f"  {len(labels)} samples ({(labels==1).sum()} fake / {(labels==0).sum()} real)")
    result = calibrate(logits, labels, args.target_recall)
    print("\nCalibration:")
    print(json.dumps(result, indent=2))

    out = args.out or str(Path(args.ckpt).with_name('calibration.json'))
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out}")
    print("Pipeline: divide logit by `temperature` before sigmoid, "
          "compare to `threshold`.")


if __name__ == '__main__':
    main()
