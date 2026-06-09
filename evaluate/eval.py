"""
Evaluate the DeepShield ensemble on WildDeepfake (final held-out test set).
WildDeepfake is NEVER used during training — test-only benchmark.

Usage:
    python evaluate/eval.py \\
        --sbi-ckpt      D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth \\
        --wclip-ckpt    D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth \\
        --weights       checkpoints/ensemble_weights.json \\
        --test-dir      D:/deepshield_data/wilddeepfake/faces \\
        --output        eval_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess.prepare_datasets import WaveletCLIPDataset
from evaluate.metrics import compute_accuracy, compute_auc, compute_fnr, compute_fpr
from models.ensemble import DeepShieldEnsemble
from models.sbi_branch import SBIBranch
from models.wavelet_clip import WaveletCLIP


@torch.no_grad()
def run_evaluation(model, loader, device) -> tuple:
    model.eval()
    all_scores, all_labels = [], []
    latencies = []

    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device, non_blocking=True)
        t0 = time.perf_counter()
        scores, _, _ = model(imgs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) / len(imgs) * 1000)  # ms/image

        all_scores.append(scores.cpu().numpy())
        all_labels.append(labels.numpy())

    scores = np.concatenate(all_scores).flatten()
    labels = np.concatenate(all_labels).flatten()
    return scores, labels, np.mean(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sbi-ckpt',   required=True, type=Path)
    parser.add_argument('--wclip-ckpt', required=True, type=Path)
    parser.add_argument('--weights',    default='checkpoints/ensemble_weights.json', type=Path)
    parser.add_argument('--test-dir',   required=True, type=Path,
                        help="Directory with test/real/ and test/fake/ (or val/real and val/fake)")
    parser.add_argument('--output',     default='eval_results.json', type=Path)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device',     default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Detect split dir name (test or val)
    split = 'val'
    if (args.test_dir / 'test').exists():
        split = 'test'
    elif (args.test_dir / 'val').exists():
        split = 'val'
    else:
        # Assume test_dir IS the split root
        split = args.test_dir.name
        args.test_dir = args.test_dir.parent

    print(f"Using split: {split} from {args.test_dir}")

    # ── Load models ──
    sbi_model = SBIBranch(pretrained=False).to(device)
    sbi_model.load_state_dict(torch.load(args.sbi_ckpt, map_location=device)['model_state'])

    wclip_model = WaveletCLIP().to(device)
    wclip_model.load_state_dict(torch.load(args.wclip_ckpt, map_location=device)['model_state'])

    ensemble = DeepShieldEnsemble(
        sbi_model, wclip_model,
        weights_path=str(args.weights) if args.weights.exists() else None,
    ).to(device)

    # ── Dataset ──
    test_ds = WaveletCLIPDataset(str(args.test_dir), split=split)
    loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, persistent_workers=False)
    print(f"Test samples: {len(test_ds)}")

    # ── Evaluate ──
    scores, labels, avg_latency_ms = run_evaluation(ensemble, loader, device)

    metrics = {
        'auc':                  compute_auc(labels, scores),
        'accuracy':             compute_accuracy(labels, scores),
        'fnr@0.5':              compute_fnr(labels, scores),
        'fpr@0.5':              compute_fpr(labels, scores),
        'avg_latency_ms':       round(avg_latency_ms, 2),
        'n_samples':            len(labels),
        'n_positive':           int(labels.sum()),
    }

    print("\n── Evaluation Results ──────────────────")
    print(f"  AUC:            {metrics['auc']:.4f}  (target > 0.80)")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  FNR@0.5:        {metrics['fnr@0.5']:.4f}  (target < 0.15)")
    print(f"  FPR@0.5:        {metrics['fpr@0.5']:.4f}")
    print(f"  Latency:        {metrics['avg_latency_ms']:.1f} ms/image  (target < 200ms)")
    print("─────────────────────────────────────────")

    # Targets
    passed = {
        'auc_pass':     metrics['auc'] > 0.80,
        'fnr_pass':     metrics['fnr@0.5'] < 0.15,
        'latency_pass': metrics['avg_latency_ms'] < 200,
    }
    for name, ok in passed.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")

    result = {**metrics, **passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
