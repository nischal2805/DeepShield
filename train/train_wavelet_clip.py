"""
Train Wavelet-CLIP branch locally (mirrors the Kaggle notebook).
Requires pre-extracted face crops (run extract_faces.py first).

Usage:
    python train/train_wavelet_clip.py --config configs/wavelet_clip_config.yaml

Note: Kaggle is recommended for this training (40 epochs × CLIP download).
      This script is for local training on machines with ≥12GB VRAM.
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess.prepare_datasets import WaveletCLIPDataset
from evaluate.metrics import compute_auc, compute_fnr
from models.wavelet_clip import WaveletCLIP


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt     = torch.exp(-bce)
        focal  = self.alpha * (1.0 - pt) ** self.gamma * bce
        return focal.float().mean()  # .float() prevents NaN with fp16


def train_one_epoch(model, loader, optimizer, scheduler, scaler, criterion, device, grad_accum, grad_clip):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * grad_accum

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    all_scores, all_labels = [], []
    total_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss    += loss.item()
        all_scores.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    scores = torch.cat(all_scores).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()
    return {
        'loss': total_loss / len(loader),
        'auc':  compute_auc(labels, scores),
        'fnr':  compute_fnr(labels, scores, threshold=0.5),
    }


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Datasets ──
    train_ds = WaveletCLIPDataset(cfg['data']['face_dir'], split='train', img_size=cfg['data']['img_size'])
    val_ds   = WaveletCLIPDataset(cfg['data']['face_dir'], split='val',   img_size=cfg['data']['img_size'])

    loader_kwargs = dict(num_workers=cfg['training']['num_workers'], pin_memory=True, persistent_workers=False)
    train_loader  = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader    = DataLoader(val_ds,   batch_size=cfg['training']['batch_size'] * 2, shuffle=False, **loader_kwargs)

    # ── Model ──
    print("Loading CLIP model (this may take a minute on first run)...")
    model = WaveletCLIP(clip_model_name=cfg['model']['clip_model']).to(device)

    criterion = FocalLoss(gamma=cfg['training']['focal_loss_gamma'], alpha=cfg['training']['focal_loss_alpha'])
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg['training']['mixed_precision'])

    param_groups = [
        {'params': model.wavelet_branch.parameters(),       'lr': cfg['training']['lr_wavelet']},
        {'params': model.clip_projection.proj.parameters(), 'lr': cfg['training']['lr_clip_proj']},
        {'params': model.fusion.parameters(),               'lr': cfg['training']['lr_wavelet']},
    ]
    # Partial CLIP unfreeze — lets the backbone adapt instead of underfitting.
    # 0 = fully frozen (old behavior). 2-4 recommended on Kaggle T4.
    n_unfreeze = cfg['training'].get('clip_unfreeze_blocks', 0)
    lr_clip_backbone = cfg['training'].get('lr_clip_backbone', 1.0e-6)
    if n_unfreeze > 0:
        clip_params = model.unfreeze_clip_layers(n_unfreeze)
        param_groups.append({'params': clip_params, 'lr': lr_clip_backbone})
        print(f"Unfroze last {n_unfreeze} CLIP blocks "
              f"({sum(p.numel() for p in clip_params)/1e6:.1f}M params) at lr={lr_clip_backbone}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg['training']['weight_decay'])

    total_steps = len(train_loader) * cfg['training']['epochs']
    # max_lr per group must match optimizer param_groups (4th appears if CLIP unfrozen)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] for g in param_groups],
        total_steps=total_steps,
    )

    save_dir = Path(cfg['checkpoints']['save_dir'])
    best_auc = 0.0
    history  = []

    use_wandb = cfg['logging']['use_wandb']
    if use_wandb:
        import wandb
        wandb.init(project=cfg['logging']['project_name'], name=cfg['logging']['run_name'], config=cfg)

    print(f"\nStarting training for {cfg['training']['epochs']} epochs...")
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_loss  = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            device, cfg['training']['grad_accum_steps'], cfg['training']['grad_clip'],
        )
        val_metrics = validate(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{cfg['training']['epochs']} | "
            f"train_loss={train_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} | "
            f"val_fnr={val_metrics['fnr']:.4f} | "
            f"lr={lr:.2e}"
        )

        record = {'epoch': epoch, 'train_loss': train_loss, 'lr': lr, **val_metrics}
        history.append(record)
        if use_wandb:
            import wandb
            wandb.log(record)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict(), 'metrics': val_metrics},
                save_dir / 'wavelet_clip_best_auc.pth',
            )
            print(f"  ★ New best AUC: {best_auc:.4f}")

        if epoch % cfg['checkpoints']['save_every_n_epochs'] == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {'epoch': epoch, 'model_state': model.state_dict()},
                save_dir / f'wavelet_clip_epoch{epoch:03d}.pth',
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'wavelet_clip_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best AUC: {best_auc:.4f}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/wavelet_clip_config.yaml')
    args = parser.parse_args()
    main(args.config)
