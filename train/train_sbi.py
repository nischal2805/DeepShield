"""
Train the SBI branch (EfficientNet-B4) locally on Windows + RTX 4060.

Usage:
    python train/train_sbi.py --config configs/sbi_config.yaml

Expected runtime: ~6-8 hours for 30 epochs on RTX 4060.

OOM mitigation:
  - Reduce batch_size to 8 and set grad_accum_steps to 4 in config
  - If still OOM, set img_size to 192
"""
import argparse
import json
import multiprocessing
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.preprocess.prepare_datasets import SBIDataset
from evaluate.metrics import compute_auc, compute_fnr
from models.sbi_branch import SBIBranch


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: SBIBranch, cfg: dict, frozen: bool) -> torch.optim.Optimizer:
    if frozen:
        params = model.get_head_params()
        return torch.optim.AdamW(params, lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    else:
        return torch.optim.AdamW([
            {'params': model.get_backbone_params(), 'lr': cfg['training']['lr_backbone_unfrozen']},
            {'params': model.get_head_params(),     'lr': cfg['training']['lr']},
        ], weight_decay=cfg['training']['weight_decay'])


def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, grad_accum_steps, grad_clip
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (imgs, labels) in enumerate(tqdm(loader, desc="  train", leave=False)):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels) / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  val  ", leave=False):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)
        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        total_loss  += loss.item()
        all_logits.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())

    scores = torch.cat(all_logits).numpy().flatten()
    labels = torch.cat(all_labels).numpy().flatten()

    return {
        'loss': total_loss / len(loader),
        'auc':  compute_auc(labels, scores),
        'fnr':  compute_fnr(labels, scores, threshold=0.5),
    }


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, save_dir: Path, tag: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'metrics':     metrics,
    }
    torch.save(ckpt, save_dir / f'sbi_{tag}.pth')


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
    train_ds = SBIDataset(
        ffhq_dir=cfg['data']['ffhq_dir'],
        is_train=True,
        img_size=cfg['data']['img_size'],
        sbi_prob=cfg['data']['sbi_prob'],
        jpeg_quality_min=cfg['data']['jpeg_quality_min'],
        jpeg_quality_max=cfg['data']['jpeg_quality_max'],
        val_split=cfg['data']['val_split'],
    )
    val_mode = cfg['data'].get('val_mode', 'ffhq_self')
    val_ds = SBIDataset(
        ffhq_dir=cfg['data']['ffhq_dir'] if val_mode == 'ffhq_self' else None,
        val_faces_dir=cfg['data'].get('celeb_df_faces_dir') if val_mode == 'celeb_df' else None,
        is_train=False,
        val_mode=val_mode,
        img_size=cfg['data']['img_size'],
        val_split=cfg['data']['val_split'],
        jpeg_quality_min=cfg['data']['jpeg_quality_min'],
        jpeg_quality_max=cfg['data']['jpeg_quality_max'],
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # Windows: persistent_workers must be False
    loader_kwargs = dict(
        num_workers=cfg['training']['num_workers'],
        pin_memory=True,
        persistent_workers=False,  # Windows requirement
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg['training']['batch_size'],
        shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg['training']['batch_size'] * 2,
        shuffle=False, **loader_kwargs,
    )

    # ── Model ──
    model = SBIBranch(pretrained=cfg['model']['pretrained']).to(device)
    model.freeze_backbone()
    print("Backbone frozen for first", cfg['training']['freeze_epochs'], "epochs")

    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg['training']['mixed_precision'])
    optimizer = build_optimizer(model, cfg, frozen=True)

    total_epochs = cfg['training']['epochs']
    freeze_epochs = cfg['training']['freeze_epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    save_dir = Path(cfg['checkpoints']['save_dir'])
    best_auc = 0.0
    history  = []

    # ── wandb (optional) ──
    use_wandb = cfg['logging']['use_wandb']
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg['logging']['project_name'],
            name=cfg['logging']['run_name'],
            config=cfg,
        )

    print(f"\nStarting training for {total_epochs} epochs...")
    for epoch in range(1, total_epochs + 1):
        # Unfreeze backbone at epoch freeze_epochs+1, rebuild optimizer
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer = build_optimizer(model, cfg, frozen=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs - freeze_epochs
            )
            print(f"\nEpoch {epoch}: backbone unfrozen, lr_backbone={cfg['training']['lr_backbone_unfrozen']}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, cfg['training']['grad_accum_steps'], cfg['training']['grad_clip'],
        )
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch:3d}/{total_epochs} | "
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

        # Save best
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            save_checkpoint(model, optimizer, epoch, val_metrics, save_dir, 'best_auc')
            print(f"  ★ New best AUC: {best_auc:.4f}")

        # Periodic checkpoint
        if epoch % cfg['checkpoints']['save_every_n_epochs'] == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, save_dir, f'epoch{epoch:03d}')

    # Save history
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'sbi_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")
    print(f"Checkpoint: {save_dir / 'sbi_best_auc.pth'}")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    # Windows multiprocessing fix — must be here
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/sbi_config.yaml')
    args = parser.parse_args()

    main(args.config)
