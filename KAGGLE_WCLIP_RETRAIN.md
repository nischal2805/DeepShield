# Kaggle Wavelet-CLIP Retrain — Required Notebook Changes

Notebook: `notebook4dab03346e.ipynb` (file persistence ON, repo + faces already on disk).

## TL;DR — why the 0.9995 AUC is fake
The previous run hit **val AUC 0.9995** because of **data leakage**, not skill:
- Cell 12 auto-split moves **random frames** to val. FF++ extracts ~17 frames per video
  (`{video_stem}_{frame:06d}.jpg`). The *same video* lands in BOTH train and val.
  Model memorizes the video → val AUC saturates at 0.999. Meaningless.
- CLIP backbone is **fully frozen** (3.7M / 431.3M trainable). The model underfits;
  only the tiny wavelet branch + fusion head learn. On real out-of-distribution
  deepfakes this generalizes poorly.

After the two fixes below, expect a **realistic val AUC ~0.85–0.95**. That lower number
is the *honest* one and will transfer far better to NCII deepfakes.

Three changes. All are **self-contained inline edits** — they do NOT require pushing the
repo or `git pull`. Apply in order.

---

## CHANGE 1 — Cell 12: split by VIDEO, not by frame (kills leakage)

In Cell 12, find the `else:` branch (the `Celeb-DF not found ... using FF++ train/val
split instead` block) and **replace the auto-split body** with a group-by-video split.

**Replace this:**
```python
    print(f'\nCeleb-DF not found (optional) — using FF++ train/val split instead')
    # Auto-split: move 10% of train faces to val
    import random
    random.seed(42)
    for label in ['real', 'fake']:
        train_dir = Path(PATHS['faces_out']) / 'train' / label
        val_dir   = Path(PATHS['faces_out']) / 'val' / label
        all_imgs = sorted(train_dir.glob('*.jpg'))
        n_val = max(1, len(all_imgs) // 10)
        val_imgs = random.sample(all_imgs, min(n_val, len(all_imgs)))
        for img in val_imgs:
            shutil.move(str(img), str(val_dir / img.name))
        print(f'  Moved {len(val_imgs)} {label} images to val/')
```

**With this (group by video id, hold out whole videos):**
```python
    print(f'\nCeleb-DF not found (optional) — using FF++ GROUP (by-video) split')
    import random
    random.seed(42)

    def video_id(fname: str) -> str:
        # extract_faces names crops "<videostem>_<frame:06d>.jpg".
        # FF++ stems are "000" (real) or "000_003" (fake). Strip the trailing
        # _<frame> only, so all frames of one video share an id.
        stem = fname.rsplit('.', 1)[0]
        return stem.rsplit('_', 1)[0]

    for label in ['real', 'fake']:
        train_dir = Path(PATHS['faces_out']) / 'train' / label
        val_dir   = Path(PATHS['faces_out']) / 'val' / label
        all_imgs = sorted(train_dir.glob('*.jpg'))

        # group frames by source video
        groups = {}
        for img in all_imgs:
            groups.setdefault(video_id(img.name), []).append(img)

        vids = sorted(groups.keys())
        random.shuffle(vids)
        n_val_vids = max(1, len(vids) // 10)         # 10% of VIDEOS
        val_vids = set(vids[:n_val_vids])

        moved = 0
        for vid in val_vids:
            for img in groups[vid]:
                shutil.move(str(img), str(val_dir / img.name))
                moved += 1
        print(f'  {label}: held out {len(val_vids)}/{len(vids)} videos '
              f'({moved} frames) to val/ — no video shared with train')
```

Why: every frame of a held-out video goes to val, so no video identity is shared. The
val AUC now measures generalization to **unseen videos**, which is what you actually need.

---

## CHANGE 2 — Cell 9: unfreeze last 3 CLIP blocks (fixes underfit)

In Cell 9 (`## Cell 9 — Model init`), **after** `model = WaveletCLIP().to(device)` and
**before** the trainable-params print, add the unfreeze loop.

**Replace this:**
```python
device = torch.device('cuda')
print('Loading CLIP (downloads ~1.7 GB on first run)...')
model = WaveletCLIP().to(device)
print('Model loaded.')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total')
```

**With this:**
```python
device = torch.device('cuda')
print('Loading CLIP (downloads ~1.7 GB on first run)...')
model = WaveletCLIP().to(device)
print('Model loaded.')

# ── Unfreeze last N CLIP vision blocks (was fully frozen => underfit) ──
N_UNFREEZE = 3
clip_layers = model.clip_projection.clip.vision_model.encoder.layers
clip_backbone_params = []
for blk in clip_layers[-N_UNFREEZE:]:
    for p in blk.parameters():
        p.requires_grad = True
        clip_backbone_params.append(p)
for p in model.clip_projection.clip.vision_model.post_layernorm.parameters():
    p.requires_grad = True
    clip_backbone_params.append(p)
print(f'Unfroze last {N_UNFREEZE} CLIP blocks: '
      f'{sum(p.numel() for p in clip_backbone_params)/1e6:.1f}M extra params')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total')
```

Expect trainable to jump from **3.7M to ~40M**. If T4 OOMs, drop `N_UNFREEZE` to 2 or
lower `batch_size` to 16 in Cell 7 (CFG).

> If the repo's `models/wavelet_clip.py` was pulled with the new helper, you can instead
> write `clip_backbone_params = model.unfreeze_clip_layers(3)` — but the inline loop above
> works regardless and needs no `git pull`.

---

## CHANGE 3 — Cell 10: add the CLIP params to the optimizer at a tiny LR

The new params from Change 2 must be in the optimizer or they won't train. Give them a
**very low LR** (1e-6) so the pretrained CLIP features adapt gently, not catastrophically.

**Replace Cell 10 entirely:**
```python
LR_CLIP_BACKBONE = 1e-6   # tiny LR for unfrozen CLIP blocks

optimizer = torch.optim.AdamW([
    {'params': model.wavelet_branch.parameters(),       'lr': CFG['lr_wavelet']},
    {'params': model.clip_projection.proj.parameters(), 'lr': CFG['lr_clip_proj']},
    {'params': model.fusion.parameters(),               'lr': CFG['lr_wavelet']},
    {'params': clip_backbone_params,                    'lr': LR_CLIP_BACKBONE},
], weight_decay=CFG['weight_decay'])

total_steps = len(train_loader) * CFG['epochs']
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[CFG['lr_wavelet'], CFG['lr_clip_proj'], CFG['lr_wavelet'], LR_CLIP_BACKBONE],
    total_steps=total_steps,
)
print(f'Optimizer ready ({len(optimizer.param_groups)} groups). Total steps: {total_steps}')
```

Note the `max_lr` list now has **4 entries** (one per param group) — must match or OneCycleLR
errors.

---

## Run order on Kaggle (persisted session)

1. **Skip re-download.** Faces already extracted on the persisted disk
   (`/kaggle/working/faces`). Do NOT re-run Cell 10/12 download+extract — it wastes the
   deadline. BUT: the old run already did the *leaky* split (frames split into val/).
   To re-split cleanly, first **merge val back into train**, then apply Change 1.
   Run this ONE-OFF cell before training:
   ```python
   import shutil
   from pathlib import Path
   for label in ['real', 'fake']:
       v = Path(PATHS['faces_out']) / 'val' / label
       t = Path(PATHS['faces_out']) / 'train' / label
       for img in list(v.glob('*.jpg')):
           shutil.move(str(img), str(t / img.name))
   print('Merged val back into train. Now re-run the group split.')
   ```
   Then run the **Change-1 group-split** code (just the split block) once.
2. Re-run Cell 7 (CFG), Cell 8 (datasets — will pick up new val), Cell 9 (Change 2),
   Cell 10 (Change 3), Cell 11, Cell 12 training loop.
3. Watch val AUC: it should climb to ~0.85–0.95 and **plateau** (not pin at 0.999).
   If it still hits 0.999 in epoch 1–3, the split didn't take — verify no shared
   `video_id` across train/val.
4. Download checkpoint from the last cell.

## After download — calibrate locally
Once `wavelet_clip_best_auc.pth` is on your machine:
```
python -m evaluate.calibrate --model wclip --ckpt D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth --faces-dir D:/deepshield_data/val_faces
```
This fits temperature scaling + a high-recall threshold and writes `calibration.json`.

## Optional: git pull instead of inline edits
If you prefer pulling the repo fixes (self-blend generator, `unfreeze_clip_layers` helper,
patched `train_wavelet_clip.py`), the local changes must first be committed+pushed to
`nischal2805/DeepShield` master. Then in Cell 2 change the `else` branch from
`print('Repo already cloned, skipping')` to:
```python
else:
    os.chdir('/kaggle/working/DeepShield')
    !git pull origin master
```
Even with a pull, you still apply **Change 1** (the notebook's split cell is not part of the
repo).
