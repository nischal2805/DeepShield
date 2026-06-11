"""
PyTorch Dataset classes for SBI training and Wavelet-CLIP training.

SBIDataset val_mode options:
  'ffhq_self'  — held-out FFHQ + SBI-generated fakes  (no Celeb-DF required)
  'celeb_df'   — pre-extracted Celeb-DF v2 real/fake crops

WaveletCLIPDataset:
  reads pre-extracted face crops from:
    faces/train/real/  faces/train/fake/
    faces/val/real/    faces/val/fake/
"""
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .sbi_generator import generate_self_blend

# ── ImageNet normalization ────────────────────────────────────────────────────
_IN_MEAN = [0.485, 0.456, 0.406]
_IN_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IN_MEAN, _IN_STD),
    ])


def get_val_transforms(img_size: int = 224) -> Callable:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IN_MEAN, _IN_STD),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _split_files(files: list, val_fraction: float = 0.1) -> tuple:
    files = sorted(files)
    split = max(1, int(len(files) * (1 - val_fraction)))
    return files[:split], files[split:]


# ── SBI Dataset ───────────────────────────────────────────────────────────────
class SBIDataset(Dataset):
    """
    val_mode='ffhq_self': validation uses held-out FFHQ + SBI fakes (no Celeb-DF).
    val_mode='celeb_df':  validation uses pre-extracted real/fake crops from val_faces_dir.
    """

    def __init__(
        self,
        ffhq_dir: Optional[str] = None,
        val_faces_dir: Optional[str] = None,
        is_train: bool = True,
        val_mode: str = 'ffhq_self',
        img_size: int = 224,
        sbi_prob: float = 0.5,
        jpeg_quality_min: int = 40,
        jpeg_quality_max: int = 100,
        val_split: float = 0.1,
        preload: bool = False,
    ):
        self.is_train          = is_train
        self.val_mode          = val_mode
        self.img_size          = img_size
        self.sbi_prob          = sbi_prob
        self.jpeg_quality_min  = jpeg_quality_min
        self.jpeg_quality_max  = jpeg_quality_max
        self.transform         = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)
        self.real_files: list  = []
        self.samples:    list  = []
        self._cache: Optional[list] = None

        # Load FFHQ file list if needed
        if ffhq_dir is not None:
            all_files = sorted(Path(ffhq_dir).rglob('*.png')) + \
                        sorted(Path(ffhq_dir).rglob('*.jpg'))
            if not all_files:
                raise FileNotFoundError(f"No images found in {ffhq_dir}")
            train_files, val_files = _split_files(all_files, val_split)
        else:
            train_files, val_files = [], []

        if is_train:
            if not train_files:
                raise ValueError("ffhq_dir required for training")
            self.real_files = train_files
            print(f"[SBIDataset] train={len(self.real_files)} real images")

        else:
            if val_mode == 'ffhq_self':
                if not val_files:
                    raise ValueError("ffhq_dir required for val_mode='ffhq_self'")
                self.real_files = val_files
                print(f"[SBIDataset] val=ffhq_self  {len(self.real_files)} held-out images")

            else:  # celeb_df
                if val_faces_dir is None:
                    raise ValueError("val_faces_dir required for val_mode='celeb_df'")
                faces_path = Path(val_faces_dir)
                for label, subdir in [(0, 'real'), (1, 'fake')]:
                    d = faces_path / subdir
                    if d.exists():
                        for f in sorted(d.rglob('*.jpg')) + sorted(d.rglob('*.png')):
                            self.samples.append((f, label))
                if not self.samples:
                    raise FileNotFoundError(f"No val samples in {val_faces_dir}")
                print(f"[SBIDataset] val=celeb_df  {len(self.samples)} samples")

        # Preload images into RAM (eliminates disk I/O during training)
        if preload and self.real_files:
            print(f"[SBIDataset] preloading {len(self.real_files)} images into RAM...")
            self._cache = [_load_rgb(p) for p in self.real_files]
            loaded = sum(1 for x in self._cache if x is not None)
            print(f"[SBIDataset] preload done  {loaded}/{len(self.real_files)} loaded")

    def __len__(self) -> int:
        if self.is_train or self.val_mode == 'ffhq_self':
            return len(self.real_files)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.is_train or self.val_mode == 'ffhq_self':
            return self._sbi_item(idx, self.is_train)

        # celeb_df val
        path, label = self.samples[idx]
        _img = _load_rgb(path)
        img  = _img if _img is not None else np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

    def _get_img(self, idx: int) -> np.ndarray:
        if self._cache is not None:
            img = self._cache[idx]
        else:
            img = _load_rgb(self.real_files[idx])
        return img if img is not None else np.zeros((self.img_size, self.img_size, 3), np.uint8)

    def _sbi_item(self, idx: int, use_sbi_prob: bool):
        """Generate real or SBI-fake sample. use_sbi_prob=False → 50/50 for val."""
        face1 = self._get_img(idx)

        prob = self.sbi_prob if use_sbi_prob else 0.5
        if random.random() < prob:
            # True SBI: self-blend face1 with a transformed copy of itself.
            _, fake = generate_self_blend(
                face1,
                jpeg_quality_min=self.jpeg_quality_min,
                jpeg_quality_max=self.jpeg_quality_max,
            )
            img   = cv2.resize(fake, (self.img_size, self.img_size))
            label = 1
        else:
            img   = cv2.resize(face1, (self.img_size, self.img_size))
            label = 0
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── WaveletCLIP Dataset ───────────────────────────────────────────────────────
class WaveletCLIPDataset(Dataset):
    """
    Reads pre-extracted face crops from:
        base_dir/train/real/  base_dir/train/fake/
        base_dir/val/real/    base_dir/val/fake/
    Returns (img_imagenet, label). WaveletCLIP handles CLIP re-norm internally.
    """

    def __init__(self, base_dir: str, split: str = 'train', img_size: int = 224):
        assert split in ('train', 'val')
        self.img_size  = img_size
        self.transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)
        self.samples: list = []

        base = Path(base_dir) / split
        for label, subdir in [(0, 'real'), (1, 'fake')]:
            d = base / subdir
            if not d.exists():
                print(f"[WARN] Not found: {d}")
                continue
            for f in sorted(d.rglob('*.jpg')) + sorted(d.rglob('*.png')):
                self.samples.append((f, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No samples in {base}. Run extract_faces.py first.\n"
                f"Expected: {base}/real/*.jpg  and  {base}/fake/*.jpg"
            )
        random.shuffle(self.samples)
        print(f"[WaveletCLIPDataset] split={split}  samples={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        _img = _load_rgb(path)
        img  = _img if _img is not None else np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
