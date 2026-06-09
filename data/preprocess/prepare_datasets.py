"""
PyTorch Dataset classes for SBI training and Wavelet-CLIP training.

SBIDataset:
  - real faces from FFHQ thumbnails (128×128 PNGs, already face-crops)
  - fakes generated on-the-fly via SBI blending
  - validation faces from pre-extracted Celeb-DF v2 crops

WaveletCLIPDataset:
  - reads pre-extracted face crops from directory structure:
      faces/train/real/  faces/train/fake/
      faces/val/real/    faces/val/fake/
  - returns (img_imagenet, label) — model normalizes for CLIP internally
"""
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .sbi_generator import generate_sbi_pair

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


# ── Helper ────────────────────────────────────────────────────────────────────
def _load_rgb(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _split_files(files: list, val_fraction: float = 0.1) -> tuple:
    """Last val_fraction of sorted filenames → val, rest → train."""
    files = sorted(files)
    split = max(1, int(len(files) * (1 - val_fraction)))
    return files[:split], files[split:]


# ── SBI Dataset ───────────────────────────────────────────────────────────────
class SBIDataset(Dataset):
    """
    Training dataset for SBI branch.
    - ffhq_dir: directory of FFHQ 128×128 PNG thumbnails
    - is_train: if True, use SBI augmentation to generate fakes on-the-fly
    - val_faces_dir: if provided (eval mode), load pre-extracted real/fake crops
    """
    def __init__(
        self,
        ffhq_dir: Optional[str] = None,
        val_faces_dir: Optional[str] = None,
        is_train: bool = True,
        img_size: int = 224,
        sbi_prob: float = 0.5,
        jpeg_quality_min: int = 40,
        jpeg_quality_max: int = 100,
        val_split: float = 0.1,
    ):
        self.is_train = is_train
        self.img_size = img_size
        self.sbi_prob = sbi_prob
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max
        self.transform = get_train_transforms(img_size) if is_train else get_val_transforms(img_size)

        if is_train:
            if ffhq_dir is None:
                raise ValueError("ffhq_dir required for training")
            all_files = sorted(Path(ffhq_dir).rglob('*.png')) + \
                        sorted(Path(ffhq_dir).rglob('*.jpg'))
            if not all_files:
                raise FileNotFoundError(f"No images found in {ffhq_dir}")
            train_files, _ = _split_files(all_files, val_split)
            self.real_files = train_files
        else:
            # Validation: load pre-extracted Celeb-DF faces from real/ and fake/ subdirs
            if val_faces_dir is None:
                raise ValueError("val_faces_dir required for validation")
            faces_path = Path(val_faces_dir)
            self.samples = []  # (path, label)
            for label, subdir in [(0, 'real'), (1, 'fake')]:
                d = faces_path / subdir
                if d.exists():
                    for f in sorted(d.rglob('*.jpg')) + sorted(d.rglob('*.png')):
                        self.samples.append((f, label))
            if not self.samples:
                raise FileNotFoundError(f"No val samples found in {val_faces_dir}")

    def __len__(self) -> int:
        if self.is_train:
            return len(self.real_files)
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.is_train:
            face1 = _load_rgb(self.real_files[idx])
            if face1 is None:
                face1 = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            if random.random() < self.sbi_prob:
                # Generate fake via SBI
                donor_idx = random.randint(0, len(self.real_files) - 1)
                face2 = _load_rgb(self.real_files[donor_idx])
                if face2 is None:
                    face2 = face1.copy()
                _, fake = generate_sbi_pair(face1, face2, self.jpeg_quality_min, self.jpeg_quality_max)
                img = cv2.resize(fake, (self.img_size, self.img_size))
                label = 1
            else:
                img = cv2.resize(face1, (self.img_size, self.img_size))
                label = 0

            return self.transform(img), torch.tensor(label, dtype=torch.float32)

        else:
            path, label = self.samples[idx]
            img = _load_rgb(path)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.img_size, self.img_size))
            return self.transform(img), torch.tensor(label, dtype=torch.float32)


# ── WaveletCLIP Dataset ───────────────────────────────────────────────────────
class WaveletCLIPDataset(Dataset):
    """
    Reads pre-extracted face crops from:
        base_dir/train/real/
        base_dir/train/fake/
        base_dir/val/real/
        base_dir/val/fake/

    Returns (img_imagenet, label).
    WaveletCLIP handles CLIP re-normalization internally.
    """
    def __init__(
        self,
        base_dir: str,
        split: str = 'train',
        img_size: int = 224,
    ):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"
        self.img_size = img_size
        self.transform = get_train_transforms(img_size) if split == 'train' else get_val_transforms(img_size)

        base = Path(base_dir) / split
        self.samples = []
        for label, subdir in [(0, 'real'), (1, 'fake')]:
            d = base / subdir
            if not d.exists():
                print(f"[WARN] Directory not found: {d}")
                continue
            files = sorted(d.rglob('*.jpg')) + sorted(d.rglob('*.png'))
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise FileNotFoundError(
                f"No samples found in {base}.\n"
                f"Expected structure:\n"
                f"  {base}/real/*.jpg\n"
                f"  {base}/fake/*.jpg\n"
                f"Run extract_faces.py first."
            )
        random.shuffle(self.samples)
        print(f"[WaveletCLIPDataset] split={split}  samples={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = _load_rgb(path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.resize(img, (self.img_size, self.img_size))
        return self.transform(img), torch.tensor(label, dtype=torch.float32)
