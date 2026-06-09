"""
Self-Blended Image (SBI) augmentation.
Generates fake faces on-the-fly by blending two real faces.
Never pre-saves blended images — called in dataset __getitem__ every epoch.
"""
import random
import numpy as np
import cv2


def _random_blend_mask(h: int, w: int) -> np.ndarray:
    """Returns float32 [0,1] mask H×W via ellipse, polygon, or convex hull."""
    mask = np.zeros((h, w), dtype=np.float32)
    choice = random.randint(0, 2)

    if choice == 0:
        cx = random.randint(w // 4, 3 * w // 4)
        cy = random.randint(h // 4, 3 * h // 4)
        rx = random.randint(w // 6, w // 2)
        ry = random.randint(h // 6, h // 2)
        cv2.ellipse(mask, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, (1.0,), -1)
    elif choice == 1:
        n = random.randint(5, 10)
        pts = np.array(
            [[random.randint(0, w - 1), random.randint(0, h - 1)] for _ in range(n)],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], (1.0,))
    else:
        n = random.randint(4, 8)
        pts = np.array(
            [[random.randint(0, w - 1), random.randint(0, h - 1)] for _ in range(n)],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, pts, (1.0,))

    # Soft edges — kernel must be odd and in [15, 35]
    ks = random.choice(range(15, 36, 2))
    return cv2.GaussianBlur(mask, (ks, ks), 0)


def _lab_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match color statistics of source to target in LAB space."""
    src_f = source.astype(np.float32) / 255.0
    tgt_f = target.astype(np.float32) / 255.0
    src_lab = cv2.cvtColor(src_f, cv2.COLOR_RGB2LAB)
    tgt_lab = cv2.cvtColor(tgt_f, cv2.COLOR_RGB2LAB)
    for c in range(3):
        s_mean, s_std = src_lab[:, :, c].mean(), src_lab[:, :, c].std() + 1e-6
        t_mean, t_std = tgt_lab[:, :, c].mean(), tgt_lab[:, :, c].std() + 1e-6
        tgt_lab[:, :, c] = (tgt_lab[:, :, c] - t_mean) / t_std * s_std + s_mean
    result = cv2.cvtColor(np.clip(tgt_lab, 0, None), cv2.COLOR_LAB2RGB)
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def _jpeg_compress(img: np.ndarray, quality_min: int = 40, quality_max: int = 100) -> np.ndarray:
    quality = random.randint(quality_min, quality_max)
    _, buf = cv2.imencode(
        '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def generate_sbi_pair(
    face1: np.ndarray,
    face2: np.ndarray,
    jpeg_quality_min: int = 40,
    jpeg_quality_max: int = 100,
) -> tuple:
    """
    face1: target face (H×W×3 uint8 RGB) → label 0 (real)
    face2: donor face  (H×W×3 uint8 RGB) → blended → label 1 (fake)
    Returns: (real_img, fake_img)
    """
    h, w = face1.shape[:2]

    # Optional LAB color transfer (50%)
    if random.random() < 0.5:
        face2 = _lab_color_transfer(face1, face2)

    mask = _random_blend_mask(h, w)[:, :, np.newaxis]  # H×W×1
    fake = (face1 * (1.0 - mask) + face2 * mask).clip(0, 255).astype(np.uint8)

    # Optional JPEG compression (70%)
    if random.random() < 0.7:
        fake = _jpeg_compress(fake, jpeg_quality_min, jpeg_quality_max)

    return face1, fake
