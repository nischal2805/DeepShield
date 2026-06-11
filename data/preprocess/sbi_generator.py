"""
Self-Blended Image (SBI) augmentation — faithful re-implementation.
Ref: Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images", CVPR 2022.

Core idea: take ONE real face, make a source and a target copy from it,
apply mild source-target transforms (color / resolution / sharpness) to the
source, then blend the source into the target inside a face-shaped, deformed,
soft-edged mask. The result is a "fake" whose ONLY artifact is the blend
boundary — the exact cue a real face-swap leaves. The model therefore learns
to detect blending boundaries, which transfers to unseen deepfakes.

This replaces the previous (broken) version that blended two DIFFERENT faces
with random polygon masks — that taught the model nothing transferable.

Never pre-saves blends — called on-the-fly in dataset __getitem__.
"""
import random

import cv2
import numpy as np


# ── Source–target transforms (applied to the SOURCE copy only) ──────────────────
def _hsv_jitter(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + random.randint(-25, 25), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + random.randint(-25, 25), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _rgb_shift(img: np.ndarray) -> np.ndarray:
    shift = np.array([random.randint(-15, 15) for _ in range(3)], dtype=np.int16)
    return np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)


def _brightness_contrast(img: np.ndarray) -> np.ndarray:
    alpha = 1.0 + random.uniform(-0.15, 0.15)   # contrast
    beta  = random.uniform(-15, 15)             # brightness
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def _downscale_up(img: np.ndarray) -> np.ndarray:
    """Resolution mismatch: shrink then grow back → mild blur/aliasing."""
    h, w = img.shape[:2]
    scale = random.uniform(0.4, 0.9)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=random.uniform(0.5, 1.5))
    amt  = random.uniform(0.3, 1.2)
    return np.clip(img.astype(np.float32) * (1 + amt) - blur.astype(np.float32) * amt,
                   0, 255).astype(np.uint8)


_SOURCE_TRANSFORMS = [_hsv_jitter, _rgb_shift, _brightness_contrast, _downscale_up, _sharpen]


def _apply_source_transforms(img: np.ndarray) -> np.ndarray:
    """Apply a random non-empty subset of mild transforms to the source copy."""
    fns = [f for f in _SOURCE_TRANSFORMS if random.random() < 0.5]
    if not fns:
        fns = [random.choice(_SOURCE_TRANSFORMS)]
    random.shuffle(fns)
    out = img.copy()
    for f in fns:
        out = f(out)
    return out


def _lab_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Match color statistics of source to target in LAB space (reduces color seam)."""
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


# ── Face-region mask (landmark-free, for aligned face crops) ────────────────────
def _face_region_mask(h: int, w: int, landmarks: np.ndarray = None) -> np.ndarray:
    """
    Build a soft face-shaped mask.
    If landmarks (N×2) given → convex hull of landmarks (true SBI).
    Else → centered face ellipse (works because crops are face-aligned).
    """
    mask = np.zeros((h, w), dtype=np.float32)
    if landmarks is not None and len(landmarks) >= 3:
        hull = cv2.convexHull(landmarks.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
    else:
        cx, cy = w // 2 + random.randint(-w // 20, w // 20), h // 2 + random.randint(-h // 20, h // 20)
        rx = int(w * random.uniform(0.30, 0.42))
        ry = int(h * random.uniform(0.38, 0.50))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
    return mask


def _deform_mask(mask: np.ndarray) -> np.ndarray:
    """Random affine warp + erosion/dilation + soft blur → irregular boundary."""
    h, w = mask.shape
    # small affine jitter
    src = np.float32([[0, 0], [w, 0], [0, h]])
    jit = lambda v: v + random.uniform(-0.06, 0.06) * v if v else random.uniform(-6, 6)
    dst = np.float32([[jit(0), jit(0)], [jit(w), jit(0)], [jit(0), jit(h)]])
    M = cv2.getAffineTransform(src, dst)
    mask = cv2.warpAffine(mask, M, (w, h), borderValue=0.0)
    # random erode/dilate
    k = random.choice([3, 5, 7])
    kernel = np.ones((k, k), np.uint8)
    if random.random() < 0.5:
        mask = cv2.erode(mask, kernel, iterations=1)
    else:
        mask = cv2.dilate(mask, kernel, iterations=1)
    # soft edge — odd kernel in [9, 31]
    ks = random.choice(range(9, 32, 2))
    mask = cv2.GaussianBlur(mask, (ks, ks), 0)
    return np.clip(mask, 0.0, 1.0)


# ── Public API ──────────────────────────────────────────────────────────────────
def generate_self_blend(
    face: np.ndarray,
    landmarks: np.ndarray = None,
    jpeg_quality_min: int = 40,
    jpeg_quality_max: int = 100,
) -> tuple:
    """
    True SBI: blend a face with a transformed copy of ITSELF.

    face:       H×W×3 uint8 RGB real face.
    landmarks:  optional N×2 array (e.g. 5 from MTCNN, 68 from dlib) → better mask.
    Returns:    (real_img, fake_img)  both uint8 RGB, label 0 / 1 respectively.
    """
    h, w = face.shape[:2]
    target = face
    source = _apply_source_transforms(face)

    # small geometric offset of source within the frame (translation/scale)
    if random.random() < 0.5:
        tx, ty = random.randint(-w // 25, w // 25), random.randint(-h // 25, h // 25)
        s = random.uniform(0.97, 1.03)
        M = np.float32([[s, 0, tx], [0, s, ty]])
        source = cv2.warpAffine(source, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # color-match source to target so only the boundary, not global color, gives it away
    if random.random() < 0.5:
        source = _lab_color_transfer(target, source)

    mask = _deform_mask(_face_region_mask(h, w, landmarks))[:, :, np.newaxis]
    fake = (target * (1.0 - mask) + source * mask).clip(0, 255).astype(np.uint8)

    # shared post-processing (JPEG) on the fake so compression isn't a giveaway
    if random.random() < 0.7:
        q = random.randint(jpeg_quality_min, jpeg_quality_max)
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(fake, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, q])
        fake = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return face, fake


# Backward-compatible alias. Old signature took (face1, face2); donor is now ignored
# because true SBI self-blends. Kept so existing imports don't break.
def generate_sbi_pair(face1, face2=None, jpeg_quality_min=40, jpeg_quality_max=100):
    return generate_self_blend(face1, None, jpeg_quality_min, jpeg_quality_max)
