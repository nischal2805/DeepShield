"""
Batch-test DeepShield on a folder of images.

Usage:
    python batch_test.py "C:\\path\\to\\folder"
    python batch_test.py img1.jpg img2.png ...

Prints per-image verdict + SBI/WCLIP scores and a summary.
No Streamlit — runs on the main thread, loads models once.
"""
import os
import sys
import glob

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2
import torch
from inference.pipeline import DeepShieldPipeline, _TRANSFORM

EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def collect(args):
    paths = []
    for a in args:
        if os.path.isdir(a):
            for f in sorted(os.listdir(a)):
                if f.lower().endswith(EXTS):
                    paths.append(os.path.join(a, f))
        elif os.path.isfile(a):
            paths.append(a)
        else:
            paths.extend(glob.glob(a))
    return paths


def main():
    if len(sys.argv) < 2:
        print('Usage: python batch_test.py [--branch ensemble|wclip|sbi] <folder | image ...>')
        sys.exit(1)
    args = sys.argv[1:]
    branch = 'ensemble'
    if '--branch' in args:
        i = args.index('--branch')
        branch = args[i + 1]
        del args[i:i + 2]
    wclip_ckpt = None
    if '--wclip-ckpt' in args:
        i = args.index('--wclip-ckpt')
        wclip_ckpt = args[i + 1]
        del args[i:i + 2]
    paths = collect(args)
    if not paths:
        print('No images found.')
        sys.exit(1)

    p = DeepShieldPipeline(wclip_ckpt=wclip_ckpt)
    print(f"\nTesting {len(paths)} images "
          f"(weights SBI={float(p.ensemble.w_sbi):.2f} WCLIP={float(p.ensemble.w_wclip):.2f}, "
          f"device={p.device})\n")
    print(f"decision branch = {branch}")
    print(f"{'image':<46} {'VERDICT':<8} {'final':>6} {'sbi':>6} {'wclip':>6}  notes")
    print('-' * 90)

    n_fake = n_real = n_noface = n_err = 0
    for path in paths:
        name = os.path.basename(path)[:44]
        img = cv2.imread(path)
        if img is None:
            print(f"{name:<46} {'ERROR':<8} {'-':>6} {'-':>6} {'-':>6}  cannot read")
            n_err += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = p.detector.detect(img)
        if not faces:
            print(f"{name:<46} {'NOFACE':<8} {'-':>6} {'-':>6} {'-':>6}  no face -> track 2")
            n_noface += 1
            continue
        crop = p.detector.align_and_crop(img, faces[0], size=224)
        t = _TRANSFORM(crop).unsqueeze(0).to(p.device)
        with torch.no_grad():
            f, sb, wc = p.ensemble(t)
        f, sb, wc = float(f[0]), float(sb[0]), float(wc[0])
        score = {'ensemble': f, 'sbi': sb, 'wclip': wc}[branch]
        verdict = 'FAKE' if score > p.threshold else 'REAL'
        if verdict == 'FAKE':
            n_fake += 1
        else:
            n_real += 1
        note = f"{len(faces)} face(s)"
        print(f"{name:<46} {verdict:<8} {f:>6.3f} {sb:>6.3f} {wc:>6.3f}  {note}")

    print('-' * 90)
    print(f"FAKE={n_fake}  REAL={n_real}  NOFACE={n_noface}  ERROR={n_err}  (total {len(paths)})")


if __name__ == '__main__':
    main()
