"""
Fetch FaceForensics++ (c23) from the public TUM server (no access token needed)
and extract native-res face crops for LOCAL SBI training.

SBI self-blends REAL faces only, so we mainly need originals. We also grab one
fake method and hold out whole videos so we get an honest, identity-disjoint val set
(real deepfakes — NOT self-blends).

Output layout:
  <out>/sbi_real/                 real faces for SBI training (self-blend source)
  <out>/val_faces/real/           held-out real faces (val)
  <out>/val_faces/fake/           real-deepfake faces (val)

Usage:
  python -m data.preprocess.fetch_ffpp_local \
      --out D:/deepshield_data/ffpp \
      --n-real 300 --n-fake 120 --val-videos 40 --fps 1 --size 224
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

import cv2
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

# Import the detector module DIRECTLY from its file to bypass models/__init__.py,
# which eagerly imports wavelet_clip/CLIP (the SSL-malloc segfault vector).
_spec = importlib.util.spec_from_file_location(
    'retinaface_detector', str(_ROOT / 'models' / 'retinaface_detector.py'))
_rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rf)
FaceDetector = _rf.FaceDetector


def extract_from_video(video_path, output_dir, detector, fps_sample=1.0, size=224) -> int:
    """Minimal face extractor (inlined to avoid importing the models package)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(src_fps / fps_sample))
    saved = idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect(rgb)
            if faces:
                crop = detector.align_and_crop(rgb, faces[0], size=size)
                out = output_dir / f'{Path(video_path).stem}_{idx:06d}.jpg'
                cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
        idx += 1
    cap.release()
    return saved

SERVER = 'http://kaldir.vc.in.tum.de/faceforensics/v3/'
COMPRESSION = 'c23'
FAKE_METHOD = 'Deepfakes'   # val fake method (face-swap; matches NCII threat)


def _download(url: str, out_file: Path) -> bool:
    if out_file.exists():
        return True
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        fh, tmp = tempfile.mkstemp(dir=str(out_file.parent))
        os.close(fh)
        urllib.request.urlretrieve(url, tmp)
        os.rename(tmp, out_file)
        return True
    except Exception as e:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        tqdm.write(f'  FAILED {out_file.name}: {e}')
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--n-real', type=int, default=300, help='original videos to fetch')
    ap.add_argument('--n-fake', type=int, default=120, help='fake videos to fetch (val)')
    ap.add_argument('--val-videos', type=int, default=40, help='real videos held out for val')
    ap.add_argument('--fps', type=float, default=1.0)
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    out = args.out
    raw = out / 'raw'
    sbi_real = out / 'sbi_real'
    val_real = out / 'val_faces' / 'real'
    val_fake = out / 'val_faces' / 'fake'
    for d in (sbi_real, val_real, val_fake):
        d.mkdir(parents=True, exist_ok=True)

    print('Fetching FF++ file list...')
    pairs = json.loads(urllib.request.urlopen(SERVER + 'misc/filelist.json', timeout=30).read().decode())
    real_ids = sorted({i for pair in pairs for i in pair})            # '000', '001', ...
    fake_ids = ['_'.join(p) for p in pairs] + ['_'.join(p[::-1]) for p in pairs]

    real_ids = real_ids[:args.n_real]
    fake_ids = sorted(set(fake_ids))[:args.n_fake]

    # video-disjoint val: last N real ids -> val, rest -> train
    val_real_ids = set(real_ids[-args.val_videos:]) if args.val_videos else set()
    train_real_ids = [i for i in real_ids if i not in val_real_ids]
    print(f'real: {len(train_real_ids)} train + {len(val_real_ids)} val | fake(val): {len(fake_ids)}')

    detector = FaceDetector(device=args.device)
    real_base = f'{SERVER}original_sequences/youtube/{COMPRESSION}/videos/'
    fake_base = f'{SERVER}manipulated_sequences/{FAKE_METHOD}/{COMPRESSION}/videos/'

    def fetch_extract(vid_id: str, base: str, face_out: Path) -> int:
        mp4 = raw / f'{vid_id}.mp4'
        if not _download(base + f'{vid_id}.mp4', mp4):
            return 0
        n = extract_from_video(mp4, face_out, detector, fps_sample=args.fps, size=args.size)
        mp4.unlink(missing_ok=True)   # free disk immediately
        return n

    tot = 0
    for vid in tqdm(train_real_ids, desc='real->train'):
        tot += fetch_extract(vid, real_base, sbi_real)
    print(f'  SBI real faces: {len(list(sbi_real.glob("*.jpg")))}')

    for vid in tqdm(sorted(val_real_ids), desc='real->val'):
        fetch_extract(vid, real_base, val_real)
    print(f'  val real faces: {len(list(val_real.glob("*.jpg")))}')

    for vid in tqdm(fake_ids, desc='fake->val'):
        fetch_extract(vid, fake_base, val_fake)
    print(f'  val fake faces: {len(list(val_fake.glob("*.jpg")))}')

    if raw.exists():
        shutil.rmtree(raw, ignore_errors=True)

    print('\nDONE.')
    print(f'  ffhq_dir            -> {sbi_real}')
    print(f'  celeb_df_faces_dir  -> {out / "val_faces"}')


if __name__ == '__main__':
    main()
