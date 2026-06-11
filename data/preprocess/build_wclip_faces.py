"""
Build a WaveletCLIP face dataset from the locally-fetched FF++ crops.

Sources (from fetch_ffpp_local.py):
  D:/deepshield_data/ffpp/sbi_real/*.jpg          real (FF++ originals)
  D:/deepshield_data/ffpp/val_faces/real/*.jpg    real (held-out originals)
  D:/deepshield_data/ffpp/val_faces/fake/*.jpg    fake (FF++ Deepfakes)

Output (group-by-video split, balanced):
  D:/deepshield_data/wclip_faces/{train,val}/{real,fake}/*.jpg
"""
import random
import shutil
from pathlib import Path

SRC = Path('D:/deepshield_data/ffpp')
OUT = Path('D:/deepshield_data/wclip_faces')
VAL_FRAC = 0.12
SEED = 42


def video_id(name: str) -> str:
    return name.rsplit('.', 1)[0].rsplit('_', 1)[0]


def collect(paths):
    files = []
    for p in paths:
        if p.exists():
            files += sorted(p.glob('*.jpg'))
    return files


def main():
    random.seed(SEED)
    real = collect([SRC / 'sbi_real', SRC / 'val_faces' / 'real'])
    fake = collect([SRC / 'val_faces' / 'fake'])
    print(f'source: {len(real)} real, {len(fake)} fake')

    # balance: cap real to ~1.3x fake so the model is not swamped by reals
    cap = int(len(fake) * 1.3)
    if len(real) > cap:
        random.shuffle(real)
        real = real[:cap]
    print(f'after balance: {len(real)} real, {len(fake)} fake')

    for split in ('train', 'val'):
        for label in ('real', 'fake'):
            d = OUT / split / label
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    for label, files in (('real', real), ('fake', fake)):
        groups = {}
        for f in files:
            groups.setdefault(video_id(f.name), []).append(f)
        vids = sorted(groups)
        random.shuffle(vids)
        n_val = max(1, int(len(vids) * VAL_FRAC))
        val_vids = set(vids[:n_val])
        n_tr = n_va = 0
        for vid, items in groups.items():
            split = 'val' if vid in val_vids else 'train'
            for f in items:
                shutil.copy2(f, OUT / split / label / f.name)
                if split == 'val':
                    n_va += 1
                else:
                    n_tr += 1
        print(f'{label}: {len(val_vids)}/{len(vids)} videos -> val | train={n_tr} val={n_va}')

    print('\nfinal:')
    for split in ('train', 'val'):
        for label in ('real', 'fake'):
            n = len(list((OUT / split / label).glob('*.jpg')))
            print(f'  {split}/{label}: {n}')


if __name__ == '__main__':
    main()
