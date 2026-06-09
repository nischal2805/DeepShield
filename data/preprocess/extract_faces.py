"""
Extract face crops from videos and image directories.
Usage:
    python -m data.preprocess.extract_faces \\
        --input  D:/deepshield_data/celeb_df_v2 \\
        --output D:/deepshield_data/celeb_df_v2/faces \\
        --fps 1 \\
        --size 224

Supports: mp4, avi, mov, jpg, png, webp
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.retinaface_detector import FaceDetector


SUPPORTED_VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_IMAGE = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    detector: FaceDetector,
    fps_sample: float = 1.0,
    size: int = 224,
) -> int:
    """Extract face crops from a video. Returns number of faces saved."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(src_fps / fps_sample))
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect(img_rgb)
            if faces:
                crop = detector.align_and_crop(img_rgb, faces[0], size=size)
                out_path = output_dir / f"{video_path.stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(
                    str(out_path),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                saved += 1
        frame_idx += 1

    cap.release()
    return saved


def extract_from_image(
    image_path: Path,
    output_dir: Path,
    detector: FaceDetector,
    size: int = 224,
) -> int:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect(img_rgb)
    if not faces:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    crop = detector.align_and_crop(img_rgb, faces[0], size=size)
    out_path = output_dir / f"{image_path.stem}.jpg"
    cv2.imwrite(
        str(out_path),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    return 1


def check_disk_space(path: str, min_gb: float = 3.0) -> bool:
    _, _, free = shutil.disk_usage(path)
    return (free / 1e9) >= min_gb


def run_extraction(
    input_dir: Path,
    output_dir: Path,
    detector: FaceDetector,
    fps_sample: float = 1.0,
    size: int = 224,
    max_videos: int = 0,       # 0 = no limit
    min_free_gb: float = 3.0,
) -> None:
    """
    Recursively process input_dir.
    Preserves relative directory structure under output_dir.
    If input contains subdirs named 'real' / 'fake', preserves them.
    """
    all_files = [
        p for p in sorted(input_dir.rglob('*'))
        if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO | SUPPORTED_IMAGE
    ]
    if not all_files:
        print(f"[ERROR] No supported files found under {input_dir}")
        return

    video_count = 0
    total_faces = 0

    for file_path in tqdm(all_files, desc="Extracting faces", unit="file"):
        # Preserve relative subpath
        rel = file_path.relative_to(input_dir)
        out_subdir = output_dir / rel.parent

        if not check_disk_space(str(output_dir.drive or '/'), min_free_gb):
            print(f"\n[WARN] Less than {min_free_gb}GB free — stopping extraction.")
            break

        if file_path.suffix.lower() in SUPPORTED_VIDEO:
            if max_videos and video_count >= max_videos:
                print(f"\n[INFO] Reached max_videos={max_videos}, stopping.")
                break
            n = extract_from_video(file_path, out_subdir, detector, fps_sample, size)
            video_count += 1
        else:
            n = extract_from_image(file_path, out_subdir, detector, size)

        total_faces += n

    print(f"\nDone. {total_faces} face crops saved to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract face crops from videos/images")
    parser.add_argument('--input',  required=True, type=Path, help="Input directory")
    parser.add_argument('--output', required=True, type=Path, help="Output directory")
    parser.add_argument('--fps',    type=float, default=1.0,  help="Frames per second to sample from videos")
    parser.add_argument('--size',   type=int,   default=224,  help="Output crop size (square)")
    parser.add_argument('--max-videos', type=int, default=0,  help="Max videos to process (0=all)")
    parser.add_argument('--min-free-gb', type=float, default=3.0, help="Stop if disk free < N GB")
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input directory does not exist: {args.input}")
        sys.exit(1)

    detector = FaceDetector(device=args.device)
    run_extraction(
        args.input, args.output, detector,
        fps_sample=args.fps,
        size=args.size,
        max_videos=args.max_videos,
        min_free_gb=args.min_free_gb,
    )


if __name__ == '__main__':
    main()
