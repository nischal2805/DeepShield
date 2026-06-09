"""
DeepShield inference pipeline.
Loads both models + ensemble weights and runs end-to-end detection.

Usage (from Python):
    from inference.pipeline import DeepShieldPipeline
    pipe = DeepShieldPipeline()
    result = pipe.detect('path/to/image.jpg')

Usage (CLI):
    python inference/pipeline.py --image path/to/image.jpg
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.gradcam import SBIGradCAM
from inference.track2 import track2_detect
from models.ensemble import DeepShieldEnsemble
from models.retinaface_detector import FaceDetector
from models.sbi_branch import SBIBranch
from models.wavelet_clip import WaveletCLIP

_IN_MEAN = [0.485, 0.456, 0.406]
_IN_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IN_MEAN, _IN_STD),
])

_DEFAULT_SBI_CKPT   = 'D:/deepshield_data/checkpoints/sbi/sbi_best_auc.pth'
_DEFAULT_WCLIP_CKPT = 'D:/deepshield_data/checkpoints/wavelet_clip/wavelet_clip_best_auc.pth'
_DEFAULT_WEIGHTS    = 'checkpoints/ensemble_weights.json'


class DeepShieldPipeline:
    def __init__(
        self,
        sbi_ckpt:   Optional[str] = None,
        wclip_ckpt: Optional[str] = None,
        weights:    Optional[str] = None,
        device:     str = 'auto',
        threshold:  float = 0.5,
    ):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device    = torch.device(device)
        self.threshold = threshold

        self.sbi_ckpt   = sbi_ckpt   or _DEFAULT_SBI_CKPT
        self.wclip_ckpt = wclip_ckpt or _DEFAULT_WCLIP_CKPT
        self.weights    = weights    or _DEFAULT_WEIGHTS

        self.ensemble: DeepShieldEnsemble
        self.detector: FaceDetector
        self.gradcam:  SBIGradCAM
        self._loaded   = False

        self._load_models()

    def _load_models(self) -> None:
        sbi_path   = Path(self.sbi_ckpt)
        wclip_path = Path(self.wclip_ckpt)

        if not sbi_path.exists() or not wclip_path.exists():
            missing = []
            if not sbi_path.exists():
                missing.append(str(sbi_path))
            if not wclip_path.exists():
                missing.append(str(wclip_path))
            raise FileNotFoundError(
                f"Checkpoint(s) not found:\n  " + "\n  ".join(missing) + "\n\n"
                "Train first (see HOW_TO_RUN.md) or update ckpt paths."
            )

        print("Loading SBI model...")
        sbi_model = SBIBranch(pretrained=False).to(self.device)
        ckpt = torch.load(self.sbi_ckpt, map_location=self.device)
        sbi_model.load_state_dict(ckpt['model_state'])
        sbi_model.eval()

        print("Loading Wavelet-CLIP model...")
        wclip_model = WaveletCLIP().to(self.device)
        ckpt = torch.load(self.wclip_ckpt, map_location=self.device)
        wclip_model.load_state_dict(ckpt['model_state'])
        wclip_model.eval()

        self.ensemble = DeepShieldEnsemble(
            sbi_model, wclip_model,
            weights_path=self.weights if Path(self.weights).exists() else None,
        ).to(self.device)

        self.detector = FaceDetector(device=str(self.device))
        self.gradcam  = SBIGradCAM(sbi_model, device=str(self.device))
        self._loaded  = True
        print("Pipeline ready.")

    def detect(self, image_path: str) -> dict:
        """
        Run end-to-end detection on a single image.
        Returns detection result dict.
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded.")

        t_start = time.perf_counter()

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        faces = self.detector.detect(img_rgb)

        if len(faces) == 0:
            result = track2_detect(image_path)
            result['latency_ms'] = round((time.perf_counter() - t_start) * 1000, 1)
            return result

        # Use largest face only
        face      = faces[0]
        face_crop = self.detector.align_and_crop(img_rgb, face, size=224)
        face_bbox = face['bbox']

        img_tensor = _TRANSFORM(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            final_score, score_sbi, score_wclip = self.ensemble(img_tensor)

        final_score  = float(final_score[0])
        score_sbi    = float(score_sbi[0])
        score_wclip  = float(score_wclip[0])
        verdict      = 'FAKE' if final_score > self.threshold else 'REAL'

        heatmap_b64 = self.gradcam.generate(img_tensor, face_crop)

        return {
            'verdict':     verdict,
            'confidence':  round(final_score, 4),
            'score_sbi':   round(score_sbi, 4),
            'score_wclip': round(score_wclip, 4),
            'heatmap_b64': heatmap_b64,
            'face_bbox':   face_bbox,
            'track':       1,
            'latency_ms':  round((time.perf_counter() - t_start) * 1000, 1),
        }

    def detect_array(self, img_rgb: np.ndarray) -> dict:
        """Detect from a numpy RGB array directly (for Streamlit use)."""
        if not self._loaded:
            raise RuntimeError("Models not loaded.")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        try:
            result = self.detect(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        return result


def main():
    parser = argparse.ArgumentParser(description="DeepShield single-image inference")
    parser.add_argument('--image',      required=True)
    parser.add_argument('--sbi-ckpt',   default=None)
    parser.add_argument('--wclip-ckpt', default=None)
    parser.add_argument('--weights',    default=None)
    parser.add_argument('--device',     default='auto')
    parser.add_argument('--threshold',  type=float, default=0.5)
    args = parser.parse_args()

    pipe   = DeepShieldPipeline(args.sbi_ckpt, args.wclip_ckpt, args.weights, args.device, args.threshold)
    result = pipe.detect(args.image)

    print(json.dumps({k: v for k, v in result.items() if k != 'heatmap_b64'}, indent=2))

    if result.get('heatmap_b64'):
        import base64
        out_path = Path(args.image).stem + '_heatmap.png'
        with open(out_path, 'wb') as f:
            f.write(base64.b64decode(result['heatmap_b64']))
        print(f"Heatmap saved to: {out_path}")


if __name__ == '__main__':
    main()
