import numpy as np
import cv2
from typing import Optional

try:
    from insightface.app import FaceAnalysis as _FaceAnalysis  # type: ignore[import-untyped]
    _INSIGHTFACE = True
except ImportError:
    _INSIGHTFACE = False

try:
    from facenet_pytorch import MTCNN as _MTCNN  # type: ignore[import-untyped]
    _FACENET = True
except ImportError:
    _FACENET = False


class FaceDetector:
    """
    RetinaFace via insightface (preferred) with MTCNN fallback.

    Windows install note: if insightface fails, try:
        pip install insightface --no-build-isolation
    Requires Visual C++ Build Tools to be installed.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.backend: Optional[str] = None
        self._model = None
        self._init()

    def _init(self) -> None:
        if _INSIGHTFACE:
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if self.device == 'cuda'
                else ['CPUExecutionProvider']
            )
            self._model = _FaceAnalysis(providers=providers)
            self._model.prepare(
                ctx_id=0 if self.device == 'cuda' else -1,
                det_size=(640, 640),
            )
            self.backend = 'insightface'
        elif _FACENET:
            import torch
            dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model = _MTCNN(device=dev, keep_all=True, thresholds=[0.6, 0.7, 0.7])
            self.backend = 'mtcnn'
        else:
            raise RuntimeError(
                "No face detector found.\n"
                "  pip install insightface          (recommended)\n"
                "  pip install facenet-pytorch      (fallback)\n"
                "If insightface fails on Windows: pip install insightface --no-build-isolation"
            )
        print(f"[FaceDetector] backend={self.backend}")

    def detect(self, img_rgb: np.ndarray) -> list:
        """
        img_rgb: H×W×3 uint8 RGB array.
        Returns list of dicts: {'bbox': [x1,y1,x2,y2], 'score': float}
        Sorted by face area descending (largest face first).
        """
        result = []
        if self.backend == 'insightface' and self._model is not None:
            faces = self._model.get(img_rgb)
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                result.append({'bbox': [x1, y1, x2, y2], 'score': float(f.det_score)})
        elif self.backend == 'mtcnn' and self._model is not None:
            from PIL import Image
            ret = self._model.detect(Image.fromarray(img_rgb))
            boxes = ret[0]
            probs = ret[1]
            if boxes is not None and probs is not None:
                for box, prob in zip(boxes, probs):
                    if prob is not None and float(prob) > 0.9:
                        x1, y1, x2, y2 = [int(v) for v in box[:4]]
                        result.append({'bbox': [x1, y1, x2, y2], 'score': float(prob)})

        # Sort largest face first
        result.sort(
            key=lambda f: (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]),
            reverse=True,
        )
        return result

    def align_and_crop(self, img_rgb: np.ndarray, face: dict, size: int = 224) -> np.ndarray:
        """Crop face with 20% margin, resize to size×size. Returns H×W×3 uint8 RGB."""
        x1, y1, x2, y2 = face['bbox']
        h, w = img_rgb.shape[:2]
        mx = int((x2 - x1) * 0.2)
        my = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(img_rgb, (size, size))
        return cv2.resize(crop, (size, size))
