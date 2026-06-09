"""
Grad-CAM heatmap generation using pytorch-grad-cam library.
Target layer: model.backbone.conv_head (EfficientNet-B4 final conv).
"""
import base64
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image

_GRADCAM_AVAILABLE = False
try:
    from pytorch_grad_cam import GradCAM as _GradCAM  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.image import show_cam_on_image as _show_cam  # type: ignore[import-untyped]
    from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget as _BinTarget  # type: ignore[import-untyped]
    _GRADCAM_AVAILABLE = True
except ImportError:
    pass


class SBIGradCAM:
    """Grad-CAM wrapper for SBI branch. Returns None if library missing."""

    def __init__(self, model, device: str = 'cuda'):
        self.model  = model
        self.device = device
        self._cam   = None

        if _GRADCAM_AVAILABLE:
            target_layer = [self.model.backbone.conv_head]
            self._cam = _GradCAM(model=self.model, target_layers=target_layer)  # type: ignore[reportUnboundVariable]
        else:
            print("[GradCAM] pytorch-grad-cam not installed. Heatmaps disabled.")

    def generate(self, img_tensor: torch.Tensor, face_rgb: np.ndarray) -> Optional[str]:
        """
        img_tensor: 1×3×H×W ImageNet-normalized tensor on device
        face_rgb:   H×W×3 uint8 RGB array (original face crop)
        Returns base64-encoded PNG string, or None if unavailable.
        """
        if self._cam is None or not _GRADCAM_AVAILABLE:
            return None

        targets        = [_BinTarget(1)]  # type: ignore[reportUnboundVariable]
        grayscale_cam  = self._cam(input_tensor=img_tensor, targets=targets)[0]  # H×W
        face_float     = face_rgb.astype(np.float32) / 255.0
        visualization  = _show_cam(face_float, grayscale_cam, use_rgb=True)  # type: ignore[reportUnboundVariable]

        img_pil = Image.fromarray(visualization)
        buf     = BytesIO()
        img_pil.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode('utf-8')


def overlay_heatmap_on_image(
    original_rgb: np.ndarray,
    face_bbox: list,
    heatmap_b64: Optional[str],
) -> np.ndarray:
    """Composite Grad-CAM heatmap onto original image at the face bbox location."""
    result = original_rgb.copy()
    if heatmap_b64 is None:
        return result

    x1, y1, x2, y2  = face_bbox
    heatmap_bytes    = base64.b64decode(heatmap_b64)
    heatmap_pil      = Image.open(BytesIO(heatmap_bytes)).convert('RGB')
    heatmap_rgb      = np.array(
        heatmap_pil.resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)  # type: ignore[attr-defined]
    )
    result[y1:y2, x1:x2] = heatmap_rgb
    return result
