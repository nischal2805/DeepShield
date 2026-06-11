"""
AI-Generated Image Detector — UniversalFakeDetect linear probe on CLIP ViT-L/14.

Design constraints:
- CLIP CANNOT be loaded fresh (weights not in local HF cache, offline mode).
  Instead, reuse the CLIPModel already loaded inside the WaveletCLIP ensemble.
- Single nn.Linear(768, 1) probe; weights from UniversalFakeDetect pretrained.
- No extra dependencies beyond torch + transformers (already used by the project).

Usage (from flask_app.py):
    # after loading DeepShieldPipeline for Model 1:
    wclip_model = PIPELINES['1'].ensemble.wclip   # WaveletCLIP instance
    clip_handle  = wclip_model.clip_projection.clip  # transformers CLIPModel
    aigen = AIGenDetector(
        clip_model=clip_handle,
        to_clip_norm_fn=wclip_model._to_clip_norm,
        device=PIPELINES['1'].device,
        fc_path='D:/deepshield_data/checkpoints/aigen/fc_weights.pth',
    )
    prob = aigen.predict(imagenet_tensor_1x3x224x224)
"""

import torch
import torch.nn as nn


# ── Probe threshold (chosen from validation: separates Stylegan from real) ──────
# Validation (HEIC Conversions folder, 11 Stylegan + 27 Real):
#   AUC = 0.8956, raw (non-L2-normalized) features chosen (F1=0.7826 vs 0.75 for L2)
#   threshold=0.10 => TP=7 FP=2 FN=4 TN=25  (favors precision; use 0.075 for higher recall)
_DEFAULT_THRESHOLD = 0.10


class AIGenDetector(nn.Module):
    """
    Linear probe on CLIP ViT-L/14 projected image embeddings (768-d).
    Wraps an *already-loaded* CLIPModel — never calls from_pretrained().

    Args:
        clip_model      : a transformers.CLIPModel instance with real weights.
        to_clip_norm_fn : callable, converts an ImageNet-normalised tensor to
                          CLIP normalisation (wclip_model._to_clip_norm).
        device          : torch.device
        fc_path         : path to fc_weights.pth (bare Linear state_dict with
                          keys 'weight' [1,768] and 'bias' [1]).
        threshold       : decision threshold for P(AI-generated).
        use_l2_norm     : if True, L2-normalise features before the probe
                          (validated on local data — see FINAL REPORT).
    """

    def __init__(
        self,
        clip_model,
        to_clip_norm_fn,
        device,
        fc_path: str = 'D:/deepshield_data/checkpoints/aigen/fc_weights.pth',
        threshold: float = _DEFAULT_THRESHOLD,
        use_l2_norm: bool = False,
    ):
        super().__init__()

        self.clip            = clip_model        # shared — NOT owned here
        self._to_clip_norm   = to_clip_norm_fn
        self.device          = device
        self.threshold       = threshold
        self.use_l2_norm     = use_l2_norm

        # ── Linear probe 768 → 1 ─────────────────────────────────────────────
        self.fc = nn.Linear(768, 1)
        sd = torch.load(fc_path, map_location='cpu')
        # The official UniversalFakeDetect file uses bare keys 'weight'/'bias'
        self.fc.load_state_dict(sd)
        self.fc.to(device)
        self.fc.eval()

        print(f"[AIGenDetector] loaded probe from {fc_path}  "
              f"threshold={threshold}  l2_norm={use_l2_norm}")

    # ── Forward helper (does not move the shared CLIP — it stays on its device) ──
    @torch.no_grad()
    def _extract_features(self, t_imagenet: torch.Tensor) -> torch.Tensor:
        """
        t_imagenet : B×3×224×224, ImageNet-normalised, already on self.device.
        Returns     : B×768 CLIP projected image embedding.
        """
        x_clip = self._to_clip_norm(t_imagenet)          # re-normalise to CLIP stats
        feats  = self.clip.get_image_features(pixel_values=x_clip)  # B×768
        if self.use_l2_norm:
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return feats

    @torch.no_grad()
    def predict(self, t_imagenet: torch.Tensor) -> float:
        """
        Run the probe on a single-image tensor.

        t_imagenet : 1×3×224×224, ImageNet-normalised, on self.device
                     (same tensor produced by _TRANSFORM in pipeline.py).
        Returns    : float in [0, 1] — probability the image is AI-generated.
        """
        feats  = self._extract_features(t_imagenet)   # 1×768
        logit  = self.fc(feats)                        # 1×1
        prob   = float(torch.sigmoid(logit)[0, 0])
        return prob

    @torch.no_grad()
    def predict_batch(self, t_imagenet: torch.Tensor) -> torch.Tensor:
        """
        Batch version — returns B-length float tensor of probabilities.
        """
        feats = self._extract_features(t_imagenet)
        logit = self.fc(feats)
        return torch.sigmoid(logit).squeeze(1)
