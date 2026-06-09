import json
import torch
import torch.nn as nn
from pathlib import Path


class DeepShieldEnsemble(nn.Module):
    _DEFAULT_W_SBI   = 0.4
    _DEFAULT_W_WCLIP = 0.6

    def __init__(self, sbi_model, wclip_model, weights_path=None):
        super().__init__()
        self.sbi   = sbi_model
        self.wclip = wclip_model

        w_sbi, w_wclip = self._DEFAULT_W_SBI, self._DEFAULT_W_WCLIP
        if weights_path and Path(weights_path).exists():
            with open(weights_path) as f:
                w = json.load(f)
            w_sbi   = float(w['w_sbi'])
            w_wclip = float(w['w_wclip'])
            print(f"Loaded ensemble weights: SBI={w_sbi:.3f}  WCLIP={w_wclip:.3f}")
        else:
            print(f"Using default ensemble weights: SBI={w_sbi}  WCLIP={w_wclip}")

        self.w_sbi:   torch.Tensor
        self.w_wclip: torch.Tensor
        self.register_buffer('w_sbi',   torch.tensor(w_sbi,   dtype=torch.float32))
        self.register_buffer('w_wclip', torch.tensor(w_wclip, dtype=torch.float32))

    @torch.no_grad()
    def forward(self, x_imagenet):
        """
        Returns (final_score, score_sbi, score_wclip) all in [0,1].
        x_imagenet: ImageNet-normalized B×3×224×224
        """
        score_sbi   = torch.sigmoid(self.sbi(x_imagenet)).squeeze(1)
        score_wclip = torch.sigmoid(self.wclip(x_imagenet)).squeeze(1)
        final_score = self.w_sbi * score_sbi + self.w_wclip * score_wclip
        return final_score, score_sbi, score_wclip
