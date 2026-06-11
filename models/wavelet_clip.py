import torch
import torch.nn as nn
import numpy as np
import pywt
from torchvision import models
from transformers import CLIPModel


class WaveletBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Haar DWT → 4×112×112 input
        self.stem = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64×56×56
        )
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.layer1 = resnet.layer1  # 64→64,  56×56
        self.layer2 = resnet.layer2  # 64→128, 28×28
        self.layer3 = resnet.layer3  # 128→256, 14×14
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(nn.Linear(256, 512), nn.ReLU(inplace=True))

    @staticmethod
    def _apply_dwt(x: torch.Tensor) -> torch.Tensor:
        """Convert B×3×H×W → B×4×(H/2)×(W/2) via grayscale Haar DWT."""
        # Grayscale conversion (no grad needed — DWT has no learnable params)
        gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])
        gray_np = gray.detach().cpu().float().numpy()

        batch_out = []
        for i in range(gray_np.shape[0]):
            LL, (LH, HL, HH) = pywt.dwt2(gray_np[i], 'haar')
            batch_out.append(np.stack([LL, LH, HL, HH], axis=0))  # 4×H/2×W/2

        arr = np.array(batch_out, dtype=np.float32)
        return torch.from_numpy(arr).to(x.device)

    def forward(self, x):
        dwt = self._apply_dwt(x)       # B×4×112×112
        out = self.stem(dwt)            # B×64×56×56
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool(out).flatten(1)  # B×256
        return self.proj(out)            # B×512


class CLIPProjection(nn.Module):
    # CLIP ViT-L/14 CLS token → 1024-dim → FC → 512-dim
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.clip = CLIPModel.from_pretrained(clip_model_name)
        else:
            from transformers import CLIPConfig
            cfg = CLIPConfig.from_pretrained(clip_model_name, local_files_only=True)
            self.clip = CLIPModel(cfg)  # type: ignore[arg-type]
        for p in self.clip.parameters():
            p.requires_grad = False

        self.proj = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))

    def unfreeze_last_blocks(self, n: int) -> list:
        """
        Unfreeze the last `n` CLIP vision transformer blocks (+ final layernorm)
        so the backbone can adapt to deepfake cues instead of underfitting on a
        frozen encoder. Returns the list of now-trainable parameters.
        """
        if n <= 0:
            return []
        trainable = []
        layers = self.clip.vision_model.encoder.layers
        for blk in layers[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
                trainable.append(p)
        for p in self.clip.vision_model.post_layernorm.parameters():
            p.requires_grad = True
            trainable.append(p)
        return trainable

    def forward(self, pixel_values):
        out = self.clip.vision_model(pixel_values=pixel_values)
        cls_token = out.last_hidden_state[:, 0, :]  # B×1024
        return self.proj(cls_token)                  # B×512


class WaveletCLIP(nn.Module):
    # CLIP normalization differs from ImageNet — handled here internally
    _CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    _CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
    _IN_MEAN   = [0.485, 0.456, 0.406]
    _IN_STD    = [0.229, 0.224, 0.225]

    def __init__(self, clip_model_name="openai/clip-vit-large-patch14", pretrained: bool = True):
        super().__init__()
        self.wavelet_branch  = WaveletBranch()
        self.clip_projection = CLIPProjection(clip_model_name, pretrained=pretrained)

        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Register norm tensors as buffers so they move with .to(device)
        self.register_buffer('_in_mean',   torch.tensor(self._IN_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_in_std',    torch.tensor(self._IN_STD).view(1, 3, 1, 1))
        self.register_buffer('_clip_mean', torch.tensor(self._CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_clip_std',  torch.tensor(self._CLIP_STD).view(1, 3, 1, 1))

    def unfreeze_clip_layers(self, n: int) -> list:
        """Expose CLIP partial unfreeze to the trainer. Returns trainable params."""
        return self.clip_projection.unfreeze_last_blocks(n)

    def _to_clip_norm(self, x_imagenet):
        """Re-normalize from ImageNet to CLIP normalization."""
        x_raw = x_imagenet * self._in_std + self._in_mean  # → [0,1]
        return (x_raw - self._clip_mean) / self._clip_std

    def forward(self, x_imagenet):
        """
        x_imagenet: ImageNet-normalized tensor B×3×224×224
        Returns raw logits B×1 (apply sigmoid for probability).
        """
        wavelet_feat = self.wavelet_branch(x_imagenet)

        x_clip = self._to_clip_norm(x_imagenet)
        clip_feat = self.clip_projection(x_clip)

        fused = torch.cat([wavelet_feat, clip_feat], dim=1)  # B×1024
        return self.fusion(fused)  # B×1
