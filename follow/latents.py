"""Latent spaces for following an imagined video (env: conda `sakana_hyperdrone`, which has torch + diffusers).

Two spaces, both on a 15 x 26 token grid for a 416 x 240 image (Wan2.2's 832 x 480 canvas, halved):

* "flash": the DiffWAM-Flash features of the imagined video. Z (1, 3, 39, 15, 26, 3072) is normalised per layer and
  channel and projected per token to `FLASH_DIM` dims with a PCA fitted on training scenarios -> goal (39, 15, 26, D).
  `FlashEncoder` (ResNet-18, stride 16) maps an onboard image into that space.
* "vae": the Wan2.2 VAE latent of a single frame (encoded as a 1-frame video): (15, 26, 48) -> baseline, no training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_H, IMG_W = 240, 416
GRID_H, GRID_W = 15, 26
FLASH_DIM = 64
SLOT_DT = 5.0 / 38.0  # DiffWAM slots: t_s = s * 5/38 s, s = 0..38
FPS = 24
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


def slot_frame_index(num_frames: int = 121, num_slots: int = 39) -> np.ndarray:
    """Video frame shown at each DiffWAM slot time."""
    return np.minimum(np.round(np.arange(num_slots) * SLOT_DT * FPS).astype(int), num_frames - 1)


def to_input(frames, device="cuda") -> torch.Tensor:
    """(N, H, W, 3) uint8 at any size -> (N, 3, 240, 416) float in [0, 1] on `device`."""
    x = torch.as_tensor(np.asarray(frames), device=device).permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-2:] != (IMG_H, IMG_W):
        x = F.interpolate(x, size=(IMG_H, IMG_W), mode="area")
    return x


# ----------------------------------------------------------------------------------------------- flash targets


class FlashProjector:
    """Per-(layer, channel) standardisation + per-token PCA of Flash features, fitted on training videos."""

    def __init__(self, mean, std, components, scale):
        self.mean, self.std, self.components, self.scale = mean, std, components, scale  # (3,C) (3,C) (3C,D) (D,)

    @staticmethod
    def tokens(z: torch.Tensor) -> torch.Tensor:
        """Z (1, 3, T, H, W, C) -> (T, H, W, 3, C) float."""
        return z[0].float().permute(1, 2, 3, 0, 4)

    @classmethod
    def fit(cls, zs, dim: int = FLASH_DIM, max_tokens: int = 150_000, seed: int = 0):
        tok = torch.cat([cls.tokens(z).reshape(-1, z.shape[1], z.shape[-1]) for z in zs])  # (N, 3, C)
        g = torch.Generator().manual_seed(seed)
        tok = tok[torch.randperm(len(tok), generator=g)[:max_tokens]].cuda()
        mean, std = tok.mean(0), tok.std(0).clamp(min=1e-3)
        x = ((tok - mean) / std).flatten(1)
        _, s, v = torch.pca_lowrank(x, q=dim, center=False)
        comps = v[:, :dim]
        scale = (x @ comps).std(0).clamp(min=1e-6)
        explained = float((s[:dim] ** 2).sum() / (x ** 2).sum())
        return cls(mean, std, comps, scale), explained

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Z (1, 3, T, H, W, C) -> goal (T, H, W, D), unit variance per component."""
        t = self.tokens(z).to(self.mean.device)
        x = ((t - self.mean) / self.std).flatten(-2)
        return (x @ self.components) / self.scale

    def state_dict(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "components": self.components.cpu(),
                "scale": self.scale.cpu()}

    @classmethod
    def from_state_dict(cls, d, device="cuda"):
        return cls(*(d[k].to(device) for k in ("mean", "std", "components", "scale")))


class FlashEncoder(nn.Module):
    """Onboard image (N, 3, 240, 416) in [0, 1] -> (N, 15, 26, D): ResNet-18 up to layer3 (stride 16) + head."""

    def __init__(self, dim: int = FLASH_DIM, pretrained: bool = True):
        super().__init__()
        import torchvision

        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        self.head = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU(), nn.Conv2d(256, dim, 1))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x - self.img_mean) / self.img_std)).permute(0, 2, 3, 1)


# --------------------------------------------------------------------------------------------------- VAE space


class WanVAE:
    """Wan2.2 VAE encoder for single frames: (N, 3, 240, 416) in [0, 1] -> normalised latent (N, 15, 26, 48)."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(WAN_REPO, subfolder="vae", torch_dtype=dtype).to(device).eval()
        cfg = self.vae.config
        self.mean = torch.tensor(cfg.latents_mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(cfg.latents_std, device=device).view(1, -1, 1, 1)
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
        outs = []
        for i in range(0, len(x), batch):
            v = (x[i:i + batch] * 2 - 1).to(self.dtype)[:, :, None]  # (B, 3, 1, H, W)
            lat = self.vae.encode(v).latent_dist.mean[:, :, 0].float()  # (B, 48, 15, 26)
            outs.append(((lat - self.mean) / self.std).permute(0, 2, 3, 1))
        return torch.cat(outs)


def latent_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error per sample between (N, 15, 26, D) maps (b may be (15, 26, D))."""
    return ((a - b) ** 2).mean(dim=(-3, -2, -1))
