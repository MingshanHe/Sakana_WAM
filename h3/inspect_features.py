"""Summarise a features .pt from h3_features.py and save a PCA-to-RGB picture of the token grids.

Each row of the PNG is one (eval, layer); columns are the keyframe anchor A0 followed by every target frame of Z.
Spatially coherent colour regions that move across frames mean the features carry scene / motion structure.
"""

import argparse

import numpy as np
import torch
from PIL import Image


def pca_rgb(x: torch.Tensor) -> np.ndarray:
    """(..., C) -> (..., 3) uint8 via the top-3 principal components, shared over all tokens given."""
    flat = x.reshape(-1, x.shape[-1]).float()
    flat = flat - flat.mean(0)
    _, _, v = torch.pca_lowrank(flat, q=3, center=False)
    proj = flat @ v[:, :3]
    lo, hi = proj.quantile(0.01, dim=0), proj.quantile(0.99, dim=0)
    proj = ((proj - lo) / (hi - lo)).clamp(0, 1)
    return (proj.reshape(*x.shape[:-1], 3).numpy() * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features")
    parser.add_argument("--out", default=None)
    parser.add_argument("--scale", type=int, default=8, help="Pixels per token in the PNG.")
    args = parser.parse_args()

    feats = torch.load(args.features, weights_only=False)
    z, a0 = feats["Z"], feats["A0"]
    print("request:", feats["request"])
    print("layout :", feats["layout"])
    print(f"Z  {tuple(z.shape)} {z.dtype}  (K, L, T, H, W, C)")
    if a0 is not None:
        print(f"A0 {tuple(a0.shape)} {a0.dtype}")
    for ki, k in enumerate(feats["evals"]):
        for li, l in enumerate(feats["layers"]):
            f = z[ki, li].float()
            print(
                f"eval {k} block {l:2d}: t={feats['timestep_Z'][ki].unique().tolist()} "
                f"mean={f.mean():+.3f} std={f.std():.3f} |x|max={f.abs().max():.1f} "
                f"frame-to-frame cos={torch.cosine_similarity(f[1:].flatten(1), f[:-1].flatten(1)).mean():.3f}"
            )

    rows = []
    for ki in range(z.shape[0]):
        for li in range(z.shape[1]):
            grids = z[ki, li] if a0 is None else torch.cat([a0[ki, li], z[ki, li]])
            rgb = pca_rgb(grids)  # (N, H, W, 3)
            gap = np.full((rgb.shape[1], 1, 3), 255, np.uint8)
            rows.append(np.concatenate([np.concatenate([g, gap], 1) for g in rgb], 1))
    sep = np.full((1, rows[0].shape[1], 3), 255, np.uint8)
    canvas = np.concatenate([np.concatenate([r, sep], 0) for r in rows], 0)
    out = args.out or args.features.replace(".pt", "_pca.png")
    Image.fromarray(canvas).resize((canvas.shape[1] * args.scale, canvas.shape[0] * args.scale), Image.NEAREST).save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
