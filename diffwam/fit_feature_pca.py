"""Per-layer standardisation + PCA (3072 -> D channels) of Wan2.2 DiT features, fitted on the 20 earlier Flash runs.

DiffWAM projects each tapped layer 5376 -> 512 with a learned linear map as its first operation; storing full
features (1.1 GB per sample for 4 steps) is impractical, so the data is stored already reduced by a fixed PCA.
Writes data/feature_pca.pt: mean, std (3, 3072), components (3, 3072, D), explained variance per layer.
"""

import glob
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = 128


def main():
    files = sorted(glob.glob(os.path.join(HERE, "..", "follow", "scenarios", "s*", "seed*", "flash.pt")))
    toks = []
    g = torch.Generator().manual_seed(0)
    for f in files:
        z = torch.load(f, weights_only=False, map_location="cpu")["Z"][0]  # (3, T, H, W, C)
        z = z.reshape(3, -1, z.shape[-1])
        idx = torch.randperm(z.shape[1], generator=g)[:4000]
        toks.append(z[:, idx].float())
    x = torch.cat(toks, dim=1).cuda()  # (3, N, C)
    mean, std = x.mean(1), x.std(1).clamp(min=1e-3)
    xs = (x - mean[:, None]) / std[:, None]
    comps, explained = [], []
    for layer in range(3):
        _, s, v = torch.pca_lowrank(xs[layer], q=D, center=False)
        comps.append(v[:, :D])
        explained.append(float((s[:D] ** 2).sum() / (xs[layer] ** 2).sum()))
    out = {"mean": mean.cpu(), "std": std.cpu(), "components": torch.stack(comps).cpu(), "explained": explained,
           "dim": D, "n_files": len(files)}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    torch.save(out, os.path.join(HERE, "data", "feature_pca.pt"))
    print(f"{len(files)} files, {x.shape[1]} tokens; explained variance per layer: {[round(e, 3) for e in explained]}")


if __name__ == "__main__":
    main()
