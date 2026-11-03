"""Top-down trajectories of the test samples: teacher (gen2pose of the completed rollout) vs DiffWAM / Flash heads.

Writes data/trajectories.png: one panel per test sample, start at the origin facing +x (forward up the page is +x,
left is +y), with the predicted 39 poses of each head, the teacher, and the final heading arrows.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEADS = {"diffwam": ("#1f77b4", "DiffWAM (steps 0-3)"), "flash": ("#ff7f0e", "Flash (step 0)")}


def main():
    import torch

    from train_head import ReadoutHead, batchify, load_split, to_poses

    preds = {}
    for tag in HEADS:  # run each head on every test sample (training saved predictions for trusted ones only)
        path = os.path.join(HERE, "data", "head", f"{tag}.pt")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        model = ReadoutHead(k=len(ckpt["evals"])).eval()
        model.load_state_dict(ckpt["model"])
        samples = load_split(os.path.join(HERE, "data", "samples_eval"), ckpt["evals"])
        z, a0, geo, s0, _ = batchify(samples, device="cpu")
        with torch.no_grad():
            p = to_poses(model(z, a0, geo), s0).numpy()
        preds[tag] = {s["id"]: p[i] for i, s in enumerate(samples)}
    deploy = json.load(open(os.path.join(HERE, "data", "deploy", "diffwam", "deploy.json")))["samples"]
    ids = [r["id"] for r in deploy]
    trusted = {r["id"]: r["teacher_trusted"] for r in deploy}
    cols = 5
    rows = (len(ids) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    for ax, sid in zip(axes.flat, ids):
        teacher = np.load(os.path.join(HERE, "data", "samples_eval", sid, "teacher.npz"))["pose"]
        ax.plot(-teacher[:, 1], teacher[:, 0], "k-", lw=2, label="teacher (rollout)" if sid == ids[0] else None)
        for tag, (color, label) in HEADS.items():
            if tag in preds and sid in preds[tag]:
                p = preds[tag][sid]
                ax.plot(-p[:, 1], p[:, 0], "-", color=color, lw=1.5, label=label if sid == ids[0] else None)
                ax.arrow(-p[-1, 1], p[-1, 0], -0.3 * np.sin(p[-1, 3]), 0.3 * np.cos(p[-1, 3]), color=color,
                         head_width=0.08)
        ax.plot(0, 0, "go")
        ax.set_aspect("equal")
        lim = max(1.5, np.abs(teacher[:, :2]).max() + 0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.5, 2 * lim - 0.5)
        ax.set_title(f"{sid}{'' if trusted[sid] else ' (teacher untrusted)'}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(ids):]:
        ax.axis("off")
    fig.legend(loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Top-down, start at origin facing up (x forward, left to the left) [m]", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = os.path.join(HERE, "data", "trajectories.png")
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
