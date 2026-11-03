"""Phase 2: train the onboard-camera encoder into the Wan2.2 DiffWAM-Flash latent space (env `sakana_hyperdrone`).

Pairs: (imagined frame shown at slot s, Flash goal map of slot s), 39 per imagined video. Scenario-held-out folds:
fold k trains on every scenario except two and evaluates on those two, so a follower that uses fold k's encoder on
a held-out scenario has never seen that house. Per fold it saves encoder + Flash projector to
<out-dir>/fold<k>.pt, and the evaluation to <out-dir>/metrics.json:

  r2           1 - MSE / Var of the goal maps (goals have unit variance per component)
  slot_top1    fraction of held-out frames whose nearest goal slot (latent error) is the right one, and within +-2
  rank_corr    Spearman correlation between latent error and |frame slot - goal slot| (higher = smoother basin)

It also caches the VAE goals of every imagined video: <scenario>/seed<k>/goals_vae.pt (39, 15, 26, 48).

    python train_encoder.py --scenarios scenarios --out-dir encoders
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from latents import FlashEncoder, FlashProjector, WanVAE, latent_error, slot_frame_index, to_input


def load_runs(root: str) -> list[dict]:
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "s*", "seed*"))):
        if not os.path.exists(os.path.join(d, "flash.pt")):
            continue
        video = np.load(os.path.join(d, "video.npy"), mmap_mode="r")
        frames = np.ascontiguousarray(video[slot_frame_index(len(video))])  # (39, 480, 832, 3)
        flash = torch.load(os.path.join(d, "flash.pt"), weights_only=False)
        runs.append({"dir": d, "scenario": os.path.basename(os.path.dirname(d)), "frames": frames, "z": flash["Z"]})
    return runs


def evaluate(enc, goals_list, frames_list) -> dict:
    enc.eval()
    sse = sst = 0.0
    top1 = top3 = n = 0
    corrs = []
    with torch.no_grad():
        for goals, frames in zip(goals_list, frames_list):
            pred = enc(to_input(frames))  # (39, 15, 26, D)
            sse += float(((pred - goals) ** 2).sum())
            sst += float(((goals - goals.mean(dim=(0, 1, 2))) ** 2).sum())
            d = torch.stack([latent_error(pred, g) for g in goals], dim=1)  # (frame slot, goal slot)
            best = d.argmin(1).cpu().numpy()
            idx = np.arange(len(best))
            top1 += int((best == idx).sum())
            top3 += int((np.abs(best - idx) <= 2).sum())
            n += len(best)
            dist = np.abs(idx[:, None] - idx[None, :])
            corrs.append(spearmanr(d.cpu().numpy().ravel(), dist.ravel()).statistic)
    enc.train()
    return {"r2": 1 - sse / sst, "slot_top1": top1 / n, "slot_within2": top3 / n, "rank_corr": float(np.mean(corrs))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--out-dir", default="encoders")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--split", choices=["scenario", "seed"], default="scenario",
                        help="scenario: hold out whole scenarios (folds); seed: train seed 0, test seed 1 (noise check)")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)

    runs = load_runs(args.scenarios)
    scenarios = sorted({r["scenario"] for r in runs})
    print(f"{len(runs)} imagined videos over {len(scenarios)} scenarios", flush=True)

    vae = WanVAE()
    for r in runs:
        path = os.path.join(r["dir"], "goals_vae.pt")
        if not os.path.exists(path):
            torch.save(vae(to_input(r["frames"])).cpu(), path)
    del vae
    torch.cuda.empty_cache()

    per_fold = max(1, len(scenarios) // args.folds)
    results = {"folds": [], "args": vars(args)}
    if args.split == "seed":  # same scenarios, different noise: train on seed 0, test on seed 1
        args.folds = 1
    for k in range(args.folds):
        if args.split == "seed":
            held = ["seed1 of every scenario"]
            train = [r for r in runs if r["dir"].endswith("seed0")]
            test = [r for r in runs if r["dir"].endswith("seed1")]
        else:
            held = scenarios[k * per_fold:(k + 1) * per_fold]
            train = [r for r in runs if r["scenario"] not in held]
            test = [r for r in runs if r["scenario"] in held]
        t0 = time.perf_counter()
        proj, explained = FlashProjector.fit([r["z"] for r in train])
        tr_goals = [proj(r["z"]) for r in train]
        te_goals = [proj(r["z"]) for r in test]
        x_all = torch.cat([to_input(r["frames"], "cpu") for r in train])  # (N, 3, 240, 416) on CPU
        y_all = torch.cat(tr_goals)  # (N, 15, 26, D) on GPU

        enc = FlashEncoder().cuda()
        opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
        for step in range(args.steps):
            i = torch.randint(len(x_all), (args.batch,))
            x = x_all[i].cuda()
            # Photometric jitter only: geometry must stay aligned with the goal token grid.
            x = (x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, device="cuda"))
                 + 0.1 * (torch.rand(len(x), 3, 1, 1, device="cuda") - 0.5)).clamp(0, 1)
            loss = ((enc(x) - y_all[i.cuda()]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0 or step == args.steps - 1:
                print(f"fold {k} step {step} loss {loss.item():.4f}", flush=True)

        train_m = evaluate(enc, tr_goals, [r["frames"] for r in train])
        test_m = evaluate(enc, te_goals, [r["frames"] for r in test])
        fold = {"fold": k, "held_out": held, "pca_explained": explained, "train": train_m, "test": test_m,
                "seconds": time.perf_counter() - t0}
        results["folds"].append(fold)
        print(json.dumps(fold), flush=True)
        torch.save({"encoder": enc.state_dict(), "projector": proj.state_dict(), "held_out": held},
                   os.path.join(args.out_dir, f"fold{k}.pt"))

    for split in ("train", "test"):
        results[f"mean_{split}"] = {m: float(np.mean([f[split][m] for f in results["folds"]]))
                                    for m in results["folds"][0][split]}
    json.dump(results, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    print("mean train", json.dumps(results["mean_train"]))
    print("mean test ", json.dumps(results["mean_test"]))


if __name__ == "__main__":
    main()
