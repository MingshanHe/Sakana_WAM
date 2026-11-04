"""Train DreamerV3 (PyTorch) on the HyperDrone flight replay (env: conda `sakana_hyperdrone`).

Phase 1 (world model only):   python train.py --tag wm200m --steps 60000
Phase 2 (actor-critic in imagination, world model continues to train or is frozen):
                              python train.py --tag ac200m --init data/runs/wm200m/best.pt --ac --steps 20000

Replay: data/replay/{train,val}/h<house>_r<round>/<key>.npy, streams (num_drones, steps, ...), loaded into RAM.
Batches are 16 x 64 chunks sampled uniformly; chunks may cross episode boundaries (is_first resets the RSSM).
Validation: world-model losses on the held-out houses, plus an open-loop check: 16 context steps, then 32 imagined
steps under the logged actions, compared with what really happened (image MSE, reward MAE, crash prediction).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import queue
import threading
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, HERE)
from dv3.agent import ActorCritic, WorldModel  # noqa: E402
from dv3.nets import TwoHot  # noqa: E402

KEYS = ("image", "vector", "action", "reward", "is_first", "is_terminal")


class Replay:
    def __init__(self, dirs, seed=0):
        self.streams = []
        t0 = time.time()
        for d in sorted(dirs):
            self.streams.append({k: np.load(os.path.join(d, k + ".npy")) for k in KEYS})
        self.sizes = np.array([s["reward"].shape[0] * s["reward"].shape[1] for s in self.streams], float)
        self.rng = np.random.default_rng(seed)
        self.frames = int(self.sizes.sum())
        print(f"replay: {len(self.streams)} files, {self.frames} frames, loaded in {time.time() - t0:.0f}s", flush=True)

    def add(self, d):
        self.streams.append({k: np.load(os.path.join(d, k + ".npy")) for k in KEYS})
        self.sizes = np.append(self.sizes, self.streams[-1]["reward"].size)
        self.frames = int(self.sizes.sum())

    def sample(self, b, t):
        files = self.rng.choice(len(self.streams), b, p=self.sizes / self.sizes.sum())
        out = {k: [] for k in KEYS}
        for f in files:
            s = self.streams[f]
            n, steps = s["reward"].shape
            i, k = self.rng.integers(n), self.rng.integers(steps - t + 1)
            for key in KEYS:
                out[key].append(s[key][i, k:k + t])
        batch = {k: np.stack(v) for k, v in out.items()}
        batch["is_first"][:, 0] = True  # every chunk starts from a fresh state
        return batch


def to_torch(batch, device):
    out = {}
    for k, v in batch.items():
        x = torch.from_numpy(v).pin_memory().to(device, non_blocking=True)
        out[k] = x if k == "image" else (x.bool() if k.startswith("is_") else x.float())
    return out


def prefetch(replay, b, t, n=4):
    q = queue.Queue(n)

    def work():
        while True:
            q.put(replay.sample(b, t))

    for _ in range(2):
        threading.Thread(target=work, daemon=True).start()
    return q


@torch.no_grad()
def evaluate(wm, twohot, val, device, batches=8, b=16, ctx=16, horizon=32):
    wm.eval()
    rng_state = val.rng.bit_generator.state
    val.rng = np.random.default_rng(123)  # same validation chunks every time
    agg = {}
    ol = {"img_mse_px": [], "rew_mae": [], "crash_auc_pos": [], "crash_auc_neg": []}
    for _ in range(batches):
        batch = to_torch(val.sample(b, ctx + horizon), device)
        with torch.autocast("cuda", torch.bfloat16):
            _, m, _, _ = wm.loss(batch, twohot)
            for k, v in m.items():
                agg.setdefault(k, []).append(float(v))
            ctx_batch = {k: v[:, :ctx] for k, v in batch.items()}
            tokens = wm.enc(ctx_batch["image"].flatten(0, 1), ctx_batch["vector"].flatten(0, 1)).reshape(b, ctx, -1)
            out, (state, prev_a) = wm.rssm.observe(tokens, ctx_batch["action"], ctx_batch["is_first"])
            alive = torch.ones(b, dtype=torch.bool, device=device)
            for k in range(horizon):
                a = batch["action"][:, ctx + k - 1] if k else prev_a
                state = wm.rssm.img_step(state, a.to(state["deter"].dtype))
                feat = wm.feat(state)
                img_pred, _ = wm.dec(feat)
                alive &= ~batch["is_first"][:, ctx + k]  # stop scoring a chunk once a new episode starts
                target = batch["image"][:, ctx + k].permute(0, 3, 1, 2).float() / 255.0 - 0.5
                mse = (img_pred.float() - target).pow(2).mean((1, 2, 3))
                rew = twohot.mean(wm.rew(feat))
                con = torch.sigmoid(wm.con(feat)[..., 0].float())
                term = batch["is_terminal"][:, ctx + k]
                ol["img_mse_px"].append(mse[alive].cpu().numpy())
                ol["rew_mae"].append((rew - batch["reward"][:, ctx + k]).abs()[alive].cpu().numpy())
                ol["crash_auc_pos"].append((1 - con)[alive & term].cpu().numpy())
                ol["crash_auc_neg"].append((1 - con)[alive & ~term].cpu().numpy())
                alive &= ~term
    val.rng.bit_generator.state = rng_state
    res = {f"val_{k}": float(np.mean(v)) for k, v in agg.items()}
    res["ol_img_mse_px"] = float(np.mean(np.concatenate(ol["img_mse_px"])))
    res["ol_rew_mae"] = float(np.mean(np.concatenate(ol["rew_mae"])))
    pos, neg = np.concatenate(ol["crash_auc_pos"]), np.concatenate(ol["crash_auc_neg"])
    if len(pos) and len(neg):  # probability a real crash step scores above a non-crash step
        res["ol_crash_auc"] = float((pos[:, None] > neg[None, :]).mean())
        res["ol_crashes"] = int(len(pos))
    wm.train()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data", "replay"))
    ap.add_argument("--size", default="size200m")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ac-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--vec-scale", type=float, default=1.0,
                    help="weight of the vector (state + plan waypoints) reconstruction loss (DreamerV3: 1)")
    ap.add_argument("--ac", action="store_true", help="also train the actor-critic in imagination")
    ap.add_argument("--freeze-wm", action="store_true")
    ap.add_argument("--init", default="")
    ap.add_argument("--resume", action="store_true", help="continue data/runs/<tag>/last.pt (weights, optimisers, step)")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--max-train-files", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda"
    run = os.path.join(HERE, "data", "runs", args.tag)
    os.makedirs(run, exist_ok=True)
    train_dirs = sorted(glob.glob(os.path.join(args.data, "train", "h*")))
    if args.max_train_files:
        train_dirs = train_dirs[: args.max_train_files]
    val_dirs = sorted(glob.glob(os.path.join(args.data, "val", "h*")))
    train, val = Replay(train_dirs, seed=0), Replay(val_dirs, seed=1)
    s0 = train.streams[0]
    img_ch, vec_dim, act_dim = s0["image"].shape[-1], s0["vector"].shape[-1], s0["action"].shape[-1]

    wm = WorldModel(img_ch, vec_dim, act_dim, args.size).to(device)
    wm.vec_scale = args.vec_scale
    ac = ActorCritic(wm.feat_dim, act_dim, wm.units).to(device)
    n_wm = sum(p.numel() for p in wm.parameters())
    n_ac = sum(p.numel() for p in ac.actor.parameters()) + sum(p.numel() for p in ac.critic.parameters())
    print(f"params: world model {n_wm / 1e6:.1f}M, actor+critic {n_ac / 1e6:.1f}M, total {(n_wm + n_ac) / 1e6:.1f}M", flush=True)
    start_step = 0
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        wm.load_state_dict(ck["wm"])
        if "ac" in ck and ck.get("has_ac"):
            ac.load_state_dict(ck["ac"])
        print("init from", args.init, "step", ck.get("step"), flush=True)
    wm.rssm.compile()
    twohot = TwoHot(device=device)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr, eps=1e-8)
    opt_actor = torch.optim.Adam(ac.actor.parameters(), lr=args.ac_lr, eps=1e-5)
    opt_critic = torch.optim.Adam(ac.critic.parameters(), lr=args.ac_lr, eps=1e-5)
    resume = os.path.join(run, "last.pt")
    best = float("inf")
    if args.resume and os.path.exists(resume):
        ck = torch.load(resume, map_location=device, weights_only=False)
        wm.load_state_dict(ck["wm"])
        ac.load_state_dict(ck["ac"])
        for o, k in ((opt_wm, "opt_wm"), (opt_actor, "opt_actor"), (opt_critic, "opt_critic")):
            if k in ck:
                o.load_state_dict(ck[k])
        start_step = ck["step"] + 1
        best = ck.get("best", best)
        print("resumed", resume, "at step", start_step, flush=True)
    q = prefetch(train, args.batch, args.length)
    cfg = vars(args) | {"params_wm": n_wm, "params_ac": n_ac, "img_ch": img_ch, "vec_dim": vec_dim,
                        "act_dim": act_dim, "train_frames": train.frames, "val_frames": val.frames}
    json.dump(cfg, open(os.path.join(run, "config.json"), "w"), indent=2)
    log = open(os.path.join(run, "log.jsonl"), "a")
    t0, acc = time.time(), {}

    def save(name, step, extra=None):
        state = {"wm": wm.state_dict(), "ac": ac.state_dict(), "has_ac": args.ac, "step": step, "config": cfg,
                 "best": best, **(extra or {})}
        if name == "last.pt":
            state |= {"opt_wm": opt_wm.state_dict(), "opt_actor": opt_actor.state_dict(),
                      "opt_critic": opt_critic.state_dict()}
        torch.save(state, os.path.join(run, name + ".tmp"))
        os.replace(os.path.join(run, name + ".tmp"), os.path.join(run, name))

    for step in range(start_step, args.steps + 1):
        batch = to_torch(q.get(), device)
        lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
        for g in opt_wm.param_groups:
            g["lr"] = args.lr * lr_scale
        with torch.autocast("cuda", torch.bfloat16):
            wm_loss, m, out, _ = wm.loss(batch, twohot)
        if not args.freeze_wm:
            opt_wm.zero_grad(set_to_none=True)
            wm_loss.backward()
            m["gnorm_wm"] = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000.0)
            opt_wm.step()
        if args.ac:
            start = {"deter": out["deter"].detach().flatten(0, 1), "stoch": out["stoch"].detach().flatten(0, 1)}
            start_cont = 1.0 - batch["is_terminal"].flatten(0, 1).float()
            with torch.autocast("cuda", torch.bfloat16):
                a_loss, c_loss, am = ac.loss(wm, start, start_cont, twohot)
            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            (a_loss + c_loss).backward()
            am["gnorm_actor"] = torch.nn.utils.clip_grad_norm_(ac.actor.parameters(), 100.0)
            am["gnorm_critic"] = torch.nn.utils.clip_grad_norm_(ac.critic.parameters(), 100.0)
            opt_actor.step()
            opt_critic.step()
            ac.update_slow()
            wm.zero_grad(set_to_none=True)  # the actor's REINFORCE loss does not train the world model
            m.update(am)
        for k, v in m.items():
            acc.setdefault(k, []).append(float(v.detach()) if torch.is_tensor(v) else float(v))
        if step % args.log_every == 0:
            row = {"step": step, "sps": args.log_every / (time.time() - t0) if step else 0.0,
                   **{k: float(np.mean(v)) for k, v in acc.items()}}
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            acc, t0 = {}, time.time()
        if step % args.eval_every == 0 and step:
            res = evaluate(wm, twohot, val, device)
            res["step"] = step
            print("EVAL", json.dumps(res), flush=True)
            log.write(json.dumps({"eval": res}) + "\n")
            log.flush()
            score = res["val_loss"]
            if score < best:
                best = score
                save("best.pt", step, {"eval": res})
            t0 = time.time()
        if step % args.save_every == 0 and step:
            save("last.pt", step)
    save("last.pt", args.steps)


if __name__ == "__main__":
    main()
