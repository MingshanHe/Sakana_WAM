"""DiffWAM-style trajectory readout: Wan2.2 DiT features -> 38 future poses (env `sakana_hyperdrone`).

Inputs per sample (from wan_worker.py, PCA-reduced to 128 channels):
  Z  (K=4 steps, L=3 layers, 39 slots, 15, 26, 128), A0 (K, L, 1, 15, 26, 128), and the I0 depth for the scale s0.
Target: the gen2pose teacher trajectory of the completed rollout (label_teacher.py), slots 1..38 (slot 0 = identity).

Model (compact version of DiffWAM Sec. 3.2-3.3):
  per-layer LayerNorm + Linear, softmax layer weights (Eq. 7) -> step embedding -> per slot, attention pooling of the
  K x 5 x 8 tokens into 8 memory tokens (memory 39 x 8 = 312, as in the paper) -> transformer encoder over memory +
  A0 tokens + depth tokens -> 38 pose queries in a transformer decoder -> position * s0 (Eq. 14-15), yaw (sin, cos).
Baselines: hover (identity) and instruction prior (mean training trajectory of the same instruction type).

    python train_head.py --train data/samples --test data/samples_eval --evals 0 1 2 3   # DiffWAM
    python train_head.py ... --evals 0                                                   # DiffWAM-Flash
"""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = (5, 8)


def load_split(root: str, evals) -> list[dict]:
    """Samples with features + teacher; features are spatially pooled to 5 x 8 and cached next to feats.pt."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not (os.path.exists(os.path.join(d, "teacher.npz")) and os.path.exists(os.path.join(d, "feats.pt"))):
            continue
        cache = os.path.join(d, "feats_pooled.pt")
        if not os.path.exists(cache):
            f = torch.load(os.path.join(d, "feats.pt"))
            pool = lambda x: F.adaptive_avg_pool2d(x.float().flatten(0, 2).permute(0, 3, 1, 2), POOL)  # noqa: E731
            z = pool(f["Z"]).reshape(*f["Z"].shape[:3], 128, *POOL).permute(0, 1, 2, 4, 5, 3)
            a0 = pool(f["A0"]).reshape(*f["A0"].shape[:3], 128, *POOL).permute(0, 1, 2, 4, 5, 3)
            torch.save({"Z": z.half(), "A0": a0.half()}, cache)
        f = torch.load(cache)
        sample = json.load(open(os.path.join(d, "sample.json")))
        teacher = np.load(os.path.join(d, "teacher.npz"))
        depth = np.load(os.path.join(d, "I0_depth.npy")).astype(np.float32)
        valid = depth[(depth > 0.3) & (depth < 30)]
        s0 = float(np.clip(np.median(valid), 0.5, 15)) if len(valid) else 1.0
        h, w = depth.shape
        regions = [depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3], depth[:, : w // 3], depth[:, 2 * w // 3:],
                   depth[: h // 3], depth[2 * h // 3:]]  # centre, left, right, top, bottom
        geo = np.array([np.median(r[r > 0]) if (r > 0).any() else 0.0 for r in regions], np.float32) / s0
        pose = teacher["pose"].astype(np.float32)
        speed = np.linalg.norm(np.diff(pose[:, :3], axis=0), axis=1) / (5.0 / 38.0)
        # A teacher that lost track drifts on its constant-velocity fallback: indoors in 5 s nothing flies > 7 m.
        plausible = bool(speed.max() < 2.5 and np.linalg.norm(pose[-1, :3]) < 7.0)
        out.append({"id": sample["id"], "motion": sample["motion"], "house": sample["house"],
                    "Z": f["Z"][list(evals)], "A0": f["A0"][list(evals)], "s0": s0, "geo": geo,
                    "pose": pose, "ok": float(teacher["ok"].mean()), "plausible": plausible})
    return out


class ReadoutHead(nn.Module):
    def __init__(self, k: int, d: int = 128, mem_per_slot: int = 8, enc_layers: int = 2, dec_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(128), nn.Linear(128, d)) for _ in range(3)])
        self.layer_logits = nn.Parameter(torch.zeros(3))
        self.step_emb = nn.Parameter(torch.zeros(k, 1, 1, d))
        self.space_emb = nn.Parameter(torch.randn(POOL[0] * POOL[1], d) * 0.02)
        self.pool_q = nn.Parameter(torch.randn(mem_per_slot, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, 4, dropout=dropout, batch_first=True)
        self.slot_emb = nn.Parameter(torch.randn(39, 1, d) * 0.02)
        self.a0_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.geo = nn.Linear(5, d)
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.queries = nn.Parameter(torch.randn(38, d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, 4, 4 * d, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.out = nn.Linear(d, 5)  # dx, dy, dz (in units of s0), sin yaw, cos yaw

    def fuse_layers(self, x):  # (B, K, 3, T, h, w, 128) -> (B, K, T, h*w, d)
        w = torch.softmax(self.layer_logits, 0)
        y = sum(w[l] * self.layer_proj[l](x[:, :, l].float()) for l in range(3))
        return y.flatten(3, 4) + self.space_emb

    def forward(self, z, a0, geo):
        b, k, _, t = z.shape[:4]
        f = self.fuse_layers(z) + self.step_emb[None]  # (B, K, T, S, d)
        tokens = f.permute(0, 2, 1, 3, 4).reshape(b * t, k * f.shape[3], -1)  # per slot: K x S tokens
        q = self.pool_q[None].expand(b * t, -1, -1)
        mem, _ = self.pool(q, tokens, tokens)  # (B*T, 8, d)
        mem = mem.reshape(b, t, -1, mem.shape[-1]) + self.slot_emb[None]
        mem = mem.flatten(1, 2)  # (B, 312, d)
        anchor = self.fuse_layers(a0)[:, 0, 0] + self.a0_emb  # step 0 anchor tokens (B, S, d)
        mem = self.encoder(torch.cat([mem, anchor, self.geo(geo)[:, None]], 1))
        h = self.decoder(self.queries[None].expand(b, -1, -1), mem)
        return self.out(h)  # (B, 38, 5)


def to_targets(pose, s0):
    """(B, 39, 4) teacher -> (B, 38, 5) normalised targets."""
    p = pose[:, 1:]
    return torch.cat([p[..., :3] / s0[:, None, None], torch.sin(p[..., 3:]), torch.cos(p[..., 3:])], -1)


def to_poses(out, s0):
    """(B, 38, 5) -> (B, 39, 4) x, y, z, yaw with slot 0 = identity."""
    pos = out[..., :3] * s0[:, None, None]
    yaw = torch.atan2(out[..., 3], out[..., 4])
    p = torch.cat([pos, yaw[..., None]], -1)
    return torch.cat([torch.zeros_like(p[:, :1]), p], 1)


def metrics(pred, true) -> dict:
    err = np.linalg.norm(pred[:, :, :3] - true[:, :, :3], axis=-1)
    yaw = np.degrees(np.abs(np.angle(np.exp(1j * (pred[:, -1, 3] - true[:, -1, 3])))))
    return {"ade_m": float(err.mean()), "fde_m": float(err[:, -1].mean()), "yaw_final_deg": float(yaw.mean()),
            "n": int(len(pred))}


def batchify(samples, device="cuda"):
    z = torch.stack([s["Z"] for s in samples]).to(device)
    a0 = torch.stack([s["A0"] for s in samples]).to(device)
    geo = torch.tensor(np.stack([s["geo"] for s in samples]), device=device)
    s0 = torch.tensor([s["s0"] for s in samples], device=device)
    pose = torch.tensor(np.stack([s["pose"] for s in samples]), device=device)
    return z, a0, geo, s0, pose


def evaluate(model, samples, prior):
    if not samples:
        nan = {"ade_m": float("nan"), "fde_m": float("nan"), "yaw_final_deg": float("nan"), "n": 0}
        return {"model": nan, "hover": nan, "instruction_prior": nan}, np.zeros((0, 39, 4))
    model.eval()
    with torch.no_grad():
        z, a0, geo, s0, pose = batchify(samples)
        pred = to_poses(model(z, a0, geo), s0).cpu().numpy()
    model.train()
    true = pose.cpu().numpy()
    hover = np.zeros_like(true)
    pri = np.stack([prior.get(s["motion"], np.zeros_like(true[0])) for s in samples])
    return {"model": metrics(pred, true), "hover": metrics(hover, true), "instruction_prior": metrics(pri, true)}, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", default=os.path.join(HERE, "data", "samples"))
    parser.add_argument("--test", default=os.path.join(HERE, "data", "samples_eval"))
    parser.add_argument("--evals", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of training houses held out.")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-pnp-ok", type=float, default=0.7, help="Drop samples whose teacher mostly failed.")
    parser.add_argument("--zero-future", action="store_true",
                        help="Control: blank the 39 future-slot features (keep A0 + I0 depth) -> scene-only readout.")
    parser.add_argument("--out", default=os.path.join(HERE, "data", "head"))
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    os.makedirs(args.out, exist_ok=True)

    all_data = load_split(args.train, args.evals)
    data = [s for s in all_data if s["ok"] >= args.min_pnp_ok and s["plausible"]]
    all_test = load_split(args.test, args.evals)
    test = [s for s in all_test if s["ok"] >= args.min_pnp_ok and s["plausible"]]
    print(f"teacher filter kept {len(data)}/{len(all_data)} training and {len(test)}/{len(all_test)} test samples",
          flush=True)
    if args.zero_future:
        for s in data + test:
            s["Z"] = torch.zeros_like(s["Z"])
    houses = sorted({s["house"] for s in data})
    rng = np.random.default_rng(0)
    val_houses = set(rng.choice(houses, max(1, int(len(houses) * args.val_frac)), replace=False).tolist())
    train = [s for s in data if s["house"] not in val_houses]
    val = [s for s in data if s["house"] in val_houses]
    print(f"train {len(train)} samples / {len(houses) - len(val_houses)} houses, val {len(val)} / {len(val_houses)}, "
          f"test {len(test)} (evaluation houses)", flush=True)
    prior = {m: np.mean([s["pose"] for s in train if s["motion"] == m], 0)
             for m in {s["motion"] for s in train}}
    # The test instructions are scenario-specific; map them to the nearest training template for the prior.
    alias = {"hallway": "forward", "doorway": "forward", "approach": "approach_stop", "approach+stop": "approach_stop",
             "turn left": "turn_left", "turn right": "turn_right", "strafe right": "forward", "ascend": "ascend",
             "descend": "descend"}
    for m, a in alias.items():
        if a in prior:
            prior.setdefault(m, prior[a])

    model = ReadoutHead(k=len(args.evals)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    best, best_state, t0 = float("inf"), None, time.perf_counter()
    for step in range(args.steps):
        idx = rng.choice(len(train), args.batch)
        z, a0, geo, s0, pose = batchify([train[i] for i in idx])
        out = model(z, a0, geo)
        tgt = to_targets(pose, s0)
        loss = F.smooth_l1_loss(out[..., :3], tgt[..., :3], beta=0.05) + 0.1 * F.mse_loss(out[..., 3:], tgt[..., 3:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps - 1:
            m, _ = evaluate(model, val, prior)
            if not m["model"]["ade_m"] >= best:  # also true for NaN (no validation set): keep the latest
                best, best_state = m["model"]["ade_m"], {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"step {step} loss {loss.item():.4f} val ADE {m['model']['ade_m']:.3f} FDE {m['model']['fde_m']:.3f} "
                  f"(prior {m['instruction_prior']['ade_m']:.3f}, hover {m['hover']['ade_m']:.3f})", flush=True)
    model.load_state_dict(best_state)
    val_m, _ = evaluate(model, val, prior)
    test_m, test_pred = evaluate(model, test, prior)
    tag = "diffwam" if len(args.evals) == 4 else ("flash" if args.evals == [0] else "evals" + "".join(map(str, args.evals)))
    if args.zero_future:
        tag += "_scene_only"
    result = {"tag": tag, "evals": args.evals, "train": len(train), "val": val_m, "test": test_m,
              "train_seconds": time.perf_counter() - t0, "params": sum(p.numel() for p in model.parameters()),
              "layer_weights": torch.softmax(model.layer_logits, 0).tolist(),
              "test_per_sample": [{"id": s["id"], "motion": s["motion"], "pred_final": p[-1].tolist(),
                                   "teacher_final": s["pose"][-1].tolist()} for s, p in zip(test, test_pred)]}
    torch.save({"model": model.state_dict(), "evals": args.evals, "prior": prior}, os.path.join(args.out, f"{tag}.pt"))
    np.savez(os.path.join(args.out, f"{tag}_test_pred.npz"), pred=test_pred, ids=[s["id"] for s in test])
    json.dump(result, open(os.path.join(args.out, f"{tag}_metrics.json"), "w"), indent=2)
    for split in ("val", "test"):
        print(split, json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in result[split].items()}))


if __name__ == "__main__":
    main()
