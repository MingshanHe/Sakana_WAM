"""Closed loop in HyperDrone (env `sakana_hyperdrone`): fly the trajectories predicted from Wan2.2 features.

For every test sample (evaluation houses, never trained on):
  fast path  I0 -> DiffWAM features (wan_worker.py) -> readout head (train_head.py) -> 38 poses -> fly
  slow path  I0 -> completed rollout -> gen2pose teacher -> 38 poses -> fly (upper bound, as DiffWAM's teacher)
Flying: the drone tracks the slot poses with a kinematic tracker (speed <= 1.5 m/s, yaw rate <= 1.2 rad/s, 5
sub-steps per slot); the onboard camera is rendered at every slot, and a collision is anything closer than 0.25 m in
the central view at any sub-step. Motion vs instruction uses the rules of ../follow/analyze_follow.py.

    python deploy.py --head data/head/diffwam.pt
Writes data/deploy/<tag>/<id>_{fast,slow}.mp4 (imagined rollout | onboard flight) and data/deploy/<tag>/deploy.json.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "follow"))
from analyze_follow import FLOWN_OK  # noqa: E402
from scenes import HouseCamera  # noqa: E402

from train_head import ReadoutHead, batchify, load_split, to_poses  # noqa: E402

MAX_V, MAX_W, SUB = 1.5, 1.2, 5
SLOT_DT = 5.0 / 38.0


def start_to_world(start, rel):
    """(T, 4) poses in the start FLU frame -> world x, y, z, yaw."""
    c, s = math.cos(start[3]), math.sin(start[3])
    x = start[0] + c * rel[:, 0] - s * rel[:, 1]
    y = start[1] + s * rel[:, 0] + c * rel[:, 1]
    return np.stack([x, y, start[2] + rel[:, 2], start[3] + rel[:, 3]], 1)


def fly(cam, start, rel_path):
    """Track the (39, 4) relative path; returns flown world poses (39, 4), onboard frames, collision flag."""
    target = start_to_world(start, rel_path)
    pose, flown, frames, crashed = target[0].copy(), [target[0].copy()], [], False
    for k in range(1, len(target)):
        sub = []
        for _ in range(SUB):
            d = target[k, :3] - pose[:3]
            step = d / max(1.0, np.linalg.norm(d) / (MAX_V * SLOT_DT / SUB))
            dyaw = math.atan2(math.sin(target[k, 3] - pose[3]), math.cos(target[k, 3] - pose[3]))
            pose = pose + np.r_[step, np.clip(dyaw, -MAX_W * SLOT_DT / SUB, MAX_W * SLOT_DT / SUB)]
            sub.append(pose.copy())
        rgb, depth = cam.render(np.array(sub))
        h, w = depth.shape[1:]
        centre = np.where(depth > 0, depth, 1e3)[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        crashed |= bool(centre.min() < 0.25)
        frames.append(rgb[-1])
        flown.append(pose.copy())
    first, _ = cam.render([target[0]])
    return np.array(flown), [first[0]] + frames, crashed


def summarize(start, flown):
    c, s = math.cos(start[3]), math.sin(start[3])
    d = flown[-1, :3] - start[:3]
    return {"forward_m": float(d[0] * c + d[1] * s), "left_m": float(-d[0] * s + d[1] * c), "up_m": float(d[2]),
            "yaw_deg": float(math.degrees(flown[-1, 3] - start[3]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", default=os.path.join(HERE, "data", "head", "diffwam.pt"))
    parser.add_argument("--test", default=os.path.join(HERE, "data", "samples_eval"))
    parser.add_argument("--limit", type=int, default=0, help="Only the first N test samples (debugging).")
    args = parser.parse_args()
    import imageio.v2 as imageio

    ckpt = torch.load(args.head, weights_only=False)
    tag = os.path.splitext(os.path.basename(args.head))[0]
    out_dir = os.path.join(HERE, "data", "deploy", tag)
    os.makedirs(out_dir, exist_ok=True)
    model = ReadoutHead(k=len(ckpt["evals"])).cuda().eval()
    model.load_state_dict(ckpt["model"])
    samples = load_split(args.test, ckpt["evals"])
    if args.limit:
        samples = samples[: args.limit]
    results, cams = [], {}
    for s in samples:
        meta = json.load(open(os.path.join(args.test, s["id"], "sample.json")))
        start = np.array(meta["pose"], float)
        feats = torch.load(os.path.join(args.test, s["id"], "feats.pt"))
        z, a0, geo, s0, _ = batchify([s])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            pred = to_poses(model(z, a0, geo), s0)[0].cpu().numpy()
        torch.cuda.synchronize()
        head_s = time.perf_counter() - t0
        if meta["house"] not in cams:
            cams = {meta["house"]: HouseCamera(meta["house"])}  # one house in memory at a time
        cam = cams[meta["house"]]
        imagined = np.load(os.path.join(args.test, s["id"], "slots.npz"))["frames"]
        row = {"id": s["id"], "motion": s["motion"], "feature_s": feats["seconds"], "head_s": head_s,
               "teacher_final": s["pose"][-1].tolist(), "pred_final": pred[-1].tolist(),
               "fde_m": float(np.linalg.norm(pred[-1, :3] - s["pose"][-1, :3])),
               "teacher_trusted": bool(s["ok"] >= 0.7 and s["plausible"])}
        for name, rel in (("fast", pred), ("slow", s["pose"])):
            flown, frames, crashed = fly(cam, start, rel)
            summ = summarize(start, flown)
            summ["flown_ok"] = bool(FLOWN_OK[s["motion"]](summ))
            summ["collision"] = crashed
            row[name] = summ
            side = np.concatenate([imagined, np.stack(frames)], axis=2)[:, ::2, ::2]
            imageio.mimwrite(os.path.join(out_dir, f"{s['id']}_{name}.mp4"), side, fps=round(1 / SLOT_DT),
                             macro_block_size=1)
        results.append(row)
        print(json.dumps({k: row[k] for k in ("id", "motion", "fde_m")} | {"fast": row["fast"], "slow": row["slow"]}),
              flush=True)
    agg = {name: {"flown_ok": sum(r[name]["flown_ok"] for r in results), "collisions": sum(r[name]["collision"] for r in results),
                  "n": len(results)} for name in ("fast", "slow")}
    trusted = [r for r in results if r["teacher_trusted"]]
    agg["teacher_trusted"] = len(trusted)
    agg["mean_fde_m_trusted"] = float(np.mean([r["fde_m"] for r in trusted])) if trusted else float("nan")
    agg["mean_feature_s"] = float(np.mean([r["feature_s"] for r in results]))
    agg["mean_head_ms"] = 1000 * float(np.mean([r["head_s"] for r in results]))
    json.dump({"summary": agg, "samples": results}, open(os.path.join(out_dir, "deploy.json"), "w"), indent=2)
    print(json.dumps(agg))


if __name__ == "__main__":
    main()
