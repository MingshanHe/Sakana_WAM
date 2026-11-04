"""Zero-shot comparison on the 20 DiffWAM test cases (env: conda `sakana_hyperdrone`): which global planner flies the
instruction correctly, and how fast?

  diffwam            Wan2.2 features (steps 0-3) -> DiffWAM readout head -> 38 poses, flown kinematically (as in
                     diffwam/deploy.py)
  openfly-wan        OpenFly-Agent v1 (7B), given the same Wan camera-move instruction
  openfly-nav        OpenFly-Agent v1, given a short navigation-style instruction (closer to its training data)

OpenFly runs its own closed loop, as in its train/eval.py: each step it sees the last 3 onboard frames (832x480,
resized to 224 by its processor; current frame first and the Llama-2 chat prompt, as in its training data) and picks one of 10 discrete actions, which teleports the camera. Its steps are
sized for outdoor scenes (3 m), so translations are scaled by `--scale` (default 1/6: forward 0.5 / 1.0 / 1.5 m,
up / down / sideways 0.5 m); turns stay 30 deg. The episode ends at "stop" or after `--max-steps` actions.
Every arm uses the same crash rule as diffwam/deploy.py (anything closer than 0.25 m in the central view at any of 5
sub-steps per move) and the same motion rules (follow/analyze_follow.py).

    python openfly_server.py --root ../dreamer/data/vla_requests &      # env sakana_vla
    python eval_zeroshot.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("dreamer", "follow", "diffwam"):
    sys.path.append(os.path.join(HERE, "..", p))
from scenes import HouseCamera  # noqa: E402
from analyze_follow import FLOWN_OK  # noqa: E402

SAMPLES_EVAL = os.path.join(HERE, "..", "diffwam", "data", "samples_eval")
NAV = {  # short navigation-style instructions, from the scenarios' v1 wording without the video-model tail
    "s01": "Fly straight forward across the living room toward the far wall with the window and pictures, passing to the right of the dining table.",
    "s02": "Fly straight down the hallway along the teal rug, past the dark dresser on the right, toward the end of the corridor.",
    "s03": "Fly forward through the open doorway into the next room, then stop in the middle of that room.",
    "s04": "Move forward slowly while turning to the left, across the dining table toward the left wall.",
    "s05": "Fly up toward the ceiling of the bedroom.",
    "s06": "Fly forward through the doorway into the bedroom toward the bed, then stop above the foot of the bed.",
    "s07": "Move forward while turning to the right, toward the round table with the black chairs and the plant.",
    "s08": "Move sideways to the right across the room while facing straight ahead.",
    "s09": "Fly forward toward the wooden desk with the chair and stop just in front of the desk.",
    "s10": "Move forward toward the white table and descend to table height, then stop.",
}
ACTION_NAMES = ["stop", "fwd", "left30", "right30", "up", "down", "move_left", "move_right", "fwd x2", "fwd x3"]


def openfly_move(pose, a, scale):
    """OpenFly's getPoseAfterMakeAction with translations scaled by `scale` (3 m -> 3 * scale)."""
    x, y, z, yaw = pose
    s = 3.0 * scale
    fwd = {1: 1, 8: 2, 9: 3}.get(a, 0)
    x += fwd * s * math.cos(yaw)
    y += fwd * s * math.sin(yaw)
    if a == 2:
        yaw += math.radians(30)
    elif a == 3:
        yaw -= math.radians(30)
    elif a == 4:
        z += s
    elif a == 5:
        z -= s
    elif a == 6:
        x -= s * math.sin(yaw)
        y += s * math.cos(yaw)
    elif a == 7:
        x += s * math.sin(yaw)
        y -= s * math.cos(yaw)
    return np.array([x, y, z, yaw])


def collides(cam, a, b, sub=5):
    """Kinematic move a -> b in `sub` sub-steps; crash if the central view has anything closer than 0.25 m."""
    poses = np.array([a + (b - a) * (k + 1) / sub for k in range(sub)])
    _, depth = cam.render(poses)
    h, w = depth.shape[1:]
    centre = np.where(depth > 0, depth, 1e3)[:, h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return bool(centre.min() < 0.25)


def summarize(start, end):
    c, s = math.cos(start[3]), math.sin(start[3])
    d = end[:3] - start[:3]
    return {"forward_m": float(d[0] * c + d[1] * s), "left_m": float(-d[0] * s + d[1] * c), "up_m": float(d[2]),
            "yaw_deg": float(math.degrees(math.atan2(math.sin(end[3] - start[3]), math.cos(end[3] - start[3]))))}


def ask_openfly(root, req_id, frames, text, timeout=600):
    from PIL import Image

    d = os.path.join(root, req_id)
    os.makedirs(d, exist_ok=True)
    names = []
    for k, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(d, f"f{k}.png"))
        names.append(f"f{k}.png")
    json.dump({"text": text, "frames": names}, open(os.path.join(d, "request.tmp"), "w"))
    os.replace(os.path.join(d, "request.tmp"), os.path.join(d, "request.json"))
    path = os.path.join(d, "action.json")
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(d)
        time.sleep(0.02)
    for _ in range(50):
        try:
            return json.load(open(path))
        except json.JSONDecodeError:
            time.sleep(0.05)
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "..", "dreamer", "data", "vla_requests"))
    ap.add_argument("--scale", type=float, default=1 / 6)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "dreamer", "data", "vla_zeroshot"))
    ap.add_argument("--head", default=os.path.join(HERE, "..", "diffwam", "data", "head", "diffwam.pt"))
    args = ap.parse_args()
    import imageio.v2 as imageio

    os.makedirs(args.out, exist_ok=True)
    while not os.path.exists(os.path.join(args.root, "READY")):
        print("waiting for openfly_server.py ...", flush=True)
        time.sleep(10)
    from deploy_combined import diffwam_plans
    from plans import rel_to_world

    dw = diffwam_plans(args.head)
    samples = json.load(open(os.path.join(SAMPLES_EVAL, "index.json")))
    feats_s = {}
    for s in samples:
        import torch

        feats_s[s["id"]] = float(torch.load(os.path.join(SAMPLES_EVAL, s["id"], "feats.pt"))["seconds"])
    rows, cam_house, cam = [], None, None
    for s in samples:
        if s["house"] != cam_house:
            cam, cam_house = HouseCamera(s["house"], batch=8), s["house"]
        start = np.array(s["pose"], float)
        # DiffWAM plan, flown kinematically along its 39 slots
        plan = rel_to_world(start, dw[s["id"]]["rel"]).astype(float)
        crashed, end = False, plan[-1]
        for k in range(1, len(plan)):
            if collides(cam, plan[k - 1], plan[k], sub=2):
                crashed, end = True, plan[k - 1]
                break
        summ = summarize(start, end)
        rows.append({"id": s["id"], "motion": s["motion"], "arm": "diffwam", **summ,
                     "flown_ok": bool(FLOWN_OK[s["motion"]](summ)), "collision": crashed,
                     "plan_seconds": feats_s[s["id"]], "decisions": 1})
        # OpenFly, two instruction styles
        for arm, text in (("openfly-wan", s["instruction"]), ("openfly-nav", NAV[s["id"][:3]])):
            pose = start.copy()
            first, _ = cam.render(pose[None])
            frames = [first[0]]
            acts, secs, crashed = [], [], False
            for step in range(args.max_steps):
                hist = [frames[max(0, len(frames) - 1 - k)] for k in range(3)]  # current, previous, before (padded)
                out = ask_openfly(args.root, f"{s['id']}_{arm}_{step}", hist, text)
                acts.append(out["id"])
                secs.append(out["seconds"])
                if out["id"] == 0:
                    break
                new = openfly_move(pose, out["id"], args.scale)
                if collides(cam, pose, new):
                    crashed = True
                    break
                pose = new
                rgb, _ = cam.render(pose[None])
                frames.append(rgb[0])
            summ = summarize(start, pose)
            rows.append({"id": s["id"], "motion": s["motion"], "arm": arm, **summ,
                         "flown_ok": bool(FLOWN_OK[s["motion"]](summ)), "collision": crashed,
                         "actions": [ACTION_NAMES[a] for a in acts], "plan_seconds": float(np.sum(secs)),
                         "step_seconds": float(np.mean(secs)), "decisions": len(acts)})
            imageio.mimwrite(os.path.join(args.out, f"{s['id']}_{arm}.mp4"),
                             [f[::2, ::2] for f in frames for _ in range(5)], fps=10, macro_block_size=1)
        for r in rows[-3:]:
            print(json.dumps({k: r[k] for k in ("id", "arm", "flown_ok", "collision", "forward_m", "up_m", "yaw_deg")}
                             | {"actions": r.get("actions", "")}), flush=True)
    summary = {}
    for arm in ("diffwam", "openfly-wan", "openfly-nav"):
        rs = [r for r in rows if r["arm"] == arm]
        summary[arm] = {"n": len(rs), "flown_ok": sum(r["flown_ok"] for r in rs),
                        "collisions": sum(r["collision"] for r in rs),
                        "success": sum(r["flown_ok"] and not r["collision"] for r in rs),
                        "seconds_per_episode": float(np.mean([r["plan_seconds"] for r in rs])),
                        "seconds_per_decision": float(np.mean([r.get("step_seconds", r["plan_seconds"]) for r in rs])),
                        "decisions": float(np.mean([r["decisions"] for r in rs]))}
    json.dump({"summary": summary, "runs": rows, "scale": args.scale, "max_steps": args.max_steps},
              open(os.path.join(args.out, "zeroshot.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
