"""Re-score the zero-shot runs with a strict crash rule and unwrapped yaw (env: conda `sakana_hyperdrone`, GPU).

eval_zeroshot.py used diffwam/deploy.py's crash rule (central part of the forward view, 0.2 m sub-steps). The videos
show it misses obstacles below the image centre: OpenFly flew through a kitchen counter in s04 without a "crash".
Here every path (OpenFly: its actions from the start pose; DiffWAM: its 39 plan poses) is re-flown kinematically in
steps of at most 5 cm / 5 deg, and checked with dreamer/env.py's crash rig: depth in 4 horizontal directions (central
band < 0.25 m), straight up / down (< 0.15 m), the floor / ceiling, and leaving the house (forward view mostly empty).
The path ends at the first crash. Yaw is accumulated along the path, so 8 x 30 deg right is -240 deg, not +120 deg.

    python rescore.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("dreamer", "follow", "diffwam"):
    sys.path.append(os.path.join(HERE, "..", p))
from scenes import HouseCamera  # noqa: E402
from analyze_follow import FLOWN_OK  # noqa: E402
from eval_zeroshot import ACTION_NAMES, SAMPLES_EVAL, openfly_move  # noqa: E402

N = 64  # poses per render batch
IMG = 64


class Rig:
    def __init__(self, house):
        self.cam = HouseCamera(house, width=IMG, height=IMG, batch=N * 6)
        self.floor = float(self.cam.center[2] - self.cam.half[2])
        self.ceiling = float(self.cam.center[2] + self.cam.half[2])

    def crash(self, poses):
        """(n <= N, 4) -> bool per pose."""
        n = len(poses)
        rows = []
        for k in range(4):
            rows += [self.cam.renderer.camera((x, y, z), (x + math.cos(t + k * math.pi / 2), y + math.sin(t + k * math.pi / 2), z),
                                              fov=self.cam.fov_arg) for x, y, z, t in poses]
        for sign in (1, -1):
            rows += [self.cam.renderer.camera((x, y, z), (x + 1e-3 * math.cos(t), y + 1e-3 * math.sin(t), z + sign),
                                              fov=self.cam.fov_arg) for x, y, z, t in poses]
        rows = np.stack([np.asarray(r, np.float32) for r in rows]).reshape(len(rows), -1)
        rows = np.concatenate([rows, np.repeat(rows[-1:], N * 6 - len(rows), 0)]) if len(rows) < N * 6 else rows
        self.cam.renderer.set_cameras(rows)
        self.cam.renderer.render()
        d = np.nan_to_num(np.array(self.cam.renderer.outputs[self.cam._depth].read())[: 6 * n], nan=0.0, posinf=0.0)
        h = d[: 4 * n].reshape(4, n, IMG, IMG)
        band = h[:, :, IMG // 2 - IMG // 8: IMG // 2 + IMG // 8]
        dh = np.where(band > 0, band, 1e3).reshape(4, n, -1).min(-1).min(0)
        v = d[4 * n:].reshape(2, n, IMG, IMG)[:, :, IMG // 4: 3 * IMG // 4, IMG // 4: 3 * IMG // 4]
        dv = np.where(v > 0, v, 1e3).reshape(2, n, -1).min(-1).min(0)
        outside = (h[0] <= 0).mean((-1, -2)) > 0.5
        z = poses[:, 2]
        return (dh < 0.25) | (dv < 0.15) | (z < self.floor + 0.1) | (z > self.ceiling - 0.1) | outside


def densify(path, max_d=0.05, max_yaw=math.radians(5)):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, int(math.ceil(max(np.linalg.norm(b[:3] - a[:3]) / max_d, abs(b[3] - a[3]) / max_yaw))))
        out += [a + (b - a) * (j + 1) / k for j in range(k)]
    return np.array(out)


def fly(rig, path):
    """Dense path (unwrapped yaw); returns end pose (before the first crash) and crash flag."""
    for s in range(0, len(path), N):
        c = rig.crash(path[s:s + N])
        if c.any():
            i = s + int(np.argmax(c))
            return path[max(0, i - 1)], True
    return path[-1], False


def summarize(start, end):
    c, s = math.cos(start[3]), math.sin(start[3])
    d = end[:3] - start[:3]
    return {"forward_m": float(d[0] * c + d[1] * s), "left_m": float(-d[0] * s + d[1] * c), "up_m": float(d[2]),
            "yaw_deg": float(math.degrees(end[3] - start[3]))}  # unwrapped


def main():
    from deploy_combined import diffwam_plans
    from plans import rel_to_world

    res = json.load(open(os.path.join(HERE, "..", "dreamer", "data", "vla_zeroshot", "zeroshot.json")))
    scale = res["scale"]
    samples = {s["id"]: s for s in json.load(open(os.path.join(SAMPLES_EVAL, "index.json")))}
    dw = diffwam_plans(os.path.join(HERE, "..", "diffwam", "data", "head", "diffwam.pt"))
    rows, rig, house = [], None, None
    for r in sorted(res["runs"], key=lambda r: samples[r["id"]]["house"]):
        s = samples[r["id"]]
        if s["house"] != house:
            rig, house = Rig(s["house"]), s["house"]
        start = np.array(s["pose"], float)
        if r["arm"] == "diffwam":
            plan = rel_to_world(start, dw[s["id"]]["rel"]).astype(float)
            plan[:, 3] = start[3] + np.asarray(dw[s["id"]]["rel"])[:, 3]  # keep the plan's yaw unwrapped
            path = plan
        else:
            path, pose = [start], start.copy()
            for name in r["actions"]:
                a = ACTION_NAMES.index(name)
                if a == 0:
                    break
                pose = openfly_move(pose, a, scale)
                pose[3] = path[-1][3] + math.atan2(math.sin(pose[3] - path[-1][3]), math.cos(pose[3] - path[-1][3]))
                path.append(pose.copy())
            path = np.array(path)
        end, crashed = fly(rig, densify(path))
        summ = summarize(start, end)
        ok = bool(FLOWN_OK[s["motion"]](summ))
        rows.append({**{k: r[k] for k in ("id", "motion", "arm")}, **summ, "flown_ok": ok, "collision": crashed,
                     "success": ok and not crashed, "actions": r.get("actions"), "plan_seconds": r["plan_seconds"],
                     "decisions": r["decisions"], "old_flown_ok": r["flown_ok"], "old_collision": r["collision"]})
        print(json.dumps({k: rows[-1][k] for k in ("id", "arm", "flown_ok", "collision", "forward_m", "up_m", "yaw_deg")}),
              flush=True)
    summary = {}
    for arm in ("diffwam", "openfly-wan", "openfly-nav"):
        rs = [x for x in rows if x["arm"] == arm]
        summary[arm] = {"n": len(rs), "flown_ok": sum(x["flown_ok"] for x in rs),
                        "collisions": sum(x["collision"] for x in rs), "success": sum(x["success"] for x in rs),
                        "old_rule": {"flown_ok": sum(x["old_flown_ok"] for x in rs),
                                     "collisions": sum(x["old_collision"] for x in rs)},
                        "unique_cases": len({(x["id"][:3], json.dumps(x["actions"])) for x in rs}) if arm != "diffwam" else len(rs)}
    json.dump({"summary": summary, "runs": rows}, open(os.path.join(HERE, "..", "dreamer", "data", "vla_zeroshot",
                                                                    "zeroshot_strict.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
