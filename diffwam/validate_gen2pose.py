"""Validate gen2pose on HyperDrone renders with ground-truth poses (env `sakana_hyperdrone`).

For each evaluation scenario and several scripted motions, render the 39 slot frames (832 x 480, 90 deg HFOV),
run gen2pose with the true first-frame depth, and compare with the true trajectory in the start frame.
Writes data/gen2pose_validation.json and prints ATE / final / yaw errors per motion.
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "follow"))
from follow import SLOT_DT, step_pose  # noqa: E402
from scenes import HouseCamera  # noqa: E402

from gen2pose import gen2pose, intrinsics  # noqa: E402

MOTIONS = {  # body-yaw-frame velocity (vx, vy, vz, wz)
    "forward": (0.6, 0.0, 0.0, 0.0),
    "forward_left": (0.4, 0.0, 0.0, 0.3),
    "forward_right": (0.4, 0.0, 0.0, -0.3),
    "pan_right": (0.0, 0.0, 0.0, -0.4),
    "ascend": (0.15, 0.0, 0.15, 0.0),
    "descend": (0.3, 0.0, -0.12, 0.0),
    "strafe_right": (0.0, -0.3, 0.0, 0.0),
}


def to_start_frame(path):
    p0, yaw0 = path[0, :3], path[0, 3]
    c, s = math.cos(yaw0), math.sin(yaw0)
    d = path[:, :3] - p0
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2], path[:, 3] - yaw0], 1)


def main():
    meta = json.load(open(os.path.join(HERE, "..", "follow", "scenarios", "scenarios.json")))["scenarios"]
    K = intrinsics(832, 480, 90.0)
    results = []
    for sc in meta:
        cam = HouseCamera(sc["house"])
        start = np.array(sc["start"]["pose"], float)
        for name, cmd in MOTIONS.items():
            path = [start]
            for _ in range(38):
                path.append(step_pose(path[-1][None], np.array([cmd]), SLOT_DT)[0])
            path = np.array(path)
            rgb, depth = cam.render(path)
            est = gen2pose(rgb, depth[0], K)
            gt = to_start_frame(path)
            err = np.linalg.norm(est["pose"][:, :3] - gt[:, :3], axis=1)
            yaw_err = np.degrees(np.abs(np.angle(np.exp(1j * (est["pose"][:, 3] - gt[:, 3])))))
            r = {"scenario": sc["id"], "motion": name, "ate_m": float(np.sqrt((err ** 2).mean())),
                 "final_m": float(err[-1]), "path_m": float(np.linalg.norm(np.diff(gt[:, :3], axis=0), axis=1).sum()),
                 "yaw_final_deg": float(yaw_err[-1]), "pnp_ok": float(est["ok"].mean()),
                 "min_map": int(est["n_map"].min())}
            results.append(r)
            print(json.dumps(r), flush=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    json.dump(results, open(os.path.join(HERE, "data", "gen2pose_validation.json"), "w"), indent=2)
    print("\nmotion          ATE m  final m  path m  yaw final deg  PnP ok")
    for name in MOTIONS:
        rs = [r for r in results if r["motion"] == name]
        print(f"{name:14s} {np.mean([r['ate_m'] for r in rs]):6.3f} {np.mean([r['final_m'] for r in rs]):8.3f} "
              f"{np.mean([r['path_m'] for r in rs]):7.2f} {np.mean([r['yaw_final_deg'] for r in rs]):13.2f} "
              f"{np.mean([r['pnp_ok'] for r in rs]):7.2f}")


if __name__ == "__main__":
    main()
