"""DiffWAM data, stage C (env `sakana_hyperdrone`, CPU): gen2pose teacher poses for every completed rollout.

Reads <sample>/slots.npz (39 rollout frames) + I0_depth.npy and writes <sample>/teacher.npz:
  pose (39, 4)  x, y, z [m], yaw [rad] in the drone's start FLU frame; c2s (39, 4, 4); ok (39,); n_map (39,)
Polls for new samples until none appear for --idle-exit seconds.

    python label_teacher.py --samples data/samples
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from gen2pose import gen2pose, intrinsics

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", default=os.path.join(HERE, "data", "samples"))
    parser.add_argument("--idle-exit", type=float, default=900)
    args = parser.parse_args()
    K = intrinsics(832, 480, 90.0)
    last = time.time()
    while True:
        todo = [os.path.dirname(p) for p in sorted(glob.glob(os.path.join(args.samples, "*", "slots.npz")))
                if not os.path.exists(os.path.join(os.path.dirname(p), "teacher.npz"))]
        if not todo:
            if time.time() - last > args.idle_exit:
                break
            time.sleep(30)
            continue
        for d in todo:
            try:
                frames = np.load(os.path.join(d, "slots.npz"))["frames"]
            except Exception:  # still being written
                continue
            depth0 = np.load(os.path.join(d, "I0_depth.npy")).astype(np.float32)
            est = gen2pose(frames, depth0, K)
            np.savez(os.path.join(d, "teacher.npz"), **est)
            p = est["pose"][-1]
            print(json.dumps({"id": os.path.basename(d), "final": [round(float(v), 2) for v in p[:3]],
                              "yaw_deg": round(float(np.degrees(p[3])), 1), "pnp_ok": round(float(est["ok"].mean()), 2)}),
                  flush=True)
            last = time.time()


if __name__ == "__main__":
    main()
