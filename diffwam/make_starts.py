"""DiffWAM data, stage A (env `sakana_hyperdrone`): take-off poses + first frame I0 + metric depth in many houses.

Each sample gets a house, a take-off pose, an instruction template and a Wan seed. Houses listed in --exclude
(the 10 evaluation scenarios) are left out so the test houses are never trained on.

    python make_starts.py --houses 13 200 --per-house 3 --out-dir data/samples
Writes data/samples/<sample>/{I0.png, I0_depth.npy, sample.json} and data/samples/index.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "follow"))
from scenes import HouseCamera  # noqa: E402

# Generic camera-move instructions (v2 style: describe the camera motion; never mention a drone).
TEMPLATES = {
    "forward": "Fast forward dolly shot, first-person view, the camera is moving forward continuously through the room, strong forward camera motion, objects get closer and larger every second.",
    "slow_forward": "Slow forward dolly shot, first-person view, the camera glides forward gently and continuously through the room, objects slowly get closer.",
    "turn_left": "Camera pans to the left while moving forward, first-person view, the camera turns continuously to the left, strong left yaw rotation, the room sweeps from left to right across the frame.",
    "turn_right": "Camera pans to the right while moving forward, first-person view, the camera turns continuously to the right, strong right yaw rotation, the room sweeps from right to left across the frame.",
    "pan_left": "The camera rotates to the left in place, first-person view, a steady left pan without moving forward, the room sweeps from left to right across the frame.",
    "pan_right": "The camera rotates to the right in place, first-person view, a steady right pan without moving forward, the room sweeps from right to left across the frame.",
    "ascend": "Crane up shot, first-person view, the camera rises continuously upward while drifting a little forward, strong upward camera motion, the furniture moves down in the frame.",
    "descend": "Forward and downward camera motion, first-person view, the camera moves forward and descends continuously toward the floor, the floor rises in the frame.",
    "approach_stop": "Forward dolly shot, first-person view, the camera moves forward toward the furniture ahead, it gets closer and larger, then the camera slows down and stops, hovering still.",
}
EVAL_HOUSES = [1, 3, 4, 5, 6, 7, 8, 9, 10, 12]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--houses", type=int, nargs=2, default=[13, 200], help="Range [first, last] of house ids.")
    parser.add_argument("--max-houses", type=int, default=80)
    parser.add_argument("--per-house", type=int, default=3)
    parser.add_argument("--variants", type=int, default=2, help="(instruction, seed) draws per take-off pose.")
    parser.add_argument("--exclude", type=int, nargs="*", default=EVAL_HOUSES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/samples")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    houses = [h for h in range(args.houses[0], args.houses[1] + 1) if h not in args.exclude][: args.max_houses]
    names = list(TEMPLATES)
    index = []
    for house in houses:
        t0 = time.perf_counter()
        try:
            cam = HouseCamera(house)
            starts = cam.find_starts(np.random.default_rng(args.seed * 1000 + house), count=args.per_house, tries=192,
                                     min_ahead=2.5)
        except Exception as e:  # a house that fails to load or render is skipped, not fatal
            print(f"house {house}: skipped ({e})", flush=True)
            continue
        for k, start in enumerate(starts):
            rgb, depth = cam.render([start["pose"]])
            for v in range(args.variants):
                sid = f"h{house:03d}_p{k}_v{v}"
                d = os.path.join(args.out_dir, sid)
                os.makedirs(d, exist_ok=True)
                Image.fromarray(rgb[0]).save(os.path.join(d, "I0.png"))
                np.save(os.path.join(d, "I0_depth.npy"), depth[0].astype(np.float16))
                motion = names[rng.integers(len(names))]
                sample = {"id": sid, "house": house, "pose": start["pose"], "motion": motion,
                          "instruction": TEMPLATES[motion], "seed": int(rng.integers(1 << 30)),
                          "split": "train", "hfov_deg": cam.hfov_deg, "width": cam.width, "height": cam.height}
                json.dump(sample, open(os.path.join(d, "sample.json"), "w"), indent=2)
                index.append(sample)
        print(f"house {house}: {len(starts)} starts ({time.perf_counter() - t0:.1f}s), {len(index)} samples", flush=True)
        json.dump(index, open(os.path.join(args.out_dir, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
