"""Pick take-off poses in ProcTHOR houses and render the first onboard frame I0 of each (env `sakana_hyperdrone`).

    python make_scenarios.py --houses 1 2 3 ... --out-dir scenarios_raw

Writes <out-dir>/house_<k>/{I0.png, I0_depth.npy, start.json} and <out-dir>/candidates.png (all I0s, labelled).
Instructions are attached afterwards in scenarios.json.
"""

import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from scenes import HouseCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--houses", type=int, nargs="+", default=list(range(1, 15)))
    parser.add_argument("--height", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="scenarios_raw")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tiles = []
    for house in args.houses:
        t0 = time.perf_counter()
        cam = HouseCamera(house)
        starts = cam.find_starts(np.random.default_rng(args.seed + house), count=1, height=args.height)
        if not starts:
            print(f"house {house}: no open start found", flush=True)
            continue
        start = starts[0]
        rgb, depth = cam.render([start["pose"]])
        out = os.path.join(args.out_dir, f"house_{house}")
        os.makedirs(out, exist_ok=True)
        Image.fromarray(rgb[0]).save(os.path.join(out, "I0.png"))
        np.save(os.path.join(out, "I0_depth.npy"), depth[0].astype(np.float16))
        start.update(width=cam.width, height=cam.height, hfov_deg=cam.hfov_deg, fov_arg=cam.fov_arg,
                     bounds_center=cam.center.tolist(), bounds_half=cam.half.tolist())
        json.dump(start, open(os.path.join(out, "start.json"), "w"), indent=2)
        tile = Image.fromarray(rgb[0]).resize((416, 240))
        ImageDraw.Draw(tile).text((6, 4), f"house {house}  ahead {start['depth_ahead_m']:.1f} m", fill=(255, 255, 0))
        tiles.append(np.asarray(tile))
        print(f"house {house}: pose {np.round(start['pose'], 2).tolist()} ahead {start['depth_ahead_m']:.1f} m "
              f"fov_arg {cam.fov_arg:.1f} ({time.perf_counter() - t0:.1f}s)", flush=True)

    cols = 4
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), 1)
            for i in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out_dir, "candidates.png"))


if __name__ == "__main__":
    main()
