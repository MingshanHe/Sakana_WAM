"""Evaluate the Wan2.2 imaginations: does the camera move, and in the instructed direction? (env `sakana_dreamer`)

Dense Farneback flow every 4th frame, from which (image coordinates, x right, y down):
  flow_px     mean flow magnitude [px / 4 frames]                     -> amount of motion
  div         mean divergence in the central half (> 0 = expanding)   -> forward (< 0 backward)
  fx, fy      mean flow in the central band [px / 4 frames]           -> fx > 0: scene moves right = camera turns /
                                                                         slides left; fy > 0: scene moves down = rising
  persist     fraction of frame pairs whose fx has the run's sign     -> steady turn vs jitter
The instructed motion is checked with sign rules on these (thresholds in RULES), plus a still-image check.
Writes scenarios/imagination_eval.{json,md} and scenarios/overview.png (I0 + frames at 1, 2.5, 5 s per run).
"""

import glob
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_FLOW = 1.0  # px / 4 frames: below this the "video" is a still image
RULES = {
    "forward": lambda m: m["div"] > 0.004,
    "hallway": lambda m: m["div"] > 0.004,
    "doorway": lambda m: m["div"] > 0.004,
    "approach": lambda m: m["div"] > 0.004,
    "approach+stop": lambda m: m["div"] > 0.004,
    "turn left": lambda m: m["fx"] > 0.5,
    "turn right": lambda m: m["fx"] < -0.5,
    "strafe right": lambda m: m["fx"] < -0.5,
    "ascend": lambda m: m["fy"] > 0.5,
    "descend": lambda m: m["fy"] < -0.5,
}


def flow_stats(frames: np.ndarray, stride: int = 4) -> dict:
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[::stride]]
    h, w = gray[0].shape
    c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
    mags, divs, fxs, fys = [], [], [], []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mags.append(np.linalg.norm(flow, axis=-1).mean())
        divs.append((np.gradient(fx, axis=1) + np.gradient(fy, axis=0))[c].mean())
        fxs.append(fx[c].mean())
        fys.append(fy[c].mean())
    fxs = np.array(fxs)
    sign = np.sign(fxs.mean()) or 1.0
    return {"flow_px": float(np.mean(mags)), "div": float(np.mean(divs)), "fx": float(fxs.mean()),
            "fy": float(np.mean(fys)), "persist": float((np.sign(fxs) == sign).mean())}


def main():
    root = os.path.join(HERE, "scenarios")
    scenarios = json.load(open(os.path.join(root, "scenarios.json")))["scenarios"]
    rows, tiles = [], []
    for sc in scenarios:
        for run in sorted(glob.glob(os.path.join(root, sc["id"], "seed*"))):
            video = np.load(os.path.join(run, "video.npy"), mmap_mode="r")
            m = flow_stats(np.asarray(video))
            m.update(id=sc["id"], motion=sc["motion"], seed=int(run[-1]), still=m["flow_px"] < MIN_FLOW)
            m["direction_ok"] = bool(not m["still"] and RULES[sc["motion"]](m))
            m.update(json.load(open(os.path.join(run, "metrics.json"))))
            rows.append(m)
            picks = [0, 24, 60, 120]
            strip = np.concatenate([np.asarray(Image.fromarray(video[i]).resize((208, 120))) for i in picks], 1)
            img = Image.fromarray(strip)
            ImageDraw.Draw(img).text((4, 2), f"{sc['id']} s{m['seed']}  {'OK' if m['direction_ok'] else 'X'}",
                                     fill=(255, 255, 0))
            tiles.append(np.asarray(img))
    json.dump(rows, open(os.path.join(root, "imagination_eval.json"), "w"), indent=2)
    with open(os.path.join(root, "imagination_eval.md"), "w") as f:
        f.write("| scenario | instructed | seed | flow px | div | fx | fy | persist | still | direction ok |\n"
                "|---|---|---:|---:|---:|---:|---:|---:|:-:|:-:|\n")
        for r in rows:
            f.write(f"| {r['id']} | {r['motion']} | {r['seed']} | {r['flow_px']:.2f} | {r['div']:+.4f} | "
                    f"{r['fx']:+.2f} | {r['fy']:+.2f} | {r['persist']:.2f} | {'yes' if r['still'] else ''} | "
                    f"{'✅' if r['direction_ok'] else '❌'} |\n")
    half = (len(tiles) + 1) // 2
    cols = [np.concatenate(tiles[:half], 0), np.concatenate(tiles[half:] + [np.zeros_like(tiles[0])] * (2 * half - len(tiles)), 0)]
    Image.fromarray(np.concatenate(cols, 1)).save(os.path.join(root, "overview.png"))
    ok = sum(r["direction_ok"] for r in rows)
    print(f"{ok}/{len(rows)} runs move in the instructed direction; "
          f"{sum(r['still'] for r in rows)} still; scenarios with >=1 good seed: "
          f"{len({r['id'] for r in rows if r['direction_ok']})}/{len(scenarios)}")
    print(open(os.path.join(root, "imagination_eval.md")).read())


if __name__ == "__main__":
    main()
