"""Aggregate follower episodes (follow_runs/*/summary.json) into follow_runs/follow_eval.{json,md}.

replay: position RMSE / final error vs the scripted reference (and vs hovering); success = final error < 25 % of
        the reference path length.
wan:    does the flown displacement match the instructed motion (rules below), and is the mean latent error below
        the hover baseline's (ratio < 1)?
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLOWN_OK = {
    "forward": lambda s: s["forward_m"] > 0.5,
    "hallway": lambda s: s["forward_m"] > 0.5,
    "doorway": lambda s: s["forward_m"] > 0.5,
    "approach": lambda s: s["forward_m"] > 0.5,
    "approach+stop": lambda s: s["forward_m"] > 0.5,
    "turn left": lambda s: s["yaw_deg"] > 15,
    "turn right": lambda s: s["yaw_deg"] < -15,
    "strafe right": lambda s: s["left_m"] < -0.3,
    "ascend": lambda s: s["up_m"] > 0.2,
    "descend": lambda s: s["up_m"] < -0.1,
}


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(HERE, "follow_runs", "*", "summary.json")))]
    imag = {(r["id"], r["seed"]): r for r in json.load(open(os.path.join(HERE, "scenarios", "imagination_eval.json")))}
    lines = []
    for r in runs:
        r["flown_ok"] = bool(FLOWN_OK[r["motion"]](r))
        r["err_ratio"] = r["err_mean"] / max(r["err_hover_mean"], 1e-12)
        if r["task"] == "replay":
            r["success"] = bool(r["pos_final_m"] < 0.25 * max(r["ref_length_m"], 1e-6))
        else:
            r["imagined_ok"] = imag[(r["scenario"], r["seed"])]["direction_ok"]

    lines.append("## Replay (goal = rendered scripted flight, ground truth known)\n")
    lines.append("| space | episodes | pos RMSE m | final err m | hover RMSE m | success | flown dir ok | err / hover |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for space in ("pixel", "vae", "flash"):
        rs = [r for r in runs if r["task"] == "replay" and r["space"] == space]
        if rs:
            lines.append(f"| {space} | {len(rs)} | {np.mean([r['pos_rmse_m'] for r in rs]):.2f} | "
                         f"{np.mean([r['pos_final_m'] for r in rs]):.2f} | {np.mean([r['hover_pos_rmse_m'] for r in rs]):.2f} | "
                         f"{sum(r['success'] for r in rs)}/{len(rs)} | {sum(r['flown_ok'] for r in rs)}/{len(rs)} | "
                         f"{np.mean([r['err_ratio'] for r in rs]):.2f} |")
    lines.append("\n## Wan2.2 imagined videos (goal = the imagination)\n")
    lines.append("| space | episodes | flown dir ok | ...where the imagination itself was ok | err / hover |")
    lines.append("|---|---:|---:|---:|---:|")
    for space in ("vae", "flash"):
        rs = [r for r in runs if r["task"] == "wan" and r["space"] == space]
        if rs:
            good = [r for r in rs if r["imagined_ok"]]
            lines.append(f"| {space} | {len(rs)} | {sum(r['flown_ok'] for r in rs)}/{len(rs)} | "
                         f"{sum(r['flown_ok'] for r in good)}/{len(good)} | {np.mean([r['err_ratio'] for r in rs]):.2f} |")
    lines.append("\n## Per episode\n")
    lines.append("| scenario | motion | task | space | seed | fwd m | left m | up m | yaw ° | ref fwd/left/up/yaw | "
                 "pos RMSE | flown ok | err/hover |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|---:|:-:|---:|")
    for r in runs:
        ref = (f"{r['ref_forward_m']:.1f}/{r['ref_left_m']:.1f}/{r['ref_up_m']:.1f}/{r['ref_yaw_deg']:.0f}"
               if r["task"] == "replay" else "")
        rmse = f"{r['pos_rmse_m']:.2f}" if r["task"] == "replay" else ""
        lines.append(f"| {r['scenario']} | {r['motion']} | {r['task']} | {r['space']} | {r['seed']} | "
                     f"{r['forward_m']:+.2f} | {r['left_m']:+.2f} | {r['up_m']:+.2f} | {r['yaw_deg']:+.0f} | {ref} | "
                     f"{rmse} | {'✅' if r['flown_ok'] else '❌'} | {r['err_ratio']:.2f} |")
    json.dump(runs, open(os.path.join(HERE, "follow_runs", "follow_eval.json"), "w"), indent=2)
    open(os.path.join(HERE, "follow_runs", "follow_eval.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
