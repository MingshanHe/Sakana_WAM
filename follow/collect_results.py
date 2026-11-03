"""Copy the report assets of the 10-scenario test into ../results/wan2.2-ti2v-5b/scenarios10/ (small files only).

  imagination/overview.jpg                 I0 + frames at 1 / 2.5 / 5 s for every scenario x seed
  imagination/<id>_seed<k>.mp4, _frames.jpg  the seed the follower used (one per scenario)
  imagination_eval.md / .json, scenarios.json, I0_sheet.jpg
  follow/<episode>.mp4                     goal | follower side by side (416 x 240 each), for every wan episode
                                           and the replay episodes of a few scenarios
  follow_eval.md / .json, encoder_metrics.json, encoder_metrics_seedsplit.json
"""

import glob
import json
import os
import shutil

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "wan2.2-ti2v-5b", "scenarios10")
REPLAY_EXAMPLES = ("s01_room_forward", "s04_turn_left", "s07_turn_right")


def jpg(src, dst, quality=82, scale=1.0):
    im = Image.open(src).convert("RGB")
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    im.save(dst, quality=quality)


def main():
    os.makedirs(os.path.join(OUT, "imagination"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "follow"), exist_ok=True)
    sc = os.path.join(HERE, "scenarios")
    for name in ("imagination_eval.md", "imagination_eval.json", "scenarios.json"):
        shutil.copy(os.path.join(sc, name), OUT)
    jpg(os.path.join(sc, "overview.png"), os.path.join(OUT, "imagination", "overview.jpg"))
    jpg(os.path.join(HERE, "scenarios_raw", "candidates.png"), os.path.join(OUT, "I0_candidates.jpg"))

    runs = json.load(open(os.path.join(HERE, "follow_runs", "follow_eval.json")))
    used = {(r["scenario"], r["seed"]) for r in runs if r["task"] == "wan"}
    for sid, seed in sorted(used):
        run = os.path.join(sc, sid, f"seed{seed}")
        shutil.copy(os.path.join(run, "video.mp4"), os.path.join(OUT, "imagination", f"{sid}_seed{seed}.mp4"))
        jpg(os.path.join(run, "frames.png"), os.path.join(OUT, "imagination", f"{sid}_seed{seed}_frames.jpg"), scale=0.5)

    for d in sorted(glob.glob(os.path.join(HERE, "follow_runs", "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d):
            continue
        if "_wan_" in name or (name.split("_replay_")[0] in REPLAY_EXAMPLES):
            shutil.copy(os.path.join(d, "goal_vs_follower.mp4"), os.path.join(OUT, "follow", f"{name}.mp4"))
    for name in ("follow_eval.md", "follow_eval.json"):
        shutil.copy(os.path.join(HERE, "follow_runs", name), OUT)
    shutil.copy(os.path.join(HERE, "encoders", "metrics.json"), os.path.join(OUT, "encoder_metrics.json"))
    shutil.copy(os.path.join(HERE, "encoders_seedsplit", "metrics.json"),
                os.path.join(OUT, "encoder_metrics_seedsplit.json"))
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p))
    print(f"collected into {os.path.relpath(OUT, HERE)}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
