"""Copy the DiffWAM / gen2pose report assets to ../results/wan2.2-ti2v-5b/diffwam_gen2pose/ (small files only)."""

import glob
import os
import shutil

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "wan2.2-ti2v-5b", "diffwam_gen2pose")
VIDEOS = ("s02_hallway_seed0", "s03_doorway_seed1", "s05_ascend_seed1", "s06_bed_approach_seed0",
          "s07_turn_right_seed1", "s09_approach_stop_seed0")


def main():
    os.makedirs(os.path.join(OUT, "flights"), exist_ok=True)
    Image.open(os.path.join(HERE, "data", "trajectories.png")).convert("RGB").save(
        os.path.join(OUT, "trajectories.jpg"), quality=85)
    shutil.copy(os.path.join(HERE, "data", "gen2pose_validation.json"), OUT)
    for tag in ("diffwam", "flash", "diffwam_scene_only"):
        shutil.copy(os.path.join(HERE, "data", "head", f"{tag}_metrics.json"), OUT)
    for tag in ("diffwam", "flash"):
        shutil.copy(os.path.join(HERE, "data", "deploy", tag, "deploy.json"), os.path.join(OUT, f"deploy_{tag}.json"))
        for sid in VIDEOS:
            for path in ("fast", "slow") if tag == "diffwam" else ("fast",):
                src = os.path.join(HERE, "data", "deploy", tag, f"{sid}_{path}.mp4")
                name = f"{sid}_{'teacher' if path == 'slow' else tag}.mp4"
                shutil.copy(src, os.path.join(OUT, "flights", name))
    total = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "**", "*"), recursive=True) if os.path.isfile(p))
    print(f"collected into {os.path.relpath(OUT, HERE)}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
