"""Wan2.2-TI2V-5B imagination for every scenario x seed (env: conda `sakana_dreamer`).

For each scenario in scenarios.json and each seed, with the same noise for both:
  flash.pt      DiffWAM-Flash features: Z (1, 3, 39, 15, 26, 3072) bf16 + slot times  -> the latent goal
  video.npy     the full imagined rollout, (121, 480, 832, 3) uint8                   -> encoder targets / VAE baseline
  video.mp4, frames.png, metrics.json (optical-flow motion, see wan/sweep_imagine.py)

    python imagine_scenarios.py --scenarios scenarios/scenarios.json --seeds 0 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "wan"), os.path.join(HERE, "..", "h3")]
from generate_video import contact_sheet  # noqa: E402
from sweep_imagine import NEGATIVE, motion_metrics  # noqa: E402
from wan_features import WAN_LAYERS, WanFeatureExtractor, load_pipeline, to_diffwam_slots  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default=os.path.join(HERE, "scenarios", "scenarios.json"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--only", nargs="*", default=None, help="Scenario ids to run (default: all).")
    args = parser.parse_args()
    from diffusers.utils.export_utils import encode_video

    root = os.path.dirname(os.path.abspath(args.scenarios))
    scenarios = json.load(open(args.scenarios))["scenarios"]
    pipe = load_pipeline()
    for sc in scenarios:
        if args.only and sc["id"] not in args.only:
            continue
        image = Image.open(os.path.join(root, sc["id"], "I0.png")).convert("RGB")
        for seed in args.seeds:
            out = os.path.join(root, sc["id"], f"seed{seed}")
            os.makedirs(out, exist_ok=True)
            kwargs = dict(image=image, prompt=sc["instruction"], negative_prompt=NEGATIVE, height=480, width=832,
                          num_frames=121, num_inference_steps=50, guidance_scale=5.0)

            t0 = time.perf_counter()
            with WanFeatureExtractor(pipe.transformer, WAN_LAYERS, (0,), passes_per_step=2) as ext:
                ext.run(lambda: pipe(**kwargs, generator=torch.Generator("cuda").manual_seed(seed),
                                     output_type="latent"))
            flash = to_diffwam_slots(ext.result())
            flash_s = time.perf_counter() - t0
            torch.save(flash, os.path.join(out, "flash.pt"))

            t0 = time.perf_counter()
            video = np.asarray(pipe(**kwargs, generator=torch.Generator("cuda").manual_seed(seed),
                                    output_type="np").frames[0]).clip(0, 1)
            video_s = time.perf_counter() - t0
            frames = (video * 255).round().astype(np.uint8)
            np.save(os.path.join(out, "video.npy"), frames)
            encode_video(video, fps=24, output_path=os.path.join(out, "video.mp4"))
            contact_sheet(frames, 12).save(os.path.join(out, "frames.png"))
            metrics = {"id": sc["id"], "seed": seed, "flash_s": flash_s, "video_s": video_s, **motion_metrics(frames)}
            json.dump(metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
            print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
