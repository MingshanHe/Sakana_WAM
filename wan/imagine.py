"""Phase 1: imagine the flight from the sim's first frame with Wan2.2-TI2V-5B.

For one (I0, instruction, seed) it saves, in `--out-dir`:
  flash.pt        DiffWAM-Flash features (step 0 only): Z (1, 3, 39, 15, 26, 3072), A0, slot times -> latent goal
  diffwam.pt      DiffWAM features (steps 0-3), for comparison
  imagined.mp4    the full rollout of the same request + seed (the "imagined video"), 121 frames, 24 fps
  imagined.npy    its frames (T, H, W, 3) uint8: encoder-training targets and the VAE-latent baseline
  imagined_frames.png  contact sheet
  meta.json       request, timings

    python imagine.py --image ../sim/outputs/takeoff/first_frame.png --prompt-file ../sim/prompts/office_forward.txt \
        --out-dir outputs/office_forward
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from wan_features import add_request_args, extract, load_pipeline, pipe_kwargs, resolve_request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "h3"))
from generate_video import contact_sheet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    prompt, image = resolve_request(args)
    image.save(os.path.join(args.out_dir, "input.png"))
    t0 = time.perf_counter()
    pipe = load_pipeline(args.repo)
    meta = {"request": vars(args), "prompt": prompt, "load_s": time.perf_counter() - t0}

    for name, evals in (("flash", (0,)), ("diffwam", (0, 1, 2, 3))):
        feats = extract(pipe, pipe_kwargs(args, prompt, image), evals=evals)
        torch.save(feats, os.path.join(args.out_dir, f"{name}.pt"))
        meta[f"{name}_s"] = feats["total_seconds"]
        print(f"{name}: Z {tuple(feats['Z'].shape)} in {feats['total_seconds']:.2f}s", flush=True)

    t0 = time.perf_counter()
    video = np.asarray(pipe(**pipe_kwargs(args, prompt, image), output_type="np").frames[0]).clip(0, 1)
    meta["full_video_s"] = time.perf_counter() - t0
    frames = (video * 255).round().astype(np.uint8)
    np.save(os.path.join(args.out_dir, "imagined.npy"), frames)
    from diffusers.utils.export_utils import encode_video

    encode_video(video, fps=24, output_path=os.path.join(args.out_dir, "imagined.mp4"))
    contact_sheet(frames, 12).save(os.path.join(args.out_dir, "imagined_frames.png"))
    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2, default=str)
    print(f"full video {frames.shape} in {meta['full_video_s']:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
