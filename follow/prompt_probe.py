"""Quick Wan2.2 prompt probe on one first frame: full video per prompt variant + optical-flow motion score.

    python prompt_probe.py --image scenarios/s01_room_forward/I0.png --prompts prompts.json --out-dir probe
prompts.json: {"name": "prompt text", ...}
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "wan"), os.path.join(HERE, "..", "h3")]
from generate_video import contact_sheet  # noqa: E402
from sweep_imagine import NEGATIVE, motion_metrics  # noqa: E402
from wan_features import load_pipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pipe = load_pipeline()
    image = Image.open(args.image).convert("RGB")
    for name, prompt in json.load(open(args.prompts)).items():
        video = np.asarray(pipe(image=image, prompt=prompt, negative_prompt=NEGATIVE, height=480, width=832,
                                num_frames=121, num_inference_steps=50, guidance_scale=5.0,
                                generator=torch.Generator("cuda").manual_seed(args.seed), output_type="np").frames[0])
        frames = (video.clip(0, 1) * 255).round().astype(np.uint8)
        contact_sheet(frames, 12).save(os.path.join(args.out_dir, f"{name}.png"))
        print(json.dumps({"name": name, **motion_metrics(frames)}), flush=True)


if __name__ == "__main__":
    main()
