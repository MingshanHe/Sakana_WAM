"""Sweep prompts / seeds / resolutions for Wan2.2 imagination from one sim first frame, and score camera motion.

Loads the pipeline once, generates the full video for every combination, and measures motion with Farneback
optical flow between consecutive frames (every 4th frame):
  flow_px     mean flow magnitude [px per 4 frames, at 832 px width]   -> how much the camera moves
  divergence  mean flow divergence in the central region (> 0 = scene expands = forward motion)
  curl        mean flow curl (non-zero = rotation / turning)
  end_change  mean |last - first| / 255                                -> how different the end view is

Writes <out-dir>/<variant>/imagined.{mp4,npy}, imagined_frames.png, and <out-dir>/summary.{json,md}.

    python sweep_imagine.py --image ../sim/outputs/takeoff/first_frame.png --out-dir outputs/sweep_office
"""

import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from wan_features import load_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "h3"))
from generate_video import contact_sheet  # noqa: E402
from h3_features import prepare_image  # noqa: E402

# Wan's recommended negative prompt (it explicitly penalises static frames), plus what went wrong before.
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
    "drone, quadcopter, aircraft, hand, person, people, static camera, still image"
)
TAIL = ("Starts exactly from the supplied image. Constant height of about one meter, steady and smooth, continuous "
        "single shot, no cuts, no people, no third-person view. Preserve the room, walls, desks, chairs and plants.")
PROMPTS = {
    "forward": "First-person view: the camera flies straight forward quickly along the open office aisle, travelling "
               "about five meters; the desks and chairs ahead grow larger and pass by on both sides. " + TAIL,
    "forward_right": "First-person view: the camera flies forward along the office aisle for three seconds, then turns "
                     "smoothly to the right by ninety degrees and continues forward slowly. " + TAIL,
    "approach": "First-person dolly shot: the camera travels forward through the office toward the desks at the end of "
                "the aisle, getting much closer to them, then slows down and holds still in front of them. " + TAIL,
}


def motion_metrics(frames: np.ndarray, stride: int = 4) -> dict:
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[::stride]]
    h, w = gray[0].shape
    mags, divs, curls = [], [], []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.1, 0)
        fx, fy = flow[..., 0], flow[..., 1]
        mags.append(np.linalg.norm(flow, axis=-1).mean())
        dfx_dx, dfy_dy = np.gradient(fx, axis=1), np.gradient(fy, axis=0)
        dfy_dx, dfx_dy = np.gradient(fy, axis=1), np.gradient(fx, axis=0)
        c = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
        divs.append((dfx_dx + dfy_dy)[c].mean())
        curls.append((dfy_dx - dfx_dy)[c].mean())
    scale = 832.0 / w
    return {
        "flow_px": float(np.mean(mags) * scale),
        "divergence": float(np.mean(divs)),
        "curl": float(np.mean(curls)),
        "end_change": float(np.abs(frames[-1].astype(np.float32) - frames[0].astype(np.float32)).mean() / 255),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resolutions", nargs="+", default=["832x480"], help="WxH, e.g. 832x480 1280x704")
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from diffusers.utils.export_utils import encode_video

    pipe = load_pipeline()
    summary = []
    for res, prompt_name, seed in itertools.product(args.resolutions, args.prompts, args.seeds):
        w, h = map(int, res.split("x"))
        name = f"{prompt_name}_s{seed}_{res}"
        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        image = prepare_image(args.image, h, w)
        t0 = time.perf_counter()
        video = pipe(image=image, prompt=PROMPTS[prompt_name], negative_prompt=NEGATIVE, height=h, width=w,
                     num_frames=args.num_frames, num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                     generator=torch.Generator("cuda").manual_seed(seed), output_type="np").frames[0]
        seconds = time.perf_counter() - t0
        video = np.asarray(video).clip(0, 1)
        frames = (video * 255).round().astype(np.uint8)
        np.save(os.path.join(out, "imagined.npy"), frames)
        encode_video(video, fps=24, output_path=os.path.join(out, "imagined.mp4"))
        contact_sheet(frames, 12).save(os.path.join(out, "imagined_frames.png"))
        row = {"name": name, "prompt": prompt_name, "seed": seed, "resolution": res, "seconds": seconds,
               **motion_metrics(frames)}
        summary.append(row)
        print(json.dumps(row), flush=True)
        json.dump({"prompts": {k: PROMPTS[k] for k in args.prompts}, "negative": NEGATIVE, "runs": summary},
                  open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)

    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("| variant | flow px | divergence | curl | end change | s |\n|---|---:|---:|---:|---:|---:|\n")
        for r in sorted(summary, key=lambda r: -r["flow_px"]):
            f.write(f"| {r['name']} | {r['flow_px']:.2f} | {r['divergence']:+.4f} | {r['curl']:+.4f} | "
                    f"{r['end_change']:.3f} | {r['seconds']:.0f} |\n")
    print(open(os.path.join(args.out_dir, "summary.md")).read())


if __name__ == "__main__":
    main()
