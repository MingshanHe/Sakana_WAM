"""Generate and decode the whole MiniMax-H3 video for a request, i.e. the rollout the DiffWAM features come from.

`h3_features.py` stops inside the 4th transformer evaluation; this script runs the same request to the end of the
schedule and VAE-decodes it. With the same image, prompt, canvas, frame count, step count and seed, the noise is
the same, so this video is exactly the completion of the rollout whose features were extracted (the DiffWAM
"teacher" rollout, Sec. 3.5 / Eq. 16-17).

Outputs, next to `--out` (an .mp4 path):
  <out>.mp4             video + generated stereo audio, 24 fps
  <out>_input.png       the cropped/resized first-frame condition actually fed to H3
  <out>_frames.png      contact sheet of every `--sheet-every`-th frame
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from h3_features import add_request_args, load_pipeline, resolve_request


def contact_sheet(frames: np.ndarray, every: int, cols: int = 6) -> Image.Image:
    """(T, H, W, 3) uint8 -> grid image of frames 0, every, 2*every, ... with the frame index in the corner."""
    from PIL import ImageDraw

    picks = list(range(0, len(frames), every))
    rows = (len(picks) + cols - 1) // cols
    h, w = frames.shape[1:3]
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for n, i in enumerate(picks):
        tile = Image.fromarray(frames[i])
        ImageDraw.Draw(tile).text((6, 4), f"#{i}", fill=(255, 255, 0))
        sheet.paste(tile, ((n % cols) * w, (n // cols) * h))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--out", default="outputs/video.mp4")
    parser.add_argument("--sheet-every", type=int, default=12, help="Contact-sheet stride in frames (12 = 0.5 s).")
    args = parser.parse_args()

    from diffusers.utils.export_utils import encode_video

    prompt, image = resolve_request(args)
    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if image is not None:
        image.save(f"{stem}_input.png")

    t0 = time.perf_counter()
    pipe = load_pipeline(args.placement, args.repo, args.attention_backend)
    print(f"loaded pipeline in {time.perf_counter() - t0:.1f}s")

    kwargs = dict(
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator().manual_seed(args.seed),
        output_type="np",
        output=["videos", "audio", "sampling_rate"],
    )
    if image is not None:
        kwargs["image"] = image
    t0 = time.perf_counter()
    results = pipe(**kwargs)
    print(f"generated in {time.perf_counter() - t0:.1f}s")

    video = np.asarray(results["videos"][0])  # (T, H, W, 3) float in [0, 1]
    frames = (video.clip(0, 1) * 255).round().astype(np.uint8)
    encode_video(
        video.clip(0, 1),
        fps=24,
        output_path=args.out,
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    contact_sheet(frames, args.sheet_every).save(f"{stem}_frames.png")
    print(f"saved {args.out} ({frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]}), {stem}_frames.png")


if __name__ == "__main__":
    main()
