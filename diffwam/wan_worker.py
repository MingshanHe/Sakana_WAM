"""DiffWAM data, stage B (env `sakana_dreamer`): Wan2.2 features + completed rollout for every sample.

For one sample (I0, instruction, seed), with the same noise for both passes:
  feats.pt   DiffWAM features of schedule steps 0..3 at blocks 9/15/21 (conditional CFG pass), each layer
             standardised and PCA-projected to 128 channels: Z (4, 3, 39, 15, 26, 128) on DiffWAM's 39 slots and
             A0 (4, 3, 1, 15, 26, 128) = latent frame 0 (the clean first frame), float16
  slots.npz  the completed rollout at the 39 slot times, (39, 480, 832, 3) uint8 -> teacher input for gen2pose
Shards the index by `--shard i --num-shards n`, skips finished samples, and keeps polling the index (stage A may
still be adding samples) until no new work appears for `--idle-exit` seconds.

    python wan_worker.py --shard 0 --num-shards 3
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "wan"), os.path.join(HERE, "..", "follow")]
from latents import slot_frame_index  # noqa: E402
from sweep_imagine import NEGATIVE  # noqa: E402
from wan_features import WAN_LAYERS, WanFeatureExtractor, latent_slot_times, load_pipeline, resample_slots  # noqa: E402


def project(z: torch.Tensor, pca: dict) -> torch.Tensor:
    """(K, 3, T, H, W, 3072) -> (K, 3, T, H, W, D) float16 with the per-layer standardisation + PCA."""
    mean = pca["mean"].cuda()[None, :, None, None, None]
    std = pca["std"].cuda()[None, :, None, None, None]
    x = (z.cuda().float() - mean) / std
    return torch.einsum("klthwc,lcd->klthwd", x, pca["components"].cuda()).half().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", default=os.path.join(HERE, "data", "samples"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--idle-exit", type=float, default=600)
    parser.add_argument("--gpu", type=int, default=0, help="Device index within the visible GPUs.")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)  # every "cuda" below (pipeline, generator, PCA) resolves to this device
    pca = torch.load(os.path.join(HERE, "data", "feature_pca.pt"))
    pipe = load_pipeline()
    last_work = time.time()
    while True:
        index = json.load(open(os.path.join(args.samples, "index.json")))
        todo = [s for s in index if zlib.crc32(s["id"].encode()) % args.num_shards == args.shard
                and not os.path.exists(os.path.join(args.samples, s["id"], "slots.npz"))]
        if not todo:
            if time.time() - last_work > args.idle_exit:
                break
            time.sleep(30)
            continue
        for s in todo:
            d = os.path.join(args.samples, s["id"])
            image = Image.open(os.path.join(d, "I0.png")).convert("RGB")
            kw = dict(image=image, prompt=s["instruction"], negative_prompt=NEGATIVE, height=480, width=832,
                      num_frames=121, num_inference_steps=args.steps, guidance_scale=5.0)
            t0 = time.perf_counter()
            with WanFeatureExtractor(pipe.transformer, WAN_LAYERS, (0, 1, 2, 3), passes_per_step=2) as ext:
                ext.run(lambda: pipe(**kw, generator=torch.Generator("cuda").manual_seed(s["seed"]), output_type="latent"))
            res = ext.result()
            feat_s = time.perf_counter() - t0
            z = res["Z"]  # (4, 3, 31, 15, 26, 3072) native latent frames
            a0 = project(z[:, :, :1], pca)
            native_t = latent_slot_times(z.shape[2])
            z39 = resample_slots(project(z, pca).float(), native_t, torch.linspace(0, 5, 39), dim=2).half()
            torch.save({"Z": z39, "A0": a0, "timesteps": res["timestep_Z"][:, 1:, 0, 0].max(1).values,
                        "layers": list(WAN_LAYERS), "steps": args.steps, "seconds": feat_s},
                       os.path.join(d, "feats.pt"))
            t0 = time.perf_counter()
            video = pipe(**kw, generator=torch.Generator("cuda").manual_seed(s["seed"]), output_type="np").frames[0]
            frames = (np.asarray(video).clip(0, 1) * 255).round().astype(np.uint8)
            np.savez_compressed(os.path.join(d, "slots.npz"), frames=frames[slot_frame_index(len(frames))])
            print(json.dumps({"id": s["id"], "feat_s": round(feat_s, 2), "video_s": round(time.perf_counter() - t0, 1)}),
                  flush=True)
            last_work = time.time()


if __name__ == "__main__":
    main()
