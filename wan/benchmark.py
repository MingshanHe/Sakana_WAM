"""Wan2.2-TI2V-5B: time DiffWAM features vs the whole decoded video, and save the video. Counterpart of h3/benchmark.py.

One process, everything resident on one GPU (no offload), warm-up first, then on the same request and seed:
  diffwam        steps 0-3, taps 9/15/21 (conditional pass), stop after block 21 of step 3
  diffwam_flash  step 0 only
  full_video     all 50 steps (x2 passes for CFG) + VAE decode; saved as --video (mp4) + _frames.png + _input.png

Stages (CUDA-synchronised): text_encoder (umT5), vae_encode, dit_compute (block compute summed over all
transformer passes), vae_decode, other.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from wan_features import (WAN_LAYERS, WanFeatureExtractor, add_request_args, load_pipeline, pipe_kwargs,
                          resolve_request, to_diffwam_slots)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "h3"))
from generate_video import contact_sheet  # noqa: E402


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class StageTimer:
    def __init__(self, pipe):
        self.pipe, self._handles, self._patched = pipe, [], []

    def __enter__(self):
        self.t = {k: 0.0 for k in ("text_encoder", "vae_encode", "dit_compute", "vae_decode")}
        self.passes = 0
        blocks = self.pipe.transformer.blocks

        def te_pre(*_):
            self._te = _sync()

        def te_post(*_):
            self.t["text_encoder"] += _sync() - self._te

        def b0_pre(*_):
            self.passes += 1
            self._b0 = _sync()
            self._open = True

        def b_post(*_):
            self._b_end = _sync()

        def tr_post(*_):
            self._close()

        te = self.pipe.text_encoder
        self._handles += [te.register_forward_pre_hook(te_pre), te.register_forward_hook(te_post),
                          blocks[0].register_forward_pre_hook(b0_pre),
                          self.pipe.transformer.register_forward_hook(tr_post)]
        self._handles += [b.register_forward_hook(b_post) for b in blocks]
        self._open = False
        for comp, method, stage in ((self.pipe.vae, "encode", "vae_encode"), (self.pipe.vae, "decode", "vae_decode")):
            original = getattr(comp, method)

            def timed(*a, _f=original, _s=stage, **k):
                t0 = _sync()
                out = _f(*a, **k)
                self.t[_s] += _sync() - t0
                return out

            setattr(comp, method, timed)
            self._patched.append((comp, method, original))
        return self

    def _close(self):
        if self._open:
            self.t["dit_compute"] += self._b_end - self._b0
            self._open = False

    def __exit__(self, *exc):
        self._close()  # the early-stopped pass never reaches the transformer's post hook
        for h in self._handles:
            h.remove()
        for comp, method, original in self._patched:
            setattr(comp, method, original)
        return False

    def report(self, total):
        t = dict(self.t)
        t["other"] = total - sum(t.values())
        t["total"] = total
        return {"seconds": t, "transformer_passes": self.passes}


def run_features(pipe, kwargs, evals, slots):
    passes = 2 if kwargs["guidance_scale"] > 1.0 else 1
    with StageTimer(pipe) as timer, WanFeatureExtractor(pipe.transformer, WAN_LAYERS, evals, passes) as ext:
        t0 = _sync()
        ext.run(lambda: pipe(**kwargs, output_type="latent"))
        total = _sync() - t0
    out = timer.report(total)
    out["Z_shape"] = list(to_diffwam_slots(ext.result(), slots)["Z"].shape)
    return out


def run_full_video(pipe, kwargs, video_path):
    from diffusers.utils.export_utils import encode_video

    with StageTimer(pipe) as timer:
        t0 = _sync()
        video = pipe(**kwargs, output_type="np").frames[0]
        total = _sync() - t0
    out = timer.report(total)
    video = np.asarray(video).clip(0, 1)
    out["video_shape"] = list(video.shape)
    stem = os.path.splitext(video_path)[0]
    encode_video(video, fps=24, output_path=video_path)
    contact_sheet((video * 255).round().astype(np.uint8), 12).save(f"{stem}_frames.png")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--slots", type=int, default=39)
    parser.add_argument("--out", default="outputs/benchmark.json")
    parser.add_argument("--video", default="outputs/test_video.mp4")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    prompt, image = resolve_request(args)
    image.save(os.path.splitext(args.video)[0] + "_input.png")
    t0 = time.perf_counter()
    pipe = load_pipeline(args.repo)
    results = {"pipeline_load_s": time.perf_counter() - t0, "gpu": torch.cuda.get_device_name(0),
               "peak_memory_gb": None, "request": {k: v for k, v in vars(args).items() if k not in ("out", "video")}}

    print("warm-up ...")
    run_features(pipe, pipe_kwargs(args, prompt, image), (0, 1, 2, 3), args.slots)
    torch.cuda.reset_peak_memory_stats()
    for name, fn in (("diffwam", lambda: run_features(pipe, pipe_kwargs(args, prompt, image), (0, 1, 2, 3), args.slots)),
                     ("diffwam_flash", lambda: run_features(pipe, pipe_kwargs(args, prompt, image), (0,), args.slots)),
                     ("full_video", lambda: run_full_video(pipe, pipe_kwargs(args, prompt, image), args.video))):
        print(f"timing {name} ...", flush=True)
        results[name] = fn()
        print(json.dumps(results[name]), flush=True)
    results["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    modes = ("diffwam", "diffwam_flash", "full_video")
    print(f"\n{'stage':<14}" + "".join(f"{m:>15}" for m in modes))
    for s in results["full_video"]["seconds"]:
        print(f"{s:<14}" + "".join(f"{results[m]['seconds'][s]:>15.2f}" for m in modes))
    print(f"{'DiT passes':<14}" + "".join(f"{results[m]['transformer_passes']:>15}" for m in modes))
    print(f"peak GPU memory {results['peak_memory_gb']:.1f} GB; saved {args.out}, {args.video}")


if __name__ == "__main__":
    main()
