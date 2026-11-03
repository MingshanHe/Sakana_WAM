"""Time DiffWAM-style feature extraction against generating + decoding the whole H3 video, stage by stage.

One process loads the pipeline once, does a warm-up feature run, then times on the same request and seed:
  diffwam        evals 0-3, taps 15/25/35, stop after block 35 of eval 3   (DiffWAM)
  diffwam_flash  eval 0 only, stop after block 35                          (DiffWAM-Flash)
  full_video     all 49 evaluations + video VAE decode + audio VAE decode  (what a video world model would run)

Stages are measured with CUDA-synchronised wall clock:
  text_encoder   Qwen3-VL prompt + keyframe encoding compute (`text_encoder.model`)
  te_offload     swapping the 62 GB text encoder onto the GPU under CPU offload (its `_hf_hook.pre_forward`)
  vae_encode     first-frame VAE encode
  dit_compute    transformer block compute, summed over evaluations (block 0 start -> last executed block end)
  dit_offload    the rest of the transformer calls: swapping the 62 GB transformer onto the GPU under CPU offload
  vae_decode / audio_decode
  other          everything else (layout, scheduler, pre/post-processing)

Writes `--out` (JSON) and prints a table.
"""

import argparse
import json
import time

import numpy as np
import torch

from h3_features import H3FeatureExtractor, add_request_args, load_pipeline, resolve_request, to_diffwam_slots


def _sync():
    torch.cuda.synchronize()
    return time.perf_counter()


class StageTimer:
    """Hooks the pipeline components and accumulates per-stage seconds for one pipeline call."""

    def __init__(self, pipe):
        self.pipe = pipe
        self._handles, self._patched = [], []

    def __enter__(self):
        self.t = {k: 0.0 for k in ("te_offload", "text_encoder", "vae_encode", "dit_call", "dit_compute", "vae_decode",
                                   "audio_decode")}
        self.evals, self.eval_blocks = 0, []
        tr = self.pipe.transformer
        blocks = tr.transformer_blocks

        def call_pre(name):
            def hook(*_args, **_kwargs):
                self._start = _sync()
            return hook

        def call_post(name):
            def hook(*_args, **_kwargs):
                self.t[name] += _sync() - self._start
            return hook

        # The H3 encoder block calls `text_encoder.model` directly and fires the offload hook by hand.
        te = self.pipe.text_encoder.model
        self._handles += [te.register_forward_pre_hook(call_pre("text_encoder")),
                          te.register_forward_hook(call_post("text_encoder"))]

        def tr_pre(*_):
            self.evals += 1
            self.eval_blocks.append(0)
            self._tr_start = _sync()
            self._block_start = None

        def tr_post(*_):
            self.t["dit_call"] += _sync() - self._tr_start
            self._close_blocks()

        def block0_pre(*_):
            self._block_start = _sync()

        def block_post(*_):
            self._block_end = _sync()
            self.eval_blocks[-1] += 1

        self._handles += [tr.register_forward_pre_hook(tr_pre), tr.register_forward_hook(tr_post),
                          blocks[0].register_forward_pre_hook(block0_pre)]
        self._handles += [b.register_forward_hook(block_post) for b in blocks]

        te_hook = getattr(self.pipe.text_encoder, "_hf_hook", None)
        timed_methods = [(te_hook, "pre_forward", "te_offload")] if hasattr(te_hook, "pre_forward") else []
        for comp, method, stage in timed_methods + [(self.pipe.vae, "encode", "vae_encode"), (self.pipe.vae, "decode", "vae_decode"),
                                    (self.pipe.audio_vae, "decode", "audio_decode")]:
            original = getattr(comp, method)

            def timed(*a, _f=original, _s=stage, **k):
                t0 = _sync()
                out = _f(*a, **k)
                self.t[_s] += _sync() - t0
                return out

            setattr(comp, method, timed)
            self._patched.append((comp, method, original))
        return self

    def _close_blocks(self):
        if self._block_start is not None:
            self.t["dit_compute"] += self._block_end - self._block_start
            self._block_start = None

    def close_truncated_eval(self):
        """The early-stopped evaluation never reaches the transformer's post hook."""
        if self._block_start is not None:
            self.t["dit_call"] += self._block_end - self._tr_start
            self._close_blocks()

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        for comp, method, original in self._patched:
            setattr(comp, method, original)
        return False

    def report(self, total: float) -> dict:
        t = dict(self.t)
        t["dit_offload"] = t.pop("dit_call") - t["dit_compute"]
        t["other"] = total - sum(t.values())
        t["total"] = total
        return {"seconds": t, "transformer_evals": self.evals, "blocks_per_eval": self.eval_blocks}


def run_features(pipe, kwargs, evals, slots):
    with StageTimer(pipe) as timer, H3FeatureExtractor(pipe.transformer, evals=evals) as ext:
        t0 = _sync()
        ext.run(lambda: pipe(**kwargs, output=["videos"]))
        timer.close_truncated_eval()
        total = _sync() - t0
    feats = to_diffwam_slots(ext.result(), slots)
    out = timer.report(total)
    out["Z_shape"] = list(feats["Z"].shape)
    return out


def run_full_video(pipe, kwargs):
    with StageTimer(pipe) as timer:
        t0 = _sync()
        res = pipe(**kwargs, output_type="np", output=["videos", "audio", "sampling_rate"])
        total = _sync() - t0
    out = timer.report(total)
    out["video_shape"] = list(np.asarray(res["videos"][0]).shape)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--slots", type=int, default=39)
    parser.add_argument("--out", default="outputs/benchmark.json")
    args = parser.parse_args()

    prompt, image = resolve_request(args)
    t0 = time.perf_counter()
    pipe = load_pipeline(args.placement, args.repo, args.attention_backend)
    load_s = time.perf_counter() - t0

    def kwargs():
        kw = dict(prompt=prompt, height=args.height, width=args.width, num_frames=args.num_frames,
                  num_inference_steps=args.num_inference_steps, generator=torch.Generator().manual_seed(args.seed))
        if image is not None:
            kw["image"] = image
        return kw

    print("warm-up ...")
    run_features(pipe, kwargs(), (0, 1, 2, 3), args.slots)
    results = {"pipeline_load_s": load_s, "gpu": torch.cuda.get_device_name(0), "placement": args.placement,
               "request": {k: v for k, v in vars(args).items() if k not in ("out",)}}
    for name, fn in (("diffwam", lambda: run_features(pipe, kwargs(), (0, 1, 2, 3), args.slots)),
                     ("diffwam_flash", lambda: run_features(pipe, kwargs(), (0,), args.slots)),
                     ("full_video", lambda: run_full_video(pipe, kwargs()))):
        print(f"timing {name} ...")
        results[name] = fn()
        print(json.dumps(results[name]["seconds"], indent=None))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    stages = list(results["full_video"]["seconds"].keys())
    modes = ("diffwam", "diffwam_flash", "full_video")
    print(f"\n{'stage':<14}" + "".join(f"{m:>15}" for m in modes))
    for s in stages:
        print(f"{s:<14}" + "".join(f"{results[m]['seconds'][s]:>15.2f}" for m in modes))
    print(f"{'DiT evals':<14}" + "".join(f"{results[m]['transformer_evals']:>15}" for m in modes))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
