"""DiffWAM-style predictive features from Wan2.2-TI2V-5B, the small counterpart of `h3/h3_features.py`.

Same read-out as for MiniMax-H3: run the image-to-video rollout for a few denoising steps and record DiT hidden
states, with no VAE decode. Differences from H3:

  * Wan uses classifier-free guidance: every step runs the transformer twice (conditional, then unconditional).
    Steps 0..K-2 run both passes so the latent follows exactly the rollout of the full video; features come from
    the conditional pass only, and the last step stops right after its deepest tap (no unconditional pass).
  * 30 blocks instead of 50: the taps default to 9 / 15 / 21, the same relative depth as H3's 15 / 25 / 35.
  * The first frame is not a separate keyframe block: TI2V puts the clean image latent in latent frame 0
    (timestep 0) and denoises the others. `A0` is latent frame 0; `Z` is all latent frames, resampled like H3.
  * Video VAE 16x16x4 + patch 1x2x2 -> 15 x 26 tokens at 832 x 480; 121 frames (5.04 s at 24 fps) -> 31 latents.

    python wan_features.py --image ../h3/assets/test.jpg --prompt-file ../h3/assets/test_instruction --out outputs/x.pt
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "h3"))
from h3_features import DIFFWAM_EVALS, DIFFWAM_HORIZON_S, DIFFWAM_SLOTS, prepare_image, resample_slots  # noqa: E402

REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_LAYERS = (9, 15, 21)
WAN_FPS = 24
NEGATIVE_PROMPT = ""


class _StopAfterLastTap(Exception):
    pass


class WanFeatureExtractor:
    r"""
    Hooks `WanTransformer3DModel.blocks` and records the conditional-pass outputs of `layers` at denoising steps
    `evals`, then aborts the rollout after the deepest tap of the last step. With CFG the transformer is called
    `passes_per_step` times per step, conditional first (as in `WanImageToVideoPipeline`).
    """

    def __init__(self, transformer, layers=WAN_LAYERS, evals=DIFFWAM_EVALS, passes_per_step=2,
                 store_dtype=torch.bfloat16):
        self.transformer, self.passes_per_step = transformer, passes_per_step
        self.layers, self.evals = tuple(sorted(layers)), tuple(sorted(evals))
        self.store_dtype = store_dtype
        self._handles = []

    def _pre_hook(self, module, args, kwargs):
        self.call_idx += 1
        self.step, self.pass_idx = divmod(self.call_idx, self.passes_per_step)
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        _, _, f, h, w = hidden.shape
        p = self.transformer.config.patch_size
        self.grid = (f // p[0], h // p[1], w // p[2])
        timestep = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if self.pass_idx == 0 and self.step in self.evals:
            self.timesteps[self.step] = timestep[0].float().cpu()  # (seq,) per-token timestep (expand_timesteps)

    def _make_hook(self, layer):
        def hook(module, args, output):
            if self.pass_idx != 0 or self.step not in self.evals:
                return
            self.features[(self.step, layer)] = output[0].to("cpu", self.store_dtype)
            if self.step == self.evals[-1] and layer == self.layers[-1]:
                self.finished = True
                raise _StopAfterLastTap
        return hook

    def __enter__(self):
        self.call_idx, self.features, self.timesteps, self.finished = -1, {}, {}, False
        self._handles.append(self.transformer.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for layer in self.layers:
            self._handles.append(self.transformer.blocks[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def run(self, fn):
        try:
            fn()
        except _StopAfterLastTap:
            pass
        if not self.finished:
            raise RuntimeError("The rollout ended before the last requested step; raise `num_inference_steps`.")

    def result(self) -> dict:
        t, h, w = self.grid
        z = torch.stack([torch.stack([self.features[(k, l)].view(t, h, w, -1) for l in self.layers])
                         for k in self.evals])  # (K, L, T, H, W, C)
        tz = torch.stack([self.timesteps[k].view(t, h, w) for k in self.evals])  # (K, T, H, W), 0..1000
        return {"Z": z, "A0": z[:, :, :1].clone(), "timestep_Z": tz, "layers": list(self.layers),
                "evals": list(self.evals), "layout": {"num_frames": t, "grid_h": h, "grid_w": w,
                                                      "sequence_length": t * h * w}}


def latent_slot_times(num_latent_frames: int, fps: float = WAN_FPS) -> torch.Tensor:
    """Wan VAE: latent 0 = pixel frame 0, latent k packs pixel frames 4k-3..4k; time = centre of the packed frames."""
    return torch.tensor([0.0] + [(4 * k - 1.5) / fps for k in range(1, num_latent_frames)])


def to_diffwam_slots(feats: dict, num_slots: int = DIFFWAM_SLOTS, horizon_s: float = DIFFWAM_HORIZON_S) -> dict:
    native_t = latent_slot_times(feats["Z"].shape[2])
    feats["slot_time_native_s"] = native_t
    if not num_slots:
        feats["slot_time_s"] = native_t
        return feats
    target_t = torch.linspace(0.0, horizon_s, num_slots)
    feats["Z"] = resample_slots(feats["Z"], native_t, target_t, dim=2)
    feats["timestep_Z"] = resample_slots(feats["timestep_Z"], native_t, target_t, dim=1)
    feats["slot_time_s"] = target_t
    return feats


def load_pipeline(repo: str = REPO_ID, device: str = "cuda"):
    """Everything resident on one GPU (~35 GB): no offload. The VAE stays float32 as Wan recommends."""
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(repo, vae=vae, torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def add_request_args(parser: argparse.ArgumentParser):
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file")
    parser.add_argument("--image", required=True, help="First frame (TI2V image-to-video).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=121, help="4n+1; 121 = 5.04 s at 24 fps -> 31 latents.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=REPO_ID)


def pipe_kwargs(args, prompt, image) -> dict:
    return dict(image=image, prompt=prompt, negative_prompt=NEGATIVE_PROMPT, height=args.height, width=args.width,
                num_frames=args.num_frames, num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, generator=torch.Generator("cuda").manual_seed(args.seed))


def resolve_request(args):
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    return prompt, prepare_image(args.image, args.height, args.width)


def extract(pipe, kwargs: dict, layers=WAN_LAYERS, evals=DIFFWAM_EVALS, num_slots=DIFFWAM_SLOTS) -> dict:
    passes = 2 if kwargs.get("guidance_scale", 5.0) > 1.0 else 1
    t0 = time.perf_counter()
    with WanFeatureExtractor(pipe.transformer, layers, evals, passes_per_step=passes) as ext:
        ext.run(lambda: pipe(**kwargs, output_type="latent"))
    feats = to_diffwam_slots(ext.result(), num_slots)
    feats["total_seconds"] = time.perf_counter() - t0
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=list(WAN_LAYERS))
    parser.add_argument("--evals", type=int, nargs="+", default=list(DIFFWAM_EVALS))
    parser.add_argument("--slots", type=int, default=DIFFWAM_SLOTS)
    parser.add_argument("--out", default="outputs/features.pt")
    args = parser.parse_args()

    prompt, image = resolve_request(args)
    t0 = time.perf_counter()
    pipe = load_pipeline(args.repo)
    print(f"loaded pipeline in {time.perf_counter() - t0:.1f}s")
    feats = extract(pipe, pipe_kwargs(args, prompt, image), tuple(args.layers), tuple(args.evals), args.slots)
    feats["request"] = {k: v for k, v in vars(args).items() if k != "out"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(feats, args.out)
    print(f"saved {args.out}")
    print(f"  Z  {tuple(feats['Z'].shape)}  (K, L, T, H, W, C)   A0 {tuple(feats['A0'].shape)}")
    print(f"  layout {feats['layout']}, slots {len(feats['slot_time_s'])} (native {len(feats['slot_time_native_s'])})")
    for k, tz in zip(feats["evals"], feats["timestep_Z"]):
        print(f"  step {k}: target timestep {tz.max().item():.1f}/1000")
    print(f"  total {feats['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
