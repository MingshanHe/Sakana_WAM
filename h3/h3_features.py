"""Extract DiffWAM-style predictive features from the frozen MiniMax-H3 video model.

DiffWAM (arXiv 2609.39763, Sec. 3) never generates a video. Given an initial RGB observation and an instruction,
it runs a few evaluations of the frozen H3 DiT and reads hidden states out of DiT blocks 15 / 25 / 35:

  * schedule indices K = {0, 1, 2, 3}: the first three evaluations run to completion (so the scheduler advances
    the latent), the fourth stops right after block 35. DiffWAM-Flash uses K = {0}.
  * no further scheduler updates and no VAE decode.

This module reproduces that read-out on top of the diffusers MiniMax-H3 modular pipeline: the pipeline does the
prompt / keyframe encoding, the packed-sequence layout and the scheduler steps, while forward hooks on the
transformer record the block outputs and abort the denoise loop after the last needed block.

The H3 transformer runs one packed sequence `[text | keyframe rows | audio | target video]` (see
`MiniMaxH3PrepareLayoutStep.build_packed_sequence`). The hooks keep only the video rows and split them into

  * `Z`  - target (predicted-future) video tokens, (K, L, T, H, W, C)
  * `A0` - keyframe-conditioning tokens (the clean observation anchor), (K, L, N_keyframes, H, W, C)

with C = 5376, H x W = latent grid / 2 (patch 1x2x2), e.g. 15 x 26 for a 480 x 832 input.
"""

import argparse
import logging
import time
from contextlib import contextmanager

import torch


DIFFWAM_LAYERS = (15, 25, 35)
DIFFWAM_EVALS = (0, 1, 2, 3)
# DiffWAM (Sec. 3.1, Eq. 6): identity pose + 38 future poses over a nominal 5 s horizon, T_v = 39 video-time slots.
DIFFWAM_SLOTS, DIFFWAM_HORIZON_S = 39, 5.0
H3_FPS = 24
# Pixel frames packed into each latent frame, repeating per 17-frame VAE chunk (diffusers `_ROPE_FRAMES_PER_LATENT`).
H3_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
REPO_ID = "MiniMaxAI/MiniMax-H3"


class _StopAfterLastTap(Exception):
    """Raised from a block hook once every requested feature is collected, to abort the denoise loop."""


def _video_grid(position_ids: torch.Tensor) -> tuple[int, int]:
    """Number of distinct (h, w) rotary coordinates of one frame's rows."""
    n_h = torch.unique(position_ids[:, 1]).numel()
    n_w = torch.unique(position_ids[:, 2]).numel()
    return n_h, n_w


class H3FeatureExtractor:
    r"""
    Hooks a `MiniMaxH3Transformer3DModel` to record DiT block outputs at the given evaluations, and stops the forward
    after the deepest requested block of the last requested evaluation.

    Use as a context manager around anything that calls the transformer (the full pipeline, or a hand-written loop):

        with H3FeatureExtractor(pipe.transformer) as ext:
            ext.run(lambda: pipe(prompt=..., image=..., output=["videos"]))
        feats = ext.result()
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        layers: tuple[int, ...] = DIFFWAM_LAYERS,
        evals: tuple[int, ...] = DIFFWAM_EVALS,
        store_dtype: torch.dtype = torch.bfloat16,
        store_device: str | torch.device = "cpu",
    ):
        num_blocks = len(transformer.transformer_blocks)
        if not layers or max(layers) >= num_blocks or min(layers) < 0:
            raise ValueError(f"`layers` must be in [0, {num_blocks}), got {layers}.")
        if not evals or min(evals) < 0:
            raise ValueError(f"`evals` must be non-negative, got {evals}.")
        self.transformer = transformer
        self.layers = tuple(sorted(set(layers)))
        self.evals = tuple(sorted(set(evals)))
        self.store_dtype = store_dtype
        self.store_device = store_device
        self._handles = []
        self.reset()

    def reset(self):
        self.eval_idx = -1
        self.features: dict[tuple[int, int], torch.Tensor] = {}
        self.row_timesteps: dict[int, torch.Tensor] = {}
        self.eval_times: list[float] = []
        self.layout = None
        self.finished = False

    # -- hooks --------------------------------------------------------------------------------------------------

    def _pre_hook(self, module, args, kwargs):
        self.eval_idx += 1
        self.eval_times.append(time.perf_counter())
        if self.layout is None:
            video_indices = kwargs["video_indices"]
            position_ids = kwargs["position_ids"].index_select(0, video_indices).cpu()
            vi = video_indices.cpu()
            # `video_indices` is `[keyframe rows] ++ [target rows]`, two contiguous runs separated by the audio rows.
            breaks = torch.nonzero(vi.diff() != 1).flatten()
            num_cond = int(breaks[0]) + 1 if breaks.numel() else 0
            if breaks.numel() > 1:
                raise NotImplementedError("Only the t2va / fl2va layout (keyframes then target video) is supported.")
            n_h, n_w = _video_grid(position_ids[num_cond:])
            rows_per_frame = n_h * n_w
            num_target = vi.numel() - num_cond
            if num_target % rows_per_frame or num_cond % rows_per_frame:
                raise RuntimeError(f"Video rows ({num_cond} + {num_target}) are not whole {n_h}x{n_w} frames.")
            # Rows are frame-major then h-major; check it so the reshape below is a real (T, H, W) grid.
            pos = position_ids[num_cond:].view(-1, n_h, n_w, 3)
            if not (pos[:, :, 1:, 2] > pos[:, :, :-1, 2]).all() or not (pos[:, 1:, :, 1] > pos[:, :-1, :, 1]).all():
                raise RuntimeError("Unexpected video row order in the packed sequence.")
            self.layout = dict(
                video_indices=video_indices,
                num_cond_rows=num_cond,
                num_cond_frames=num_cond // rows_per_frame,
                num_frames=num_target // rows_per_frame,
                grid_h=n_h,
                grid_w=n_w,
                sequence_length=int(kwargs["position_ids"].shape[0]),
            )
        if self.eval_idx in self.evals:
            timestep, timestep_indices = kwargs["timestep"], kwargs["timestep_indices"]
            rows = timestep_indices.index_select(0, self.layout["video_indices"].to(timestep_indices.device))
            self.row_timesteps[self.eval_idx] = timestep.float()[rows].cpu()

    def _make_block_hook(self, layer: int):
        last_eval, last_layer = self.evals[-1], self.layers[-1]

        def hook(module, args, output):
            if self.eval_idx not in self.evals:
                return
            video_indices = self.layout["video_indices"].to(output.device)
            self.features[(self.eval_idx, layer)] = (
                output[0].index_select(0, video_indices).to(self.store_device, self.store_dtype)
            )
            if self.eval_idx == last_eval and layer == last_layer:
                self.finished = True
                raise _StopAfterLastTap

        return hook

    def __enter__(self):
        self.reset()
        self._handles.append(self.transformer.register_forward_pre_hook(self._pre_hook, with_kwargs=True))
        for layer in self.layers:
            block = self.transformer.transformer_blocks[layer]
            self._handles.append(block.register_forward_hook(self._make_block_hook(layer)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    # -- driving ------------------------------------------------------------------------------------------------

    def run(self, fn):
        """Call `fn` (which drives the transformer) and swallow the early-stop exception."""
        try:
            with _quiet_modular_logs():
                fn()
        except _StopAfterLastTap:
            pass
        self.eval_times.append(time.perf_counter())
        if not self.finished:
            raise RuntimeError(
                f"The transformer ran {self.eval_idx + 1} evaluations, fewer than the {self.evals[-1] + 1} needed; "
                "raise `num_inference_steps`."
            )

    def result(self) -> dict:
        lay = self.layout
        n_cond, t, h, w = lay["num_cond_rows"], lay["num_frames"], lay["grid_h"], lay["grid_w"]
        z, a0, tz, ta = [], [], [], []
        for k in self.evals:
            per_layer = [self.features[(k, l)] for l in self.layers]
            z.append(torch.stack([f[n_cond:].view(t, h, w, -1) for f in per_layer]))
            if n_cond:
                a0.append(torch.stack([f[:n_cond].view(lay["num_cond_frames"], h, w, -1) for f in per_layer]))
            tz.append(self.row_timesteps[k][n_cond:].view(t, h, w))
            if n_cond:
                ta.append(self.row_timesteps[k][:n_cond].view(lay["num_cond_frames"], h, w))
        return {
            "Z": torch.stack(z),  # (K, L, T, H, W, C)
            "A0": torch.stack(a0) if a0 else None,  # (K, L, N_keyframes, H, W, C); None without a keyframe
            "timestep_Z": torch.stack(tz),  # (K, T, H, W): per-token noise level the block saw
            "timestep_A0": torch.stack(ta) if ta else None,
            "layers": list(self.layers),
            "evals": list(self.evals),
            "layout": {k: v for k, v in lay.items() if k != "video_indices"},
            "eval_seconds": [b - a for a, b in zip(self.eval_times, self.eval_times[1:])],
        }


@contextmanager
def _quiet_modular_logs():
    # Every modular block logs a full traceback before re-raising; the early stop is not an error.
    logger = logging.getLogger("diffusers.modular_pipelines.modular_pipeline")
    level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(level)


# ---------------------------------------------------------------------------------------------------------------
# Pipeline loading


def load_pipeline(placement: str = "offload", repo: str = REPO_ID, attention_backend: str | None = None):
    r"""
    `offload`: one GPU, ComponentsManager auto CPU offload (the docs' recipe; text encoder and transformer swap in
    and out, ~125GB host RAM). `multi`: no offload, text encoder + VAEs on cuda:0 and the transformer on cuda:1.
    """
    from diffusers import ComponentsManager, ModularPipeline

    if placement == "offload":
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        pipe = ModularPipeline.from_pretrained(repo, components_manager=manager)
        pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
    elif placement == "multi":
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module

        pipe = ModularPipeline.from_pretrained(repo)
        pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
        for name in ("text_encoder", "vae", "audio_vae"):
            getattr(pipe, name).to("cuda:0")
        # The transformer is only ever called through `forward`, so an io_same_device hook is enough to run it on
        # its own card while the pipeline keeps its tensors on cuda:0.
        pipe.transformer.to("cuda:1")
        add_hook_to_module(pipe.transformer, AlignDevicesHook(execution_device="cuda:1", io_same_device=True))
    else:
        raise ValueError(f"Unknown placement {placement!r}.")

    # `load_components` logs a failed load and leaves the component as None instead of raising.
    missing = [name for name in ("transformer", "text_encoder", "vae") if getattr(pipe, name, None) is None]
    if missing:
        raise RuntimeError(f"Failed to load {missing}; see the log above.")

    if attention_backend:
        pipe.transformer.set_attention_backend(attention_backend)
    return pipe


def latent_slot_times(num_latent_frames: int, fps: float = H3_FPS) -> torch.Tensor:
    """Time [s] of each H3 target latent frame: the centre of the pixel frames it packs (1, 4, 4, 4, 4 per chunk)."""
    times, frame = [], 0
    for i in range(num_latent_frames):
        span = H3_FRAMES_PER_LATENT[i % len(H3_FRAMES_PER_LATENT)]
        times.append((frame + (span - 1) / 2) / fps)
        frame += span
    return torch.tensor(times)


def resample_slots(z: torch.Tensor, src_t: torch.Tensor, dst_t: torch.Tensor, dim: int = 2) -> torch.Tensor:
    """Linearly interpolate `z` along `dim` from slot times `src_t` to `dst_t` (clamped to the source range)."""
    dst_t = dst_t.clamp(src_t[0], src_t[-1])
    hi = torch.searchsorted(src_t, dst_t).clamp(1, len(src_t) - 1)
    lo = hi - 1
    w = ((dst_t - src_t[lo]) / (src_t[hi] - src_t[lo])).to(torch.float32)
    shape = [1] * z.ndim
    shape[dim] = -1
    w = w.view(shape)
    out = z.index_select(dim, lo).float() * (1 - w) + z.index_select(dim, hi).float() * w
    return out.to(z.dtype)


def to_diffwam_slots(feats: dict, num_slots: int = DIFFWAM_SLOTS, horizon_s: float = DIFFWAM_HORIZON_S) -> dict:
    """Resample Z (and its timesteps) from H3's native latent slots onto `num_slots` uniform times over the horizon.

    H3 gives 37 target latents for a 5 s request (5n + 2 is never 39), non-uniformly spaced; DiffWAM reads 39
    slots aligned with its 39 poses at t = j * horizon / 38. Pass `num_slots=0` to keep the native slots.
    """
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


def extract(
    pipe,
    prompt: str,
    image=None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 124,
    num_inference_steps: int = 50,
    layers: tuple[int, ...] = DIFFWAM_LAYERS,
    evals: tuple[int, ...] = DIFFWAM_EVALS,
    seed: int = 0,
    num_slots: int = DIFFWAM_SLOTS,
) -> dict:
    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
        output=["videos"],
    )
    if image is not None:
        kwargs["image"] = image
    t0 = time.perf_counter()
    with H3FeatureExtractor(pipe.transformer, layers=layers, evals=evals) as ext:
        ext.run(lambda: pipe(**kwargs))
    feats = to_diffwam_slots(ext.result(), num_slots)
    feats["total_seconds"] = time.perf_counter() - t0
    feats["request"] = {k: v for k, v in kwargs.items() if k not in ("generator", "image", "output")}
    feats["request"]["seed"] = seed
    feats["request"]["has_image"] = image is not None
    return feats


def prepare_image(path: str, height: int, width: int):
    """Load an image and center-crop it to the canvas aspect ratio, then resize.

    The H3 pipeline *stretches* a keyframe onto the canvas, so a 4:3 photo on the 16:9 DiffWAM canvas (480x832) would
    be distorted; cropping first keeps the scene geometry.
    """
    from diffusers.utils import load_image

    image = load_image(path)
    src_w, src_h = image.size
    if src_w * height > src_h * width:  # too wide
        crop_w = round(src_h * width / height)
        left = (src_w - crop_w) // 2
        image = image.crop((left, 0, left + crop_w, src_h))
    else:  # too tall
        crop_h = round(src_w * height / width)
        top = (src_h - crop_h) // 2
        image = image.crop((0, top, src_w, top + crop_h))
    return image.resize((width, height), resample=3)  # bicubic


def add_request_args(parser: argparse.ArgumentParser):
    """CLI arguments shared by the feature extractor and the full-video generator."""
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", help="Text file holding the prompt, e.g. assets/test_instruction.")
    parser.add_argument("--image", help="Initial RGB observation (fl2va). Omit for text-only (t2va).")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124, help="Snapped to 17n+5; 124 = 5 s at 24 fps.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--placement", choices=["offload", "multi"], default="offload")
    parser.add_argument("--attention-backend", default=None, help='e.g. "_flash_3_hub" on H100')
    parser.add_argument("--repo", default=REPO_ID)


def resolve_request(args) -> tuple[str, object]:
    """(prompt, image) from parsed `add_request_args` arguments."""
    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    image = prepare_image(args.image, args.height, args.width) if args.image else None
    return prompt, image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_request_args(parser)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DIFFWAM_LAYERS))
    parser.add_argument("--evals", type=int, nargs="+", default=list(DIFFWAM_EVALS), help="DiffWAM-Flash: --evals 0")
    parser.add_argument(
        "--slots", type=int, default=DIFFWAM_SLOTS,
        help="Resample Z to this many uniform slots over 5 s (DiffWAM T_v = 39); 0 keeps H3's native 37 latents.",
    )
    parser.add_argument("--out", default="features.pt")
    args = parser.parse_args()

    prompt, image = resolve_request(args)
    t0 = time.perf_counter()
    pipe = load_pipeline(args.placement, args.repo, args.attention_backend)
    print(f"loaded pipeline in {time.perf_counter() - t0:.1f}s")

    feats = extract(
        pipe,
        prompt,
        image=image,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        layers=tuple(args.layers),
        evals=tuple(args.evals),
        seed=args.seed,
        num_slots=args.slots,
    )
    torch.save(feats, args.out)

    print(f"saved {args.out}")
    print(f"  Z  {tuple(feats['Z'].shape)}  (K, L, T, H, W, C)")
    t = feats["slot_time_s"]
    print(f"  slots: {len(t)} at {t[0]:.3f}..{t[-1]:.3f} s (native H3 latents: {len(feats['slot_time_native_s'])})")
    if feats["A0"] is not None:
        print(f"  A0 {tuple(feats['A0'].shape)}  (K, L, N_keyframes, H, W, C)")
    print(f"  layout {feats['layout']}")
    for k, tz in zip(feats["evals"], feats["timestep_Z"]):
        print(f"  eval {k}: target timestep {tz.unique().tolist()}")
    print(f"  per-eval seconds {[round(s, 2) for s in feats['eval_seconds']]}, total {feats['total_seconds']:.1f}s")


if __name__ == "__main__":
    main()
