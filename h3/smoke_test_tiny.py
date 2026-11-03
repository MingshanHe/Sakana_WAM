"""CPU smoke test of `H3FeatureExtractor` with a tiny random-weight MiniMax-H3 transformer.

No weights are downloaded. The packed sequence, the per-row timestep plan and the scheduler come from the real
diffusers MiniMax-H3 code, and the denoise loop mirrors `MiniMaxH3LoopDenoiser` + `MiniMaxH3LoopSchedulerStep`.
It checks that:
  1. the hooked features equal the block outputs of an un-hooked reference run,
  2. evaluations 0..K-2 run to completion and the last one stops right after the deepest tap,
  3. Z / A0 come out as (K, L, T, H, W, C) / (K, L, 1, H, W, C) with the right per-token timesteps.
"""

import torch
from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)

from h3_features import H3FeatureExtractor


torch.manual_seed(0)
LAYERS, EVALS, NUM_BLOCKS = (15, 25, 35), (0, 1, 2, 3), 40
NUM_LATENT_FRAMES, LATENT_H, LATENT_W, NUM_TEXT, NUM_AUDIO = 7, 8, 12, 10, 20
KEYFRAME_NOISE_AUG = 0.02  # stand-in for `components.keyframe_noise_aug`

transformer = MiniMaxH3Transformer3DModel(
    num_attention_heads=2,
    attention_head_dim=32,
    hidden_size=64,
    num_layers=NUM_BLOCKS,
    num_refiner_layers=1,
    ffn_dim=128,
    text_dim=32,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=32,
    rope_freq_dim=4,
).eval()
cfg = transformer.config

position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond, _ = (
    MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(NUM_TEXT, dtype=torch.long),
        NUM_LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        NUM_AUDIO,
        tuple(cfg.patch_size),
        audio_channels=2,
        audio_tag=2,
        video_tag=0,
        keyframe_anchors=("first",),
    )
)
position_ids = position_ids.float()
patch_dim = cfg.in_channels * 4
latents0 = torch.randn(video_indices.numel(), patch_dim)
audio0 = torch.randn(audio_indices.numel(), cfg.audio_in_channels)
prompt_embeds = torch.randn(1, NUM_TEXT, cfg.text_dim)


def denoise(num_steps=50, block_outputs=None):
    """Mirror of the pipeline loop; optionally records every video-row block output via plain hooks."""
    scheduler, audio_scheduler = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    scheduler.set_timesteps(num_steps)
    audio_scheduler.set_timesteps(num_steps)
    latents, audio = latents0.clone(), audio0.clone()
    handles = []
    if block_outputs is not None:
        state = {"i": -1}
        handles.append(transformer.register_forward_pre_hook(lambda *a: state.__setitem__("i", state["i"] + 1)))
        for l in LAYERS:
            handles.append(
                transformer.transformer_blocks[l].register_forward_hook(
                    lambda m, a, o, l=l: block_outputs.__setitem__((state["i"], l), o[0, video_indices].clone())
                )
            )
    try:
        for i, (t, ta) in enumerate(zip(scheduler.timesteps, audio_scheduler.timesteps)):
            unique_t, t_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                video_indices, audio_indices, num_cond, 0, NUM_TEXT,
                float(t), float(ta), max(float(t), KEYFRAME_NOISE_AUG), 1.0,
            )
            with torch.no_grad():
                v, va = transformer(
                    hidden_states=latents[None], audio_hidden_states=audio[None], encoder_hidden_states=prompt_embeds,
                    timestep=unique_t, timestep_indices=t_idx, token_tags=token_tags, position_ids=position_ids,
                    video_indices=video_indices, audio_indices=audio_indices, text_indices=text_indices,
                    return_dict=False,
                )
            latents[num_cond:] = scheduler.step(v[0, num_cond:].float(), t, latents[num_cond:], return_dict=False)[0]
            audio = audio_scheduler.step(va[0].float(), ta, audio, return_dict=False)[0]
            if block_outputs is not None and i == EVALS[-1]:
                break
    finally:
        for h in handles:
            h.remove()
    return scheduler


# Reference: un-hooked run, recording block outputs with plain hooks.
reference = {}
scheduler = denoise(block_outputs=reference)

# Count how many blocks actually execute during the extractor run.
executed = []
counters = [
    b.register_forward_hook(lambda m, a, o, j=j: executed.append(j)) for j, b in enumerate(transformer.transformer_blocks)
]
with H3FeatureExtractor(transformer, layers=LAYERS, evals=EVALS, store_dtype=torch.float32) as ext:
    ext.run(denoise)
for c in counters:
    c.remove()
feats = ext.result()

# 1. Feature values.
for k in EVALS:
    for li, l in enumerate(LAYERS):
        flat = torch.cat([feats["A0"][k, li].flatten(0, 2), feats["Z"][k, li].flatten(0, 2)])
        torch.testing.assert_close(flat, reference[(k, l)].float(), rtol=0, atol=0)

# 2. Early stop: 3 full evaluations, then blocks 0..35 of the 4th.
assert len(executed) == 3 * NUM_BLOCKS + LAYERS[-1] + 1, len(executed)
assert executed[-1] == LAYERS[-1]
assert ext.eval_idx == EVALS[-1]

# 3. Shapes and timesteps.
H, W = LATENT_H // 2, LATENT_W // 2
assert feats["Z"].shape == (len(EVALS), len(LAYERS), NUM_LATENT_FRAMES, H, W, cfg.hidden_size), feats["Z"].shape
assert feats["A0"].shape == (len(EVALS), len(LAYERS), 1, H, W, cfg.hidden_size), feats["A0"].shape
for k in EVALS:
    t = float(scheduler.timesteps[k])
    assert torch.allclose(feats["timestep_Z"][k], torch.tensor(t)), (k, feats["timestep_Z"][k].unique(), t)
    assert torch.allclose(feats["timestep_A0"][k], torch.tensor(max(t, KEYFRAME_NOISE_AUG)))

# DiffWAM-Flash: only eval 0, stops in the first forward.
executed.clear()
with H3FeatureExtractor(transformer, evals=(0,)) as flash:
    flash.run(denoise)
assert flash.eval_idx == 0 and flash.result()["Z"].shape[0] == 1

print("Z ", tuple(feats["Z"].shape), " A0", tuple(feats["A0"].shape))
print("target timesteps per eval:", [round(float(scheduler.timesteps[k]), 5) for k in EVALS])
print("layout:", feats["layout"])
print("OK: features match reference, early stop after block", LAYERS[-1], "of eval", EVALS[-1])
