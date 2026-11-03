# sakana_dreamer/wan — Wan2.2-TI2V-5B (DiffWAM-style features)

The small counterpart of `../h3/`: the same DiffWAM read-out on
[`Wan-AI/Wan2.2-TI2V-5B-Diffusers`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) (5B DiT, 34 GB of weights,
fits on one H100 with no offload). Comparison with H3: [`../results/README.md`](../results/README.md).

- `wan_features.py` — `WanFeatureExtractor`. Taps blocks 9 / 15 / 21 (the same relative depth as H3's 15/25/35) at
  steps 0–3, from the **conditional** CFG pass only. Steps 0–2 also run the unconditional pass, so the latent follows
  the full-video rollout exactly. Step 3 stops after block 21. `Z (4, 3, 39, 15, 26, 3072)` is resampled to DiffWAM's
  39 slots over 5 s (31 native latents, `--slots 0` keeps them); `A0` = latent frame 0, the clean first-frame latent.
- `benchmark.py` — times DiffWAM / DiffWAM-Flash / full video (+ VAE decode) and saves the video.

```bash
conda activate sakana_dreamer        # needs `ftfy` (installed)
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers
python wan_features.py --image ../h3/assets/test.jpg --prompt-file ../h3/assets/test_instruction --out outputs/test_features.pt
python benchmark.py   --image ../h3/assets/test.jpg --prompt-file ../h3/assets/test_instruction
```
