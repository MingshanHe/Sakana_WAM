# sakana_dreamer/h3 — MiniMax-H3 feature extraction (DiffWAM-style)

**Timing report (features vs. whole decoded video): [REPORT.md](REPORT.md)**

Reads intermediate DiT features out of the frozen **MiniMax-H3** video model the way
**DiffWAM** (arXiv 2609.39763, *A Fast and Efficient Navigation World Action Model*) does, without generating a video.

| DiffWAM (Sec. 3) | here |
|---|---|
| inputs: first RGB frame `I0` + instruction `c` + noise | `--image` (fl2va) + `--prompt` + `--seed` |
| 4 backbone evaluations, schedule indices K = {0,1,2,3}; first 3 complete, 4th stops after block 35 | `--evals 0 1 2 3` (default); hook raises after the deepest tap |
| DiffWAM-Flash: K = {0} | `--evals 0` |
| taps: DiT blocks 15 / 25 / 35 (zero-based), C = 5376 | `--layers 15 25 35` (default) |
| 15 × 26 feature grid | `--height 480 --width 832` → latent 30×52 → patch 2 → 15×26 |
| no scheduler update after the last eval, no VAE decode | the denoise loop is aborted; decode never runs |

Not stated in the paper (so these are choices you may want to vary): the H3 variant (we use the `transformer/`
partition = t2va/fl2va), `num_inference_steps` (default 50; with shift 12 the first 4 steps are all at sigma ≈ 1),
frame count (124 = 5 s → 37 latent frames), and what exactly `A0` is (we return the keyframe-conditioning rows).

## First real run (2026-11-03, 1× H100, CPU offload, `outputs/forest.pt`)

- Input `assets/forest_obs.png` (FlapSim forest crop), 480×832, 124 frames, 50 steps.
- `Z (4, 3, 37, 15, 26, 5376)`, `A0 (4, 3, 1, 15, 26, 5376)`, sequence length 15,648 tokens, 1.9 GB file.
- Time: 695 s total. Eval 0 took 354 s (offload swapping text encoder ↔ transformer); evals 1–3 took ~2 s each.
  `--placement multi` (2 GPUs) avoids the swap.
- Feature scale: std 290 / 400 / 900 at blocks 15 / 25 / 35 with a few huge outlier channels (|x| up to 9e5),
  so normalise per channel before any head (DiffWAM uses a per-layer norm + projection 5376→512).
- PCA (`outputs/forest_pca.png`): `A0` clearly shows the pillars; `Z` is mostly noise-like. That is expected:
  with the shift-12 schedule the first four evaluations sit at sigma 1.000 / 0.998 / 0.996 / 0.995 (50 steps),
  and even at 6 steps only reach 0.889. The target rows are still almost pure noise, so their information comes from
  attention to the keyframe and text — which is what the trained readout must learn to decode.

## Test run on `assets/test.jpg` (2026-11-03, gpu1 H100, `run_test.sh`)

- Input center-cropped 4:3 → 16:9, 832×480; prompt `assets/test_instruction`; 124 frames (5.17 s), 50 steps, seed 0.
- Features: `Z (4, 3, 39, 15, 26, 5376)`, `A0 (4, 3, 1, 15, 26, 5376)`; 68 s with warm page cache
  (eval 0: 46 s incl. offload swap, evals 1–3: ~2 s each).
- Whole video (same request + seed, `generate_video.py`): 157 s. Keeps the room layout and ends in a hover at the
  whiteboard, but flies *through* the centre felt panel around 3.5–4.5 s — it does not respect solid obstacles.

## Files

- `h3_features.py` — `H3FeatureExtractor` (hooks + early stop + reshape) and a CLI around the diffusers
  `ModularPipeline`. Output `.pt`:
  - `Z`  `(K, L, 39, H, W, 5376)` target-video tokens (bf16) on DiffWAM's 39 slots `slot_time_s = j·5/38 s`
    (identity + 38 future poses over 5 s). H3 itself gives 37 non-uniform latents for 5 s (`5n+2` is never 39,
    `slot_time_native_s`), linearly resampled in time; `--slots 0` keeps the native 37.
  - `A0` `(K, L, 1, H, W, 5376)` first-frame keyframe tokens (`None` without `--image`)
  - `timestep_Z`, `timestep_A0` per-token noise level each block saw (H3 timestep = 1 − sigma, 0 = pure noise)
  - `layout`, `eval_seconds`, `total_seconds`, `request`
- `generate_video.py` — runs the *same* request (image, prompt, canvas, steps, seed) to the end and VAE-decodes it:
  the full video the features were taken from (DiffWAM's teacher rollout). Writes `.mp4` (with H3's generated audio),
  `_input.png` (the exact first frame fed to H3) and `_frames.png` (contact sheet).
- `run_test.sh` — end-to-end test on `assets/test.jpg` + `assets/test_instruction`: smoke test → features + PCA →
  whole video. Inputs are center-cropped to 16:9 (H3 would otherwise stretch a 4:3 photo onto the 832×480 canvas).
- `smoke_test_tiny.py` — CPU test with a tiny random H3 transformer + the real layout/scheduler code. Checks the hooked
  features are bit-identical to an un-hooked run and that exactly 3×50 + 36 blocks execute. **Passes.**
- `inspect_features.py` — stats per (eval, layer) + PCA→RGB image of every token grid.
- `download_h3.sh` — downloads only the fl2va/t2va subfolders (~143 GB of the 498 GB repo) to the HF cache.
- `run_extract.sh` — GPU job wrapper.

Environment: conda env `sakana_dreamer` (`conda activate sakana_dreamer`; Python 3.11 from conda-forge, torch 2.11 cu128, diffusers 0.42.0.dev0 from git main — H3 is only on main).

## Run

```bash
# the whole test (features + decoded video) on the repo's test image, from a login node:
srun -p a3 --gres=gpu:1 -c 16 --mem=300G -t 1:30:00 bash ~/sakana_dreamer/h3/run_test.sh
# outputs/test_features.pt, test_features_pca.png, test_video.mp4, test_video_frames.png, test_video_input.png
```

```bash
cd ~/sakana_dreamer/h3
~/miniconda3/envs/sakana_dreamer/bin/python smoke_test_tiny.py            # no GPU / weights needed

bash download_h3.sh                                 # ~143 GB, once (license already accepted on HF)

# Interactive, inside an `srun` shell with 1 GPU (needs ~130 GB host RAM for the offload; gpu1 gives ~227 GB):
conda activate sakana_dreamer
python h3_features.py --image assets/forest_obs.png \
  --prompt 'First-person drone view flying forward between red pillars' --out outputs/forest.pt

# 2× H100, no offload (text encoder + VAEs on cuda:0, transformer on cuda:1)
sbatch -p a3 --gres=gpu:2 -c 16 --mem=400G -t 2:00:00 -o ~/sakana_dreamer/h3/logs/%j.out \
  --wrap "bash ~/sakana_dreamer/h3/run_extract.sh --placement multi --image ~/sakana_dreamer/h3/assets/obs.png \
          --prompt 'Fly forward through the corridor and turn left at the door'"
# 1× H100: drop --placement multi (CPU offload, slower), use --gres=gpu:1

~/miniconda3/envs/sakana_dreamer/bin/python inspect_features.py outputs/features_<jobid>.pt
```

Python use:

```python
from h3_features import load_pipeline, extract
pipe = load_pipeline("multi")
feats = extract(pipe, "fly to the red door", image=img, evals=(0,))   # DiffWAM-Flash
```
