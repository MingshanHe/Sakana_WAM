# follow/ — fly along an imagined video by matching latents

Results and discussion: [`../results/wan2.2-ti2v-5b/scenarios10/README.md`](../results/wan2.2-ti2v-5b/scenarios10/README.md).

| File | Env | What |
|---|---|---|
| `scenes.py` | sakana_hyperdrone | `HouseCamera`: batched level camera in a ProcTHOR house (hyperlux OptiX), take-off search |
| `make_scenarios.py` | sakana_hyperdrone | take-off pose + first frame I0 per house |
| `scenarios/scenarios.json` | | the 10 scenarios: house, start pose, motion type, Wan instruction (v2 camera-move style) |
| `imagine_scenarios.py` | sakana_dreamer | Wan2.2: Flash features + full video from the same noise, per scenario × seed |
| `prompt_probe.py` | sakana_dreamer | compare prompt wordings on one I0 by optical-flow motion |
| `analyze_imagination.py` | sakana_dreamer | motion amount + direction vs instruction for every imagination |
| `latents.py` | sakana_hyperdrone | Flash projector (norm + PCA), `FlashEncoder` (ResNet-18), `WanVAE` single-frame latents |
| `train_encoder.py` | sakana_hyperdrone | scenario-held-out folds (`--split seed`: noise check) |
| `follow.py` | sakana_hyperdrone | rendering-in-the-loop latent-error controller; replay (ground truth) and wan tasks |
| `analyze_follow.py`, `collect_results.py` | sakana_hyperdrone | tables + copy report assets to `results/` |

Generated data (imagined videos, Flash features, encoders, follower runs; ~8.5 GB) is git-ignored; see `.gitignore`.

Cluster notes:
- Run hyperlux with the env's `bin` first on `PATH` (CMake ≥ 3.24); `scenes.py` does this itself.
- `scenes.py` skips hyperlux's `cmake --build` for libraries that are already built: on this cluster that step
  sometimes never returns (gmake exits, cmake stays in `ep_poll`; `UV_USE_IO_URING=0` did not help). A new renderer
  configuration still compiles, so reuse 832 × 480 / batch 64 where possible.
