# vla/ — a VLA (OpenFly-Agent v1) as the global planner

Results are in the main report, section 4: [`../results/wan2.2-ti2v-5b/dreamer_local_planner/README.md`](../results/wan2.2-ti2v-5b/dreamer_local_planner/README.md).
Details are in [`../results/wan2.2-ti2v-5b/vla_zeroshot/README.md`](../results/wan2.2-ti2v-5b/vla_zeroshot/README.md).

| File | Env | What |
|---|---|---|
| `openfly_server.py` | sakana_vla | OpenFly-Agent v1 (7B) action server (directory requests); training-format prompt, frame order and nearest-action decoding |
| `eval_zeroshot.py` | sakana_hyperdrone | OpenFly alone vs Wan2.2 + DiffWAM on the 20 test cases (kinematic, OpenFly's own step protocol) |
| `rescore.py` | sakana_hyperdrone | re-score with the strict 6-view crash rule and unwrapped yaw |
| `probe_openfly.py` | sakana_vla | prompt / format probe on single frames |
| `run_zeroshot.sh`, `run_openfly_dreamer.sh` | | job scripts; the second runs OpenFly every 1 s + Dreamer via `../dreamer/deploy_combined.py` |

Setup: `git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform ~/vla/OpenFly-Platform`, which provides the model
classes. Then create conda env `sakana_vla` with Python 3.10, torch 2.5.1 (cu124), transformers 4.47.1, timm 0.9.16 and
huggingface_hub < 1. Run it with `PYTHONNOUSERSITE=1`, because `~/.local` has a newer huggingface_hub.
