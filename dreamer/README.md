# dreamer/ — DreamerV3 local planner under the Wan2.2 / DiffWAM global planner

Report (English + 中文): [`../results/wan2.2-ti2v-5b/dreamer_local_planner/README.md`](../results/wan2.2-ti2v-5b/dreamer_local_planner/README.md).

| File | Env | What |
|---|---|---|
| `env.py` | sakana_hyperdrone | batched plan-following env: L2F dynamics (RAPTOR position control), 64x64 RGB-D camera, 6-view crash rig, plan waypoints, reward |
| `plans.py` | sakana_hyperdrone | Wan2.2 gen2pose teacher plans, 12 synthetic motion primitives, instruction-prior plans |
| `collect.py` | sakana_hyperdrone | data collection in the 80 diffwam training houses (track / random / safe-track behaviour) |
| `dv3/nets.py`, `dv3/agent.py` | sakana_hyperdrone | DreamerV3 in PyTorch (size12m ... size400m presets; default size200m) |
| `train.py` | sakana_hyperdrone | world-model / actor-critic training, held-out-house validation with open-loop checks, resumable |
| `policy.py` | sakana_hyperdrone | deployed Dreamer policy, optional world-model shield |
| `deploy_combined.py` | sakana_hyperdrone | 20 test cases x 5 arms in HyperDrone; `--replan-every` for live Wan replanning |
| `wan_server.py` | sakana_dreamer | Wan2.2 DiffWAM feature server (directory requests) |
| `replan.py` | sakana_hyperdrone | client: request features, DiffWAM head -> plan |
| `make_figures.py` | sakana_hyperdrone | report figures |
| `collect_all.sh`, `run_wm.sh`, `quick_check3.sh` | | job scripts (full collection, world-model run, quick end-to-end check) |
| `probe_l2f.py` | sakana_hyperdrone | probe of the L2F position controller |

`data/` (replay 32 GB, checkpoints, deploy videos) and `outputs/` are git-ignored. Run GPU steps inside a Slurm
allocation with `export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH`.
