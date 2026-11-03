# diffwam/ — DiffWAM pipeline on Wan2.2-TI2V-5B + HyperDrone

Report: [`../results/wan2.2-ti2v-5b/diffwam_gen2pose/README.md`](../results/wan2.2-ti2v-5b/diffwam_gen2pose/README.md).

| File | Env | What |
|---|---|---|
| `make_starts.py` | sakana_hyperdrone | stage A: take-off poses + I0 + metric depth in ProcTHOR houses (eval houses excluded), instruction templates |
| `fit_feature_pca.py` | sakana_hyperdrone | per-layer standardisation + PCA 3072 → 128 for storing features |
| `wan_worker.py` | sakana_dreamer | stage B: DiffWAM features (steps 0-3, blocks 9/15/21, conditional pass) + completed rollout from the same noise; sharded, resumable, `--gpu` |
| `gen2pose.py` | sakana_hyperdrone | teacher: metric camera trajectory from a rollout whose first frame has known depth (KLT + PnP + triangulation) |
| `label_teacher.py` | sakana_hyperdrone | stage C: run gen2pose on every finished rollout |
| `validate_gen2pose.py` | sakana_hyperdrone | gen2pose vs ground truth on rendered flights |
| `make_eval_samples.py` | | the 10 evaluation scenarios × 2 seeds as test samples |
| `train_head.py` | sakana_hyperdrone | DiffWAM-style readout head (`--evals 0` Flash, `--zero-future` scene-only control) + hover / instruction-prior baselines |
| `deploy.py` | sakana_hyperdrone | closed loop: predicted poses (fast) and teacher poses (slow) flown in HyperDrone |
| `plot_results.py`, `collect_results.py` | sakana_hyperdrone | figure + copy report assets |

Data (~32 GB) is git-ignored. Cluster notes: launch GPU steps with `--gres=gpu:1` (a step that sees two GPUs gets a
different hyperlux JIT key and triggers a fresh CMake configure, which can hang); within one job, pin extra Wan workers
with `--gpu N` (Slurm overrides `CUDA_VISIBLE_DEVICES` from `--export`).
Pi3 (DiffWAM's reconstruction teacher) was not used: installing its code from GitHub was blocked by the permission
policy; its weights are in the HF cache (`yyfz233/Pi3`) if that is approved later.
