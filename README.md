# sakana_dreamer

Drone world-model experiments: an Isaac Lab drone test scenario with an onboard camera (`sim/`) and MiniMax-H3
predictive-feature extraction in the style of DiffWAM (`h3/`).

```
sakana_dreamer/
├── sim/                     Isaac Lab drone scenario  (env: ~/IsaacLab/.venv, Python 3.12)
│   ├── drone_scene.py       scene cfg: Crazyflie + onboard RGB-D camera + random red-pillar field
│   ├── controller.py        Lee geometric velocity controller -> body wrench (thrust + torque)
│   ├── run_drone_test.py    fly the scenario, record onboard video / depth / pose
│   ├── indoor_scene.py      Isaac Sim indoor USD (office, hospital, ...) made visual-only for Newton
│   ├── survey_indoor.py     render candidate take-off spots in an indoor scene
│   ├── takeoff_capture.py   take off, hover, capture the first onboard frame I0 (832x480) + depth + pose
│   ├── prompts/             instructions for the video world model
│   └── outputs/<run>/       onboard.mp4, obs_first.png, episode.npz
└── h3/                      MiniMax-H3 feature extraction (env: conda `sakana_dreamer`, Python 3.11)
    ├── h3_features.py       H3FeatureExtractor: DiT blocks 15/25/35 at evals 0-3, no video decode
    ├── smoke_test_tiny.py   CPU test with a tiny random H3 transformer
    ├── inspect_features.py  stats + PCA image of the features
    ├── REPORT.md            timing: DiffWAM features vs whole decoded video
    └── README.md            details
├── wan/                     the same DiffWAM read-out on Wan2.2-TI2V-5B (5B, fits one H100, no offload)
│   ├── wan_features.py      WanFeatureExtractor: blocks 9/15/21, conditional CFG pass, steps 0-3
│   ├── benchmark.py         timing + full video
│   ├── imagine.py           I0 + instruction -> Flash / DiffWAM features + the imagined video (same seed)
│   └── sweep_imagine.py     prompt x seed x resolution sweep, scored by optical-flow camera motion
└── results/                 per-model test results on h3/assets/test.jpg
    ├── README.md            H3 vs Wan comparison (timing + video quality)
    ├── h3/                  video, frames, input, PCA, benchmark.json
    └── wan2.2-ti2v-5b/      video, frames, input, benchmark.json
```

Two environments, on purpose: Isaac Sim 6.2 needs Python 3.12 and pins its own torch, while the H3 code needs
diffusers from git main (Python >= 3.11). They talk through files (`obs_first.png`, `episode.npz`).

## Simulator (`sim/`)

Isaac Lab 3.0 in kit-less mode: **Newton** physics (MuJoCo-Warp) and the **Newton Warp** ray-traced camera, imported
as plain Python. No Isaac Sim app, no RTX cores needed, so it runs on the H100 nodes.

- Drone: Bitcraze Crazyflie (`CRAZYFLIE_CFG`, 28 g). The USD has free-spinning props and no thrust model, so
  `controller.py` computes a collective thrust + body torque from velocity commands
  `(v_forward, v_left, z_ref, yaw_rate)` and applies it with `permanent_wrench_composer`.
- Onboard camera: 416×240 RGB + depth, 90° HFOV, 6 cm ahead of the CoM, 25 Hz. Kit-less, the camera prim is not
  carried by the Newton body, so the runner sets its world pose from the body pose every step (`attach_camera`).
- Scene: checkered ground, dome light, 14 random red pillars (static colliders, `--layout-seed`), like the FlapSim forest.
- Baseline behaviour: hold 1 m, fly forward at 1.5 m/s, slow down and yaw towards the freer half of the depth
  image when the centre gets close (`--no-avoid` to disable).

```bash
# on a GPU node (srun -p a3 --gres=gpu:1 ... or inside your srun shell)
cd ~/sakana_dreamer/sim
~/IsaacLab/.venv/bin/python run_drone_test.py --device cuda:0 --duration 8 --out-dir outputs/demo
# larger frames for H3 (16:9, its 480x832 canvas):
~/IsaacLab/.venv/bin/python run_drone_test.py --device cuda:0 --cam-width 832 --cam-height 480 --out-dir outputs/h3in
```

Measured (H100, 1 env): ~25 s solver warm-up on first launch, then 0.9× real time with the camera at 25 Hz.

## Imagined-trajectory following (work in progress)

**Report (English + 中文): [results/wan2.2-ti2v-5b/REPORT_imagined_flight.md](results/wan2.2-ti2v-5b/REPORT_imagined_flight.md)**

Goal: take off in an indoor Isaac scene, give the first onboard frame + an instruction to Wan2.2, use the
DiffWAM-Flash features as a latent goal, train an onboard-camera encoder into that latent space, and train a policy
whose reward is the latent error, so it flies along the imagined video. Baseline: the same error on VAE latents.

| Phase | Status |
|---|---|
| 0. Indoor scene, take-off, capture I0 (`sim/takeoff_capture.py`) | done: office, hover 1.21 m, 832x480 I0 |
| 1. Imagination: Flash features + imagined video (`wan/imagine.py`, `wan/sweep_imagine.py`) | done: official negative prompt + seed sweep gives clear forward motion; picked `approach_s2` 832x480 ([results](results/wan2.2-ti2v-5b/office_imagine/README.md)) |
| 1b. 10 scenarios in HyperDrone ProcTHOR houses (`follow/`) | done: camera-move prompts, 15/20 imaginations move in the instructed direction ([results](results/wan2.2-ti2v-5b/scenarios10/README.md)) |
| 2. Encoder: (imagined frame t, Z[t]) pairs | done, negative for Flash targets: held-out R² −0.14; same scenes with new noise R² 0.03 → step-0 Flash features are noise-dominated |
| 3. Latent-error control (rendering-in-the-loop MPC, kinematic) | done: replay with ground truth 9–10/10 in pixel / VAE / encoder space; following Wan videos with VAE goals 9/10, with Flash goals 2/10 |
| 3b. DiffWAM method (`diffwam/`): features → gen2pose teacher → readout head → fly | done: 450 samples / 80 houses; test ADE 0.52 m vs 0.73 instruction prior; future-slot features carry the signal; flight 13/20 vs 12/20 for the full-video teacher at ~1/10 the cost ([results](results/wan2.2-ti2v-5b/diffwam_gen2pose/README.md)) |
| 3c. DreamerV3 local planner (`dreamer/`): 207M world model + actor-critic on 2.05M real L2F flight frames, flying Wan/DiffWAM plans, live replanning | quick check done (10% data, short training): DiffWAM + Dreamer 0 crashes vs 3 for the tracker, but looser tracking; full run next ([report](results/wan2.2-ti2v-5b/dreamer_local_planner/README.md)) |
| 4. RL policy (PPO), reward = -latent error + progress - crash | superseded by 3c (DreamerV3 actor-critic in imagination, L2F dynamics) |

Notes on the indoor scene (Newton backend): Isaac Sim's RTX camera cannot start in this IsaacLab venv (it ships
USD 26.08, Kit's RTX plugins need 25.11) and H100s have no RT cores, so the Newton Warp renderer is used. The office
USD imports as invisible collision shapes plus a degenerate collision plane, so `make_environment_visual_only()`
flags every environment shape VISIBLE only: the camera sees the office, but the drone collides only with the floor
(crashes must be detected from the onboard depth).

## H3 features (`h3/`)

See `h3/README.md`. Feed a sim frame in as the DiffWAM observation:

```bash
conda activate sakana_dreamer
cd ~/sakana_dreamer/h3
python h3_features.py --image ../sim/outputs/h3in/obs_first.png \
  --prompt 'First-person drone camera flying forward between tall red pillars over a checkered floor' \
  --out outputs/sim_obs.pt
```
