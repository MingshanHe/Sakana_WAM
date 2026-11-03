# sakana_dreamer

Drone world-model experiments: an Isaac Lab drone test scenario with an onboard camera (`sim/`) and MiniMax-H3
predictive-feature extraction in the style of DiffWAM (`h3/`).

```
sakana_dreamer/
├── sim/                     Isaac Lab drone scenario  (env: ~/IsaacLab/.venv, Python 3.12)
│   ├── drone_scene.py       scene cfg: Crazyflie + onboard RGB-D camera + random red-pillar field
│   ├── controller.py        Lee geometric velocity controller -> body wrench (thrust + torque)
│   ├── run_drone_test.py    fly the scenario, record onboard video / depth / pose
│   └── outputs/<run>/       onboard.mp4, obs_first.png, episode.npz
└── h3/                      MiniMax-H3 feature extraction (env: conda `sakana_dreamer`, Python 3.11)
    ├── h3_features.py       H3FeatureExtractor: DiT blocks 15/25/35 at evals 0-3, no video decode
    ├── smoke_test_tiny.py   CPU test with a tiny random H3 transformer
    ├── inspect_features.py  stats + PCA image of the features
    └── README.md            details
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

## H3 features (`h3/`)

See `h3/README.md`. Feed a sim frame in as the DiffWAM observation:

```bash
conda activate sakana_dreamer
cd ~/sakana_dreamer/h3
python h3_features.py --image ../sim/outputs/h3in/obs_first.png \
  --prompt 'First-person drone camera flying forward between tall red pillars over a checkered floor' \
  --out outputs/sim_obs.pt
```
