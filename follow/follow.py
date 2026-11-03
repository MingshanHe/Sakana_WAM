"""Phases 3-4 (controller v1): fly a drone so its onboard camera follows a goal video in a latent space (env
`sakana_hyperdrone`).

Every slot (5/38 s) the controller tries a grid of velocity commands (forward, sideways, vertical, yaw rate in the
drone's yaw frame), renders where each one leads after one and two slots with HyperDrone's ray tracer, encodes the
views, and executes the command with the lowest
    latent_error(view after 1 slot, goal[s+1]) + 0.5 * latent_error(view after 2 slots, goal[s+2])
    + collision penalty (anything closer than 0.3 m in the view) + small smoothness penalty.
The simulator is the predictive model (rendering-in-the-loop MPC): this checks whether the latent error is a usable
control signal before an RL policy is trained on it. Motion is kinematic (perfect velocity tracking).

Goal videos (--task):
  replay  a scripted reference flight in the same house, rendered at the slot times -> known ground-truth path
  wan     the Wan2.2 imagined video of the scenario (seed --seed)
Latent spaces (--space):
  flash   Flash goals: replay -> E(reference frames); wan -> the Flash features Z of the imagination (projected)
          onboard views -> E (the fold encoder that never saw this scenario)
  vae     Wan2.2 VAE latents of the goal frames vs of the onboard views
  pixel   (replay only) RGB at 15 x 26, a no-learning reference

    python follow.py                                   # every scenario, replay + wan, flash + vae
    python follow.py --scenario s01_room_forward --task replay --space pixel
"""

import argparse
import glob
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from latents import (SLOT_DT, FlashEncoder, FlashProjector, WanVAE, latent_error, slot_frame_index, to_input)
from scenes import HouseCamera

NUM_SLOTS = 39
COMMANDS = np.array(list(itertools.product([0.0, 0.3, 0.6, 0.9, 1.2], [-0.4, 0.0, 0.4], [-0.2, 0.0, 0.2],
                                           [-0.5, 0.0, 0.5])), dtype=float)  # vx, vy, vz, wz

# Scripted reference flights for --task replay, one per motion type (body-yaw frame velocities, m/s and rad/s).
# Deliberately off the COMMANDS grid, so the follower cannot reproduce the reference views exactly.
REFERENCE = {
    "forward": lambda t: (0.45, 0.0, 0.0, 0.0),
    "hallway": lambda t: (0.75, 0.0, 0.0, 0.0),
    "doorway": lambda t: (0.75 if t < 3.5 else 0.0, 0.0, 0.0, 0.0),
    "turn left": lambda t: (0.35, 0.0, 0.0, 0.3),
    "ascend": lambda t: (0.15, 0.0, 0.13, 0.0),
    "approach": lambda t: (0.75 if t < 3.5 else 0.0, 0.0, 0.0, 0.0),
    "turn right": lambda t: (0.35, 0.0, 0.0, -0.3),
    "strafe right": lambda t: (0.0, -0.3, 0.0, 0.0),
    "approach+stop": lambda t: (0.5 if t < 3.0 else 0.0, 0.0, 0.0, 0.0),
    "descend": lambda t: (0.3, 0.0, -0.1, 0.0),
}
# Replay starts the follower off the reference start, so it has to recover: 0.25 m to the left, 8 deg to the left.
START_OFFSET_LEFT_M, START_OFFSET_YAW_DEG = 0.25, 8.0
TOP_K_LOOKAHEAD = 16


def step_pose(pose: np.ndarray, cmd: np.ndarray, dt: float) -> np.ndarray:
    """Kinematic step of (N, 4) poses x, y, z, yaw under (N, 4) yaw-frame commands."""
    x, y, z, yaw = pose.T
    vx, vy, vz, wz = cmd.T
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + (vx * c - vy * s) * dt, y + (vx * s + vy * c) * dt, z + vz * dt, yaw + wz * dt], 1)


def reference_path(start: np.ndarray, motion: str) -> np.ndarray:
    poses = [start]
    for s in range(NUM_SLOTS - 1):
        poses.append(step_pose(poses[-1][None], np.array([REFERENCE[motion](s * SLOT_DT)]), SLOT_DT)[0])
    return np.stack(poses)


class Space:
    """Encodes onboard views into the chosen latent space."""

    def __init__(self, name: str, encoder: FlashEncoder | None = None, vae: WanVAE | None = None):
        self.name, self.encoder, self.vae = name, encoder, vae

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        x = to_input(frames)
        if self.name == "flash":
            return torch.cat([self.encoder(x[i:i + 64]) for i in range(0, len(x), 64)])
        if self.name == "vae":
            return self.vae(x)
        return F.interpolate(x, size=(15, 26), mode="area").permute(0, 2, 3, 1)  # pixel


def fold_checkpoint(encoders: str, scenario: str) -> str:
    for path in sorted(glob.glob(os.path.join(encoders, "fold*.pt"))):
        if scenario in torch.load(path, weights_only=False, map_location="cpu")["held_out"]:
            return path
    raise FileNotFoundError(f"no fold in {encoders} holds out {scenario}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", default="scenarios")
    parser.add_argument("--scenario", nargs="+", default=["all"], help="Scenario ids, or 'all'.")
    parser.add_argument("--task", nargs="+", choices=["replay", "wan"], default=["replay", "wan"])
    parser.add_argument("--space", nargs="+", choices=["flash", "vae", "pixel"], default=["flash", "vae"])
    parser.add_argument("--seed", nargs="+", default=["best"],
                        help="Imagined videos to follow (--task wan): seed numbers, or 'best' = per scenario the seed "
                             "whose imagination moves in the instructed direction (then the one with more motion).")
    parser.add_argument("--encoders", default="encoders")
    parser.add_argument("--out-dir", default="follow_runs")
    parser.add_argument("--smooth", type=float, default=0.02)
    args = parser.parse_args()

    all_meta = json.load(open(os.path.join(args.scenarios, "scenarios.json")))["scenarios"]
    metas = all_meta if args.scenario == ["all"] else [m for m in all_meta if m["id"] in args.scenario]
    vae = WanVAE() if "vae" in args.space else None
    for meta in metas:
        # Same renderer configuration as make_scenarios.py (832 x 480, batch 64), so no new JIT build; views are
        # downsampled to 416 x 240 by `to_input`.
        cam = HouseCamera(meta["house"])
        for space_name in args.space:
            proj = None
            if space_name == "flash":
                ckpt = torch.load(fold_checkpoint(args.encoders, meta["id"]), weights_only=False)
                encoder = FlashEncoder(pretrained=False).cuda().eval()
                encoder.load_state_dict(ckpt["encoder"])
                proj = FlashProjector.from_state_dict(ckpt["projector"])
                space = Space("flash", encoder=encoder)
            else:
                space = Space(space_name, vae=vae)
            for task in args.task:
                if task == "wan" and space_name == "pixel":
                    continue
                for seed in (wan_seeds(args, meta["id"]) if task == "wan" else [0]):
                    episode(args, meta, cam, space, proj, task, seed)


def wan_seeds(args, scenario: str) -> list[int]:
    if args.seed != ["best"]:
        return [int(s) for s in args.seed]
    rows = [r for r in json.load(open(os.path.join(args.scenarios, "imagination_eval.json"))) if r["id"] == scenario]
    return [max(rows, key=lambda r: (r["direction_ok"], r["flow_px"]))["seed"]]


def episode(args, meta, cam, space, proj, task, seed):
    t0 = time.perf_counter()
    start = np.array(meta["start"]["pose"], dtype=float)
    # Goal sequence (39 latent maps) and the frames it came from (for the side-by-side video).
    ref_path = None
    if task == "replay":
        ref_path = reference_path(start, meta["motion"])
        goal_frames, _ = cam.render(ref_path)
        goals = space(goal_frames)
    else:
        run = os.path.join(args.scenarios, meta["id"], f"seed{seed}")
        video = np.load(os.path.join(run, "video.npy"), mmap_mode="r")
        goal_frames = np.ascontiguousarray(video[slot_frame_index(len(video))])
        if space.name == "flash":
            goals = proj(torch.load(os.path.join(run, "flash.pt"), weights_only=False)["Z"])
        elif space.name == "vae":
            goals = torch.load(os.path.join(run, "goals_vae.pt")).cuda()
        else:
            goals = space(goal_frames)

    pose, prev_cmd = start.copy(), np.zeros(4)
    if task == "replay":
        pose[0] -= START_OFFSET_LEFT_M * math.sin(start[3])
        pose[1] += START_OFFSET_LEFT_M * math.cos(start[3])
        pose[3] += math.radians(START_OFFSET_YAW_DEG)
    poses, chosen_err, hover_err, frames = [pose.copy()], [], [], []
    hover_view = space(cam.render([pose])[0])

    def near_obstacle(depth):
        h, w = depth.shape[1:]
        d = np.where(depth > 0, depth, 1e3)[:, h // 6: 5 * h // 6, w // 6: 5 * w // 6]
        return d.reshape(len(depth), -1).min(1)

    n = len(COMMANDS)
    for s in range(NUM_SLOTS - 1):
        # One slot ahead for every command; two slots ahead only for the most promising ones.
        p1 = step_pose(np.repeat(pose[None], n, 0), COMMANDS, SLOT_DT)
        rgb, depth = cam.render(np.concatenate([pose[None], p1]))
        frames.append(rgb[0])
        e1 = latent_error(space(rgb[1:]), goals[s + 1]).cpu().numpy()
        crash1 = near_obstacle(depth[1:]) < 0.3
        top = np.argsort(e1 + 1e3 * crash1)[:TOP_K_LOOKAHEAD]
        p2 = step_pose(p1[top], COMMANDS[top], SLOT_DT)
        rgb2, depth2 = cam.render(p2)
        e2 = latent_error(space(rgb2), goals[min(s + 2, NUM_SLOTS - 1)]).cpu().numpy()
        crash = crash1[top] | (near_obstacle(depth2) < 0.3)
        cost = (e1[top] + 0.5 * e2 + 1e3 * crash
                + args.smooth * ((COMMANDS[top] - prev_cmd) ** 2).sum(1) * e1[top].mean())
        best = int(top[np.argmin(cost)])
        pose, prev_cmd = p1[best], COMMANDS[best]
        poses.append(pose.copy())
        chosen_err.append(float(e1[best]))
        hover_err.append(float(latent_error(hover_view, goals[s + 1])[0]))
    frames.append(cam.render([pose])[0][0])
    poses = np.array(poses)

    # Displacement in the start frame (forward, left, up) and yaw change.
    c, sn = math.cos(start[3]), math.sin(start[3])
    d = poses[-1, :3] - start[:3]
    summary = {"scenario": meta["id"], "motion": meta["motion"], "task": task, "space": space.name,
               "seed": seed, "forward_m": float(d[0] * c + d[1] * sn), "left_m": float(-d[0] * sn + d[1] * c),
               "up_m": float(d[2]), "yaw_deg": float(math.degrees(poses[-1, 3] - start[3])),
               "err_mean": float(np.mean(chosen_err)), "err_hover_mean": float(np.mean(hover_err)),
               "err_final": chosen_err[-1], "err_hover_final": hover_err[-1]}
    if ref_path is not None:
        pos_err = np.linalg.norm(poses[:, :3] - ref_path[:, :3], axis=1)
        hover_pos_err = np.linalg.norm(ref_path[:, :3] - poses[0, :3], axis=1)
        yaw_err = np.degrees(np.abs(np.angle(np.exp(1j * (poses[:, 3] - ref_path[:, 3])))))
        rd = ref_path[-1, :3] - start[:3]
        summary.update(pos_rmse_m=float(np.sqrt((pos_err ** 2).mean())), pos_final_m=float(pos_err[-1]),
                       hover_pos_rmse_m=float(np.sqrt((hover_pos_err ** 2).mean())),
                       ref_length_m=float(np.linalg.norm(np.diff(ref_path[:, :3], axis=0), axis=1).sum()),
                       yaw_final_err_deg=float(yaw_err[-1]),
                       ref_forward_m=float(rd[0] * c + rd[1] * sn), ref_left_m=float(-rd[0] * sn + rd[1] * c),
                       ref_up_m=float(rd[2]), ref_yaw_deg=float(math.degrees(ref_path[-1, 3] - start[3])))
    summary["seconds"] = time.perf_counter() - t0

    out = os.path.join(args.out_dir, f"{meta['id']}_{task}_{space.name}" + (f"_seed{seed}" if task == "wan" else ""))
    os.makedirs(out, exist_ok=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    np.savez(os.path.join(out, "trajectory.npz"), poses=poses, ref=ref_path if ref_path is not None else np.zeros(0),
             err=np.array(chosen_err), hover_err=np.array(hover_err))

    def small(x):
        return to_input(x, "cpu").permute(0, 2, 3, 1).mul(255).byte().numpy()

    side = np.concatenate([small(goal_frames), small(np.stack(frames))], axis=2)  # goal | follower
    import imageio.v2 as imageio

    imageio.mimwrite(os.path.join(out, "goal_vs_follower.mp4"), side, fps=round(1 / SLOT_DT), macro_block_size=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
