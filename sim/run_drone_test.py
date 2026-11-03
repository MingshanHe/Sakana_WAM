"""Drone test scenario in Isaac Lab: fly a Crazyflie through a pillar field and record the onboard camera.

The drone holds an altitude and flies forward; a simple depth-based reactive rule yaws away from the nearer side
when the centre of the depth image gets close (a baseline, not a learned policy). Every `--record-every` physics
steps the onboard RGB / depth frame is stored. Outputs (in `--out-dir`):

  onboard.mp4            onboard RGB video of env 0
  obs_first.png          first onboard frame of env 0 (e.g. the H3 / DiffWAM `I0` input)
  episode.npz            rgb (T, H, W, 3) uint8, depth (T, H, W) float32, pose (T, 7), vel (T, 6), cmd (T, 4), t (T,)

Run (headless, no Kit app needed):
  ~/IsaacLab/.venv/bin/python run_drone_test.py --device cuda:0 --duration 10
"""

from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=10.0, help="Seconds of flight.")
parser.add_argument("--speed", type=float, default=1.5, help="Forward speed command [m/s].")
parser.add_argument("--altitude", type=float, default=1.0, help="Altitude reference [m].")
parser.add_argument("--cam-width", type=int, default=416)
parser.add_argument("--cam-height", type=int, default=240)
parser.add_argument("--hfov", type=float, default=90.0)
parser.add_argument("--record-every", type=int, default=8, help="Physics steps per recorded frame (200 Hz / 8 = 25 fps).")
parser.add_argument("--layout-seed", type=int, default=0)
parser.add_argument("--no-avoid", action="store_true", help="Fly straight, no reactive yaw.")
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "run"))
add_launcher_args(parser)
args_cli = parser.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402

from isaaclab.utils.math import euler_xyz_from_quat, quat_apply  # noqa: E402

from controller import VelocityController  # noqa: E402
from drone_scene import CAMERA_OFFSET, make_scene_cfg, newton_physics_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 200.0


def reactive_command(depth: torch.Tensor, speed: float, altitude: float, avoid: bool) -> torch.Tensor:
    """(N, H, W) depth -> (N, 4) command. Slow down and yaw towards the freer half when the centre is close."""
    n, h, w = depth.shape
    cmd = torch.zeros(n, 4, device=depth.device)
    cmd[:, 0], cmd[:, 2] = speed, altitude
    if not avoid:
        return cmd
    band = depth[:, h // 3 : 2 * h // 3]
    centre = band[:, :, w // 3 : 2 * w // 3].flatten(1).quantile(0.05, dim=1)
    left, right = band[:, :, : w // 2].mean((1, 2)), band[:, :, w // 2 :].mean((1, 2))
    danger = ((2.0 - centre) / 1.5).clamp(0.0, 1.0)
    cmd[:, 0] = speed * (1.0 - 0.7 * danger)
    cmd[:, 3] = 1.8 * danger * torch.where(left > right, 1.0, -1.0)  # +yaw = turn left
    return cmd


def save_outputs(out_dir: str, frames: dict[str, list], fps: float):
    import imageio.v2 as imageio

    os.makedirs(out_dir, exist_ok=True)
    rgb = np.stack(frames["rgb"])
    imageio.imwrite(os.path.join(out_dir, "obs_first.png"), rgb[0])
    try:
        imageio.mimwrite(os.path.join(out_dir, "onboard.mp4"), rgb, fps=fps, macro_block_size=1)
    except Exception as e:  # no ffmpeg backend
        print(f"[WARN] mp4 export failed ({e}); writing onboard.gif instead")
        imageio.mimwrite(os.path.join(out_dir, "onboard.gif"), rgb, duration=1.0 / fps)
    np.savez_compressed(os.path.join(out_dir, "episode.npz"), **{k: np.stack(v) for k, v in frames.items()})
    print(f"[INFO] saved {len(rgb)} frames to {out_dir}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device, physics=newton_physics_cfg())
    scene_cfg = make_scene_cfg(
        num_envs=args_cli.num_envs,
        cam_width=args_cli.cam_width,
        cam_height=args_cli.cam_height,
        hfov_deg=args_cli.hfov,
        layout_seed=args_cli.layout_seed,
        camera_period=args_cli.record_every * PHYSICS_DT,
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(scene_cfg)
        sim.reset()
        robot, camera = scene["robot"], scene["camera"]
        print(f"[INFO] bodies: {robot.body_names}  joints: {robot.joint_names}")

        base_id = robot.body_names.index("body")
        masses = robot.data.body_mass.torch
        mass = masses.sum(-1)
        inertia = robot.data.body_inertia.torch[:, base_id].reshape(-1, 3, 3) * (mass / masses[:, base_id])[:, None, None]
        print(f"[INFO] mass {mass[0].item():.4f} kg, base inertia diag {inertia[0].diagonal().tolist()}")
        ctrl = VelocityController(mass, inertia)

        # Start 0.5 m behind the field, at the target altitude, hovering.
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = scene.env_origins + torch.tensor([0.0, 0.0, args_cli.altitude], device=sim.device)
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        scene.reset()
        ctrl.reset(root_pose[:, 3:7])
        cam_offset = torch.tensor(CAMERA_OFFSET, device=sim.device).expand(args_cli.num_envs, 3)

        def attach_camera():
            # Without Kit the camera prim is not carried by the Newton body, so pin it to the body pose explicitly.
            pos, quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
            camera.set_world_poses(pos + quat_apply(quat, cam_offset), quat, convention="world")

        attach_camera()

        frames = {k: [] for k in ("rgb", "depth", "pose", "vel", "cmd", "t")}
        num_steps = int(args_cli.duration / PHYSICS_DT)
        cmd = reactive_command(
            torch.full((args_cli.num_envs, args_cli.cam_height, args_cli.cam_width), 50.0, device=sim.device),
            args_cli.speed, args_cli.altitude, not args_cli.no_avoid,
        )
        t0 = time.perf_counter()
        min_clearance = float("inf")
        for step in range(num_steps):
            data = robot.data
            pos_w, quat_w = data.root_pos_w.torch, data.root_quat_w.torch
            force_b, torque_b = ctrl(cmd, pos_w - scene.env_origins, quat_w, data.root_lin_vel_w.torch,
                                     data.root_ang_vel_b.torch, PHYSICS_DT)
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_b[:, None], torques=torque_b[:, None], body_ids=[base_id]
            )
            scene.write_data_to_sim()
            sim.step()
            attach_camera()
            scene.update(PHYSICS_DT)

            if step % args_cli.record_every == 0:
                depth = camera.data.output["distance_to_image_plane"].torch[..., 0]
                cmd = reactive_command(depth, args_cli.speed, args_cli.altitude, not args_cli.no_avoid)
                rgb = camera.data.output["rgb"].torch[0, ..., :3]
                min_clearance = min(min_clearance, depth[0].min().item())
                frames["rgb"].append(rgb.cpu().numpy().astype(np.uint8))
                frames["depth"].append(depth[0].cpu().numpy().astype(np.float32))
                frames["pose"].append(torch.cat([pos_w[0] - scene.env_origins[0], quat_w[0]]).cpu().numpy())
                frames["vel"].append(
                    torch.cat([data.root_lin_vel_w.torch[0], data.root_ang_vel_b.torch[0]]).cpu().numpy()
                )
                frames["cmd"].append(cmd[0].cpu().numpy())
                frames["t"].append(np.float32(step * PHYSICS_DT))
            if step % 200 == 0:
                p = (pos_w[0] - scene.env_origins[0]).tolist()
                v = data.root_lin_vel_w.torch[0].tolist()
                roll, pitch, yaw = (a[0].item() for a in euler_xyz_from_quat(quat_w[:1]))
                print(
                    f"t={step * PHYSICS_DT:5.2f}s pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                    f"vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) rpy=({roll:+.2f},{pitch:+.2f},{yaw:+.2f}) "
                    f"cmd=({', '.join(f'{c:+.2f}' for c in cmd[0].tolist())})"
                )

        wall = time.perf_counter() - t0
        print(f"[INFO] {num_steps} steps in {wall:.1f}s ({num_steps * PHYSICS_DT / wall:.1f}x real time), "
              f"min depth seen {min_clearance:.2f} m")
        save_outputs(args_cli.out_dir, frames, fps=1.0 / (args_cli.record_every * PHYSICS_DT))


if __name__ == "__main__":
    main()
