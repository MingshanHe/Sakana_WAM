"""Phase 0: take off indoors, hover, and capture the first onboard frame I0 for the video world model.

The drone starts on the floor at `--start x y` facing `--yaw`, climbs to `--altitude`, hovers until it settles,
and saves (in `--out-dir`):

  first_frame.png   832x480 onboard RGB at hover -> Wan2.2 / DiffWAM-Flash input I0
  first_depth.npy   matching depth [m]
  first_pose.json   drone pose (x, y, z, qx, qy, qz, qw), velocity and camera intrinsics at capture time
  takeoff.mp4       onboard video of the take-off

  ~/IsaacLab/.venv/bin/python takeoff_capture.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--env", default="office")
parser.add_argument("--start", type=float, nargs=2, default=[-16.2, 27.1], help="Take-off point x y [m].")
parser.add_argument("--yaw", type=float, default=90.0, help="Heading [deg].")
parser.add_argument("--altitude", type=float, default=1.2)
parser.add_argument("--climb-time", type=float, default=3.0, help="Seconds to reach altitude.")
parser.add_argument("--hover-time", type=float, default=2.0, help="Seconds of hover before the capture.")
parser.add_argument("--cam-width", type=int, default=832)
parser.add_argument("--cam-height", type=int, default=480)
parser.add_argument("--hfov", type=float, default=90.0)
parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "takeoff"))
add_launcher_args(parser)
args_cli = parser.parse_args()

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from controller import VelocityController  # noqa: E402
from drone_scene import CAMERA_OFFSET, newton_physics_cfg  # noqa: E402
from indoor_scene import make_environment_visual_only, make_indoor_scene_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 200.0
RECORD_EVERY = 8  # 25 fps


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    yaw = math.radians(args_cli.yaw)
    quat = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device, physics=newton_physics_cfg())
    scene_cfg = make_indoor_scene_cfg(
        args_cli.env, "newton", cam_width=args_cli.cam_width, cam_height=args_cli.cam_height,
        hfov_deg=args_cli.hfov, camera_period=RECORD_EVERY * PHYSICS_DT,
        start_pos=(args_cli.start[0], args_cli.start[1], 0.05), start_yaw_quat=quat,
    )
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        make_environment_visual_only()
        scene = instantiate(scene_cfg)
        sim.reset()
        robot, camera = scene["robot"], scene["camera"]
        base_id = robot.body_names.index("body")
        masses = robot.data.body_mass.torch
        mass = masses.sum(-1)
        inertia = robot.data.body_inertia.torch[:, base_id].reshape(-1, 3, 3) * (mass / masses[:, base_id])[:, None, None]
        ctrl = VelocityController(mass, inertia)
        ctrl.reset(robot.data.root_quat_w.torch)
        offset = torch.tensor(CAMERA_OFFSET, device=sim.device).expand(scene.num_envs, 3)

        def attach_camera():
            pos, q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
            camera.set_world_poses(pos + quat_apply(q, offset), q, convention="world")

        frames, cmd = [], torch.zeros(scene.num_envs, 4, device=sim.device)
        climb_steps = int(args_cli.climb_time / PHYSICS_DT)
        total_steps = climb_steps + int(args_cli.hover_time / PHYSICS_DT)
        for step in range(total_steps):
            # Altitude reference ramps from the floor to the target, then holds (smooth take-off).
            cmd[:, 2] = args_cli.altitude * min(1.0, step / climb_steps)
            data = robot.data
            force_b, torque_b = ctrl(cmd, data.root_pos_w.torch, data.root_quat_w.torch, data.root_lin_vel_w.torch,
                                     data.root_ang_vel_b.torch, PHYSICS_DT)
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=force_b[:, None], torques=torque_b[:, None], body_ids=[base_id])
            scene.write_data_to_sim()
            sim.step()
            attach_camera()
            scene.update(PHYSICS_DT)
            if step % RECORD_EVERY == 0:
                frames.append(camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8))
            if step % 200 == 0:
                p, v = data.root_pos_w.torch[0].tolist(), data.root_lin_vel_w.torch[0].tolist()
                print(f"t={step * PHYSICS_DT:4.1f}s pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                      f"vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) z_ref={cmd[0, 2].item():.2f}")

        # Capture I0 at hover.
        camera.update(dt=0.0, force_recompute=True)
        rgb = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
        depth = camera.data.output["distance_to_image_plane"].torch[0, ..., 0].cpu().numpy().astype(np.float32)
        data = robot.data
        pose = {
            "pos": data.root_pos_w.torch[0].tolist(), "quat_xyzw": data.root_quat_w.torch[0].tolist(),
            "lin_vel_w": data.root_lin_vel_w.torch[0].tolist(), "ang_vel_b": data.root_ang_vel_b.torch[0].tolist(),
            "camera_offset_b": list(CAMERA_OFFSET), "image_size": [args_cli.cam_width, args_cli.cam_height],
            "hfov_deg": args_cli.hfov, "env": args_cli.env, "start": args_cli.start, "yaw_deg": args_cli.yaw,
            "intrinsics": camera.data.intrinsic_matrices.torch[0].tolist(),
        }
        imageio.imwrite(os.path.join(args_cli.out_dir, "first_frame.png"), rgb)
        np.save(os.path.join(args_cli.out_dir, "first_depth.npy"), depth)
        json.dump(pose, open(os.path.join(args_cli.out_dir, "first_pose.json"), "w"), indent=2)
        imageio.mimwrite(os.path.join(args_cli.out_dir, "takeoff.mp4"), np.stack(frames), fps=1 / (RECORD_EVERY * PHYSICS_DT),
                         macro_block_size=1)
        print(f"[INFO] hover pose {np.round(pose['pos'], 3).tolist()}, depth min {depth[depth > 0].min():.2f} m, "
              f"saved {args_cli.out_dir}/first_frame.png ({len(frames)} take-off frames)")


if __name__ == "__main__":
    main()
