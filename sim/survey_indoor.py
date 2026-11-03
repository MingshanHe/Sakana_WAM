"""Survey an indoor environment: print its bounds and render onboard-camera views from a grid of candidate spots.

Used to pick a take-off point. Saves outputs/survey_<env>/sheet.png (rows = spots, columns = yaw 0/90/180/270 deg)
and bounds.json.

  ~/IsaacLab/.venv/bin/python survey_indoor.py --env office --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--env", default="office")
parser.add_argument("--backend", choices=["kit", "newton"], default="newton",
                    help="kit = Isaac Sim RTX camera; does not start in the IsaacLab venv (USD 26.08 vs Kit's 25.11).")
parser.add_argument("--height", type=float, default=1.2, help="Camera height for the survey views [m].")
parser.add_argument("--grid", type=int, default=4, help="Candidate spots per axis inside the bounds.")
parser.add_argument("--out-dir", default=None)
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.backend == "kit":
    args_cli.enable_cameras = True

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.utils import instantiate  # noqa: E402

from drone_scene import newton_physics_cfg, physx_physics_cfg  # noqa: E402
from indoor_scene import make_environment_visual_only, make_indoor_scene_cfg  # noqa: E402


def yaw_quat(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]  # (x, y, z, w)


def main():
    out_dir = args_cli.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", f"survey_{args_cli.env}_{args_cli.backend}")
    os.makedirs(out_dir, exist_ok=True)
    physics = physx_physics_cfg() if args_cli.backend == "kit" else newton_physics_cfg()
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 200, device=args_cli.device, physics=physics)
    scene_cfg = make_indoor_scene_cfg(args_cli.env, args_cli.backend, cam_width=416, cam_height=240)
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        if args_cli.backend == "newton":
            make_environment_visual_only()
        scene = instantiate(scene_cfg)
        sim.reset()
        camera = scene["camera"]

        stage = sim_utils.get_current_stage()
        env_prim = stage.GetPrimAtPath("/World/Environment")
        # The USD's overall bound is dominated by a few huge outliers (sky / ground), so estimate the interior from
        # the 5-95 % percentiles of the centres of the meshes between floor and ceiling height.
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        centres = []
        for prim in Usd.PrimRange(env_prim, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                r = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
                if not r.IsEmpty():
                    centres.append(np.array(r.GetMidpoint()))
        centres = np.array(centres)
        centres = centres[(centres[:, 2] > -0.5) & (centres[:, 2] < 3.5)]
        lo, hi = np.percentile(centres, 5, axis=0), np.percentile(centres, 95, axis=0)
        lo[2] = 0.0
        n_meshes = len(centres)
        print(f"[INFO] {args_cli.env}: interior ~ min {lo.round(2)} max {hi.round(2)} from {n_meshes} mesh centres")
        json.dump({"min": lo.tolist(), "max": hi.tolist(), "meshes": n_meshes}, open(os.path.join(out_dir, "bounds.json"), "w"))

        g = args_cli.grid
        xs = np.linspace(lo[0], hi[0], g + 2)[1:-1]
        ys = np.linspace(lo[1], hi[1], g + 2)[1:-1]
        z = lo[2] + args_cli.height
        rows = []
        for x in xs:
            for y in ys:
                tiles = []
                for k in range(4):
                    pos = torch.tensor([[x, y, z]], device=sim.device, dtype=torch.float32)
                    quat = torch.tensor([yaw_quat(k * math.pi / 2)], device=sim.device)
                    camera.set_world_poses(pos, quat, convention="world")
                    sim.step()
                    camera.update(dt=0.0, force_recompute=True)
                    rgb = camera.data.output["rgb"].torch[0, ..., :3].cpu().numpy().astype(np.uint8)
                    tile = Image.fromarray(rgb)
                    ImageDraw.Draw(tile).text((4, 4), f"({x:.1f},{y:.1f}) yaw {90 * k}", fill=(255, 255, 0))
                    tiles.append(np.asarray(tile))
                rows.append(np.concatenate(tiles, 1))
        sheet = np.concatenate(rows, 0)
        Image.fromarray(sheet).save(os.path.join(out_dir, "sheet.png"))
        print(f"[INFO] saved {out_dir}/sheet.png ({len(rows)} spots x 4 yaws)")


if __name__ == "__main__":
    main()
