"""Isaac Lab scene for the drone test scenario: a Crazyflie with an onboard forward camera flying through red pillars.

Physics is Newton (MuJoCo-Warp) and the camera is rendered by the Newton Warp ray tracer, so the scene runs without
the Isaac Sim Kit app and without RTX cores (works on H100). Quaternions are (x, y, z, w), as everywhere in Isaac Lab 3.
"""

from __future__ import annotations

import math
import random

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, replace

from isaaclab_assets.robots.quadcopter import CRAZYFLIE_CFG  # isort: skip

PILLAR_COLOR = (0.75, 0.22, 0.12)
CAMERA_OFFSET = (0.06, 0.0, 0.015)  # in the drone body frame [m], ahead of the props


def newton_physics_cfg() -> NewtonCfg:
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=40, nconmax=40, cone="pyramidal", integrator="implicitfast"),
        num_substeps=1,
        use_cuda_graph=True,
    )


def physx_physics_cfg():
    from isaaclab_physx.physics import PhysxCfg

    return PhysxCfg()


def onboard_camera_cfg(width: int = 416, height: int = 240, hfov_deg: float = 90.0, update_period: float = 0.0,
                       renderer: str = "newton"):
    """Forward-looking pinhole camera on the drone body.

    Kit-less, the prim is not moved by the Newton body: the runner sets its world pose from the body pose every
    step (see `CAMERA_OFFSET`), so `offset` only matters for the initial frame.
    """
    aperture = 20.955  # mm, the USD default
    focal = aperture / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=update_period,
        width=width,
        height=height,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal,
            horizontal_aperture=aperture,
            vertical_aperture=aperture * height / width,
            clipping_range=(0.02, 50.0),
        ),
        # "world" convention: camera looks along +X of its parent with +Z up, i.e. along the drone's nose.
        offset=CameraCfg.OffsetCfg(pos=CAMERA_OFFSET, rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        renderer_cfg=_renderer_cfg(renderer),
    )


def _renderer_cfg(renderer: str):
    """"newton": Newton Warp ray tracer (kit-less, any GPU). "rtx": Isaac Sim RTX (needs the Kit app)."""
    if renderer == "rtx":
        from isaaclab_physx.renderers import IsaacRtxRendererCfg

        return IsaacRtxRendererCfg(depth_clipping_behavior="max")
    return NewtonWarpRendererCfg(depth_clipping_behavior="max")


def pillar_layout(
    num_pillars: int = 14,
    x_range: tuple[float, float] = (2.5, 14.0),
    y_range: tuple[float, float] = (-3.0, 3.0),
    min_gap: float = 1.1,
    seed: int = 0,
) -> list[tuple[float, float, float]]:
    """Random (x, y, radius) pillars with a minimum surface-to-surface gap, kept off the start corridor x < 2."""
    rng = random.Random(seed)
    pillars: list[tuple[float, float, float]] = []
    for _ in range(2000):
        if len(pillars) == num_pillars:
            break
        x, y, r = rng.uniform(*x_range), rng.uniform(*y_range), rng.uniform(0.15, 0.35)
        if all(math.hypot(x - px, y - py) > r + pr + min_gap for px, py, pr in pillars):
            pillars.append((x, y, r))
    return pillars


@configclass
class DroneForestSceneCfg(InteractiveSceneCfg):
    """Ground, light, drone and onboard camera. Pillars are added per layout by :func:`make_scene_cfg`."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    )
    robot = replace(CRAZYFLIE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    camera = onboard_camera_cfg()


def make_scene_cfg(
    num_envs: int = 1,
    env_spacing: float = 20.0,
    pillar_height: float = 4.0,
    cam_width: int = 416,
    cam_height: int = 240,
    hfov_deg: float = 90.0,
    layout_seed: int = 0,
    num_pillars: int = 14,
    camera_period: float = 0.0,
) -> DroneForestSceneCfg:
    cfg = DroneForestSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    cfg.camera = onboard_camera_cfg(cam_width, cam_height, hfov_deg, camera_period)
    for i, (x, y, r) in enumerate(pillar_layout(num_pillars, seed=layout_seed)):
        setattr(
            cfg,
            f"pillar_{i:02d}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Pillar_{i:02d}",
                # No rigid-body props: a static collider.
                spawn=sim_utils.CylinderCfg(
                    radius=r,
                    height=pillar_height,
                    axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PILLAR_COLOR, roughness=0.6),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, pillar_height / 2)),
            ),
        )
    return cfg
