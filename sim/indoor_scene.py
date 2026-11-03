"""Indoor drone scene: an Isaac Sim environment USD (office by default) + Crazyflie + onboard camera.

Two backends: "kit" (default) = Isaac Sim app, PhysX physics, RTX camera, the USD loaded as authored;
"newton" = the kit-less stack of `drone_scene.py`. The environment USD is spawned once
under /World/Environment and shared by all envs. Its own colliders are disabled: the office USD
carries zero-size collision shapes that MuJoCo-Warp rejects, and the camera only needs its visual meshes. An
invisible collision floor at z = 0 is what the drone takes off from.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from drone_scene import CRAZYFLIE_CFG, onboard_camera_cfg

INDOOR_ENVS = {
    "office": f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd",
    "hospital": f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd",
    "simple_room": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Room/simple_room.usd",
    "warehouse": f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
}


def strip_environment_colliders(stage, root: str = "/World/Environment") -> int:
    """Make the environment visual-only.

    Deactivates its `CollisionPlane` prims (the office's ground collision plane is imported by Newton as a zero-size
    non-plane shape, which MuJoCo-Warp rejects) and removes any collision APIs.
    """
    from pxr import Usd, UsdPhysics

    removed = 0
    traversal = Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
    plane_paths = [p.GetPath() for p in traversal if p.GetName() == "CollisionPlane"]
    for path in plane_paths:
        # Instance proxies are read-only: de-instance the enclosing instance first.
        ancestor = stage.GetPrimAtPath(path)
        while ancestor.IsInstanceProxy():
            ancestor = ancestor.GetParent()
        for parent in [stage.GetPrimAtPath(p) for p in path.GetPrefixes()]:
            if parent.IsValid() and parent.IsInstance():
                parent.SetInstanceable(False)
        stage.GetPrimAtPath(path).SetActive(False)
        removed += 1
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        for api in (UsdPhysics.MeshCollisionAPI, UsdPhysics.CollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                removed += 1
    return removed


def make_environment_visual_only(prefix: str = "/World/Environment"):
    """Newton backend: before MuJoCo-Warp is built, flag every environment shape as VISIBLE only.

    The office USD imports as ~3.6k collision-only (invisible) shapes plus a degenerate collision plane. Visual-only
    shapes are skipped by the MuJoCo converter and drawn by the Warp renderer, so the camera sees the office and the
    solver only handles the drone and the floor. The drone therefore does not collide with furniture; detect crashes
    from the onboard depth instead.
    """
    from newton import ModelBuilder, ShapeFlags

    if getattr(ModelBuilder, "_sakana_visual_only_patch", False):
        return
    original_finalize = ModelBuilder.finalize

    def patched_finalize(self, *args, **kwargs):
        # Patch the builder, not the finalized model: the renderer and the solver both read the flags from it.
        for i, label in enumerate(self.shape_label):
            if label.startswith(prefix):
                self.shape_flags[i] = int(ShapeFlags.VISIBLE)
        return original_finalize(self, *args, **kwargs)

    ModelBuilder.finalize = patched_finalize
    ModelBuilder._sakana_visual_only_patch = True


def _spawn_visual_only(prim_path, cfg, *args, **kwargs):
    """`UsdFileCfg.func` + strip the colliders right away, so Newton parses the cleaned prims during scene setup."""
    prim = sim_utils.spawn_from_usd(prim_path, cfg, *args, **kwargs)
    strip_environment_colliders(prim.GetStage(), prim_path)
    return prim


def environment_cfg(env_name: str, backend: str = "kit") -> AssetBaseCfg:
    """Kit/PhysX loads the USD as authored; for Newton the colliders are stripped (see `strip_environment_colliders`)."""
    spawn = sim_utils.UsdFileCfg(usd_path=INDOOR_ENVS[env_name])
    if backend == "newton":
        spawn.func = _spawn_visual_only
    return AssetBaseCfg(prim_path="/World/Environment", spawn=spawn)


@configclass
class IndoorDroneSceneCfg(InteractiveSceneCfg):
    environment = environment_cfg("office")
    floor = AssetBaseCfg(
        prim_path="/World/CollisionFloor",
        spawn=sim_utils.CuboidCfg(size=(200.0, 200.0, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0)))
    robot = replace(CRAZYFLIE_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    camera = onboard_camera_cfg()


def make_indoor_scene_cfg(
    env_name: str = "office",
    backend: str = "kit",
    num_envs: int = 1,
    cam_width: int = 832,
    cam_height: int = 480,
    hfov_deg: float = 90.0,
    camera_period: float = 0.0,
    start_pos: tuple[float, float, float] = (0.0, 0.0, 0.05),
    start_yaw_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> IndoorDroneSceneCfg:
    cfg = IndoorDroneSceneCfg(num_envs=num_envs, env_spacing=0.0)
    cfg.environment = environment_cfg(env_name, backend)
    cfg.camera = onboard_camera_cfg(cam_width, cam_height, hfov_deg, camera_period,
                                    renderer="rtx" if backend == "kit" else "newton")
    cfg.robot = replace(cfg.robot, init_state=replace(cfg.robot.init_state, pos=start_pos, rot=start_yaw_quat))
    return cfg
