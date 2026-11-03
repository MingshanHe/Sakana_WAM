"""HyperDrone / hyperlux scenes for the imagined-video-following experiments (env: conda `sakana_hyperdrone`).

`HouseCamera` renders onboard views of a ProcTHOR house at arbitrary drone poses (x, y, z, yaw) in a batch: the
first frame I0 for Wan2.2, candidate views for the controller, and data for the encoders. Poses are in hyperlux's
world FLU frame (+X forward, +Y left, +Z up). The camera is level (yaw only), like a gimbal-stabilised onboard camera.
"""

from __future__ import annotations

import math
import os

import numpy as np

# hyperlux JITs its native core on first use; make sure the env's CMake (>= 3.24) is found before the system one.
os.environ["PATH"] = os.path.join(os.path.dirname(os.path.realpath(os.sys.executable))) + os.pathsep + os.environ["PATH"]


def _skip_rebuilds_of_built_targets():
    """hyperlux runs `cmake --build` on every load, even when the library is already built. On this cluster that
    step sometimes never returns (gmake exits, `cmake --build` stays in ep_poll), so return early when the artifact
    exists; targets that are really missing (a new renderer configuration) are still compiled."""
    from hyperlux._jit import _build

    if getattr(_build, "_sakana_skip_patch", False):
        return
    original = _build.build_target

    def build_target(component, target):
        directory = _build.ensure_configured(component)
        core = target == component.core_target
        if any(directory.glob(f"{target}*.so" if core else f"jit/{target}.so")):
            return directory
        return original(component, target)

    _build.build_target = build_target
    _build._sakana_skip_patch = True


_skip_rebuilds_of_built_targets()

WIDTH, HEIGHT, HFOV_DEG = 832, 480, 90.0  # Wan2.2 canvas -> 15 x 26 DiT tokens


def house_name(index: int) -> str:
    return f"ProcTHOR-Train-{index}"


class HouseCamera:
    """Batched level camera in one house: `render(poses)` with poses (N, 4) = x, y, z, yaw -> rgb (N,H,W,3), depth."""

    def __init__(self, house: int, width: int = WIDTH, height: int = HEIGHT, hfov_deg: float = HFOV_DEG,
                 batch: int = 64, fidelity: str = "veryhigh"):
        import hyperlux
        from hyperlux.camera import CameraSpec, Depth, RGB

        self.house, self.width, self.height, self.hfov_deg, self.batch = house, width, height, hfov_deg, batch
        scene = hyperlux.load_scene(hyperlux.datasets.procthor.Train200()[house_name(house)].reference)
        self._rgb, self._depth = RGB("rgba8"), Depth()
        self.renderer = hyperlux.HyperLux(width=width, height=height, batch_size=batch, fidelity=fidelity,
                                          cameras=(CameraSpec("front", (self._rgb, self._depth)),))
        self.renderer.init(scene)
        b = self.renderer.scene_bounds
        self.center = np.asarray(b["center"], dtype=float)
        self.half = np.asarray(b["half_extent"], dtype=float)
        self.fov_arg = self._fov_argument(hfov_deg)

    def _fov_argument(self, hfov_deg: float) -> float:
        """`renderer.camera(fov=...)` is documented only as degrees; measure which axis it spans and convert."""
        base = np.asarray(self.renderer.camera((0, 0, 0), (1, 0, 0), fov=60.0), dtype=float).reshape(-1)
        d00, du, dv = base[3:6], base[6:9], base[9:12]
        centre = d00 + 0.5 * du + 0.5 * dv
        span_u = 2 * math.degrees(math.atan2(0.5 * np.linalg.norm(du), np.linalg.norm(centre)))
        if abs(span_u - 60.0) < 1.0:  # fov is horizontal
            return hfov_deg
        vfov = 2 * math.degrees(math.atan(math.tan(math.radians(hfov_deg) / 2) * self.height / self.width))
        return vfov

    def camera_rows(self, poses: np.ndarray) -> np.ndarray:
        rows = []
        for x, y, z, yaw in poses:
            target = (x + math.cos(yaw), y + math.sin(yaw), z)
            rows.append(np.asarray(self.renderer.camera((x, y, z), target, fov=self.fov_arg), dtype=np.float32))
        return np.stack(rows).reshape(len(poses), -1)

    def render(self, poses) -> tuple[np.ndarray, np.ndarray]:
        poses = np.asarray(poses, dtype=float).reshape(-1, 4)
        rgbs, depths = [], []
        for start in range(0, len(poses), self.batch):
            chunk = poses[start:start + self.batch]
            rows = self.camera_rows(chunk)
            if len(chunk) < self.batch:  # pad the fixed-size batch
                rows = np.concatenate([rows, np.repeat(rows[-1:], self.batch - len(chunk), axis=0)])
            self.renderer.set_cameras(rows)
            self.renderer.render()
            rgbs.append(np.array(self.renderer.outputs[self._rgb].read())[: len(chunk), ..., :3])
            depths.append(np.array(self.renderer.outputs[self._depth].read())[: len(chunk)])
        depth = np.nan_to_num(np.concatenate(depths), nan=0.0, posinf=0.0)
        return np.concatenate(rgbs), depth

    def find_starts(self, rng: np.random.Generator, count: int = 1, height: float = 1.2, tries: int = 512,
                    min_ahead: float = 3.0, min_clear: float = 0.6) -> list[dict]:
        """Take-off poses inside the house with open space ahead.

        A candidate (x, y, yaw) is kept when the views ahead / behind / left / right all hit geometry (no miss = 0
        depth, so the camera is enclosed by the house), nothing is closer than `min_clear`, and the central band
        ahead is at least `min_ahead` deep. The best `count` by depth ahead are returned, at least 1.5 m apart.
        """
        xy = self.center[:2] + (rng.random((tries, 2)) * 2 - 1) * self.half[:2] * 0.9
        yaw = rng.random(tries) * 2 * math.pi
        rig = np.concatenate([np.stack([xy[:, 0], xy[:, 1], np.full(tries, height), yaw + k * math.pi / 2], 1)
                              for k in range(4)])
        _, depth = self.render(rig)
        depth = depth.reshape(4, tries, *depth.shape[1:])
        h, w = depth.shape[2:]
        enclosed = (depth > 0).reshape(4, tries, -1).mean(-1).min(0) > 0.995
        clear = np.where(depth > 0, depth, 1e3).reshape(4, tries, -1).min(-1).min(0) > min_clear
        ahead = np.median(depth[0, :, h // 3: 2 * h // 3, w // 3: 2 * w // 3].reshape(tries, -1), axis=1)
        ok = enclosed & clear & (ahead >= min_ahead)
        starts = []
        for i in np.argsort(-ahead):
            if not ok[i]:
                continue
            if all(np.hypot(*(xy[i] - np.array(s["pose"][:2]))) > 1.5 for s in starts):
                starts.append({"house": self.house, "pose": [float(xy[i, 0]), float(xy[i, 1]), height, float(yaw[i])],
                               "depth_ahead_m": float(ahead[i])})
            if len(starts) == count:
                break
        return starts
