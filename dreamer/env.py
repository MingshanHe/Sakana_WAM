"""Batched plan-following flight env for DreamerV3 (env: conda `sakana_hyperdrone`).

N Crazyflies fly at once in one ProcTHOR house. Each one has:
- L2F dynamics (HyperDrone, CPU build). It uses `control="position"`, so the built-in RAPTOR position controller turns
  a world (x, y, z, yaw) setpoint into motor commands.
- Actions in [-1, 1]^4. They are body-frame velocity commands (v_forward, v_left, v_up, yaw_rate), scaled by `ACT_SCALE`.
  The env integrates them into a "carrot" setpoint at the 100 Hz dynamics rate; the carrot is kept within `CARROT_MAX`
  of the drone. The policy runs at 10 Hz (`SUBSTEPS` dynamics steps per action).
- A level onboard camera (gimbal-stabilised, yaw only, like the Wan / DiffWAM experiments): 64x64 RGB + depth, 90 deg.
- A collision rig: the same camera rendered forward / left / back / right / up / down. A crash means a horizontal view
  has something closer than `CRASH_DIST` in its central band, or the up / down view has something closer than
  `CRASH_DIST_VERT`. The dynamics have no collision geometry, so the renders are the only way to detect contact.
- A plan: world poses (T, 4) at times `plan_t`, from the gen2pose teacher of a Wan imagination or a synthetic
  primitive. The observation holds the next `N_WP` plan waypoints (time-indexed, every `WP_DT` s) in the drone's yaw
  frame.

Observation per drone:
  image  (64, 64, 4) uint8: RGB + depth (0-8 m -> 0-255, sqrt-scaled so near range has more resolution)
  vector (VEC_DIM,) float32: body velocity (3), yaw rate, roll, pitch, height, has_plan, waypoints N_WP x (dx, dy, dz,
         sin dyaw, cos dyaw)
Reward per step in [-1, 1]: 0.7 exp(-d^2 / 2 0.4^2) + 0.3 exp(-dyaw^2 / 2 0.3^2), where d is the distance to the
plan pose at the current time. A crash gives -1 and ends the episode (is_terminal).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "follow"))
from scenes import HouseCamera  # noqa: E402  (also puts the env's CMake on PATH)

IMG = 64
HFOV = 90.0
CTRL_DT = 0.1
SUBSTEPS = 10  # dynamics dt is 0.01 s
ACT_SCALE = np.array([1.5, 1.0, 0.75, 1.2], np.float32)  # m/s forward, left, up; rad/s yaw
CARROT_MAX = 1.0
CRASH_DIST = 0.25
CRASH_DIST_VERT = 0.15
N_WP = 8
WP_DT = 0.25
VEC_DIM = 8 + N_WP * 5
DEPTH_MAX = 8.0
VIEWS = 6  # forward, left, back, right, up, down


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_wxyz_to_rpy(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def yaw_quat(yaw):
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1).astype(np.float32)


def encode_depth(d):
    d = np.clip(np.where(d > 0, d, DEPTH_MAX), 0, DEPTH_MAX)
    return (np.sqrt(d / DEPTH_MAX) * 255).astype(np.uint8)


def decode_depth(u8):
    return (u8.astype(np.float32) / 255) ** 2 * DEPTH_MAX


def plan_at(plan_t, plan_p, t):
    """Interpolate a plan (T,), (T, 4) at times t (any shape); holds the last pose after the end."""
    t = np.clip(t, plan_t[0], plan_t[-1])
    out = np.stack([np.interp(t, plan_t, plan_p[:, k]) for k in range(3)], -1)
    yaw = np.unwrap(plan_p[:, 3])
    out = np.concatenate([out, np.interp(t, plan_t, yaw)[..., None]], -1)
    return out


class FlightEnv:
    def __init__(self, house: int, num: int = 64, seed: int = 0, fidelity: str = "veryhigh"):
        from hyperdrone import dynamics

        self.house, self.num = house, num
        self.rng = np.random.default_rng(seed)
        self.cam = HouseCamera(house, width=IMG, height=IMG, hfov_deg=HFOV, batch=num * VIEWS, fidelity=fidelity)
        self.sim = dynamics.Sim(num, parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, seed=seed,
                                device="cpu", control="position")
        assert abs(self.sim.dt * SUBSTEPS - CTRL_DT) < 1e-6
        self.sim.reset()
        self.floor = float(self.cam.center[2] - self.cam.half[2])
        self.ceiling = float(self.cam.center[2] + self.cam.half[2])
        self.carrot = np.zeros((num, 4), np.float32)
        self.t = np.zeros(num)  # time since plan start
        self.plans = [None] * num  # (plan_t, plan_p world)
        self.has_plan = np.ones(num, bool)
        self.prev_action = np.zeros((num, 4), np.float32)

    # ---------------------------------------------------------------- state
    def pose(self):
        p = self.sim.state["position"]
        roll, pitch, yaw = quat_wxyz_to_rpy(self.sim.state["orientation"])
        return np.concatenate([p, yaw[:, None]], 1), roll, pitch

    def reset(self, idx, starts, plans, has_plan=None):
        """Put drones `idx` at world poses `starts` (k, 4) at rest, with plans [(plan_t, plan_p world)]."""
        idx = np.asarray(idx)
        pos, ori = self.sim.state["position"], self.sim.state["orientation"]
        vel, ang = self.sim.state["linear_velocity"], self.sim.state["angular_velocity"]
        mask = np.zeros(self.num, bool)
        mask[idx] = True
        self.sim.reset(mask)
        motor = self.sim.state["motor_command"]
        pos[idx] = starts[:, :3]
        ori[idx] = yaw_quat(starts[:, 3])
        vel[idx] = 0
        ang[idx] = 0
        for name, v in (("position", pos), ("orientation", ori), ("linear_velocity", vel), ("angular_velocity", ang),
                        ("motor_command", motor)):
            self.sim.state[name] = v
        self.carrot[idx] = starts
        self.t[idx] = 0
        self.prev_action[idx] = 0
        for j, i in enumerate(idx):
            self.plans[i] = plans[j]
            self.has_plan[i] = True if has_plan is None else bool(has_plan[j])

    def set_plan(self, idx, plans):
        for j, i in enumerate(np.asarray(idx)):
            self.plans[i] = plans[j]
        self.t[np.asarray(idx)] = 0

    # ---------------------------------------------------------------- step
    def step(self, action):
        """action (num, 4) in [-1, 1] -> obs dict, reward, terminal, info."""
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        cmd = a * ACT_SCALE
        dt = self.sim.dt
        for _ in range(SUBSTEPS):
            c, s = np.cos(self.carrot[:, 3]), np.sin(self.carrot[:, 3])
            self.carrot[:, 0] += (c * cmd[:, 0] - s * cmd[:, 1]) * dt
            self.carrot[:, 1] += (s * cmd[:, 0] + c * cmd[:, 1]) * dt
            self.carrot[:, 2] += cmd[:, 2] * dt
            self.carrot[:, 3] += cmd[:, 3] * dt
            pos = self.sim.state["position"]
            off = self.carrot[:, :3] - pos
            n = np.linalg.norm(off, axis=1, keepdims=True)
            self.carrot[:, :3] = pos + off * np.minimum(1.0, CARROT_MAX / np.maximum(n, 1e-6))
            self.sim.step(self.carrot)
        self.t += CTRL_DT
        self.prev_action = a
        return self.observe()

    def observe(self):
        pose, roll, pitch = self.pose()
        rgb, depth = self.render(pose)
        n = self.num
        h = IMG
        band = slice(h // 2 - h // 8, h // 2 + h // 8)
        dh = np.where(depth[: 4 * n] > 0, depth[: 4 * n], 1e3).reshape(4, n, h, h)[:, :, band].reshape(4, n, -1).min(-1).min(0)
        dv = np.where(depth[4 * n:] > 0, depth[4 * n:], 1e3).reshape(2, n, h, h)[:, :, h // 4: 3 * h // 4, h // 4: 3 * h // 4]
        dv = dv.reshape(2, n, -1).min(-1).min(0)
        crash = (dh < CRASH_DIST) | (dv < CRASH_DIST_VERT) | (pose[:, 2] < self.floor + 0.1) | (pose[:, 2] > self.ceiling - 0.1)
        crash |= ~np.isfinite(pose).all(1)
        image = np.concatenate([rgb[:n], encode_depth(depth[:n])[..., None]], -1)
        vec, track = self.vector(pose, roll, pitch)
        d, dyaw = track
        reward = 0.7 * np.exp(-d ** 2 / (2 * 0.4 ** 2)) + 0.3 * np.exp(-dyaw ** 2 / (2 * 0.3 ** 2))
        reward = np.where(self.has_plan, reward, 0.0)
        reward = np.where(crash, -1.0, reward).astype(np.float32)
        info = {"pose": pose.astype(np.float32), "carrot": self.carrot.copy(), "track_d": d.astype(np.float32),
                "track_dyaw": dyaw.astype(np.float32), "clear_h": dh.astype(np.float32), "clear_v": dv.astype(np.float32)}
        return {"image": image, "vector": vec}, reward, crash, info

    def render(self, pose):
        n = self.num
        rig = np.concatenate([pose + np.array([0, 0, 0, k * np.pi / 2]) for k in range(4)])
        rgb, depth = self.cam.render(rig)  # horizontal views first
        dvert = self._render_vertical(pose)
        return rgb, np.concatenate([depth, dvert])

    def _render_vertical(self, pose):
        """Depth straight up and straight down (camera look_at along +-z, nudged forward to avoid a degenerate up)."""
        rows = []
        for sign in (1, -1):
            for x, y, z, yaw in pose:
                target = (x + 1e-3 * math.cos(yaw), y + 1e-3 * math.sin(yaw), z + sign)
                rows.append(np.asarray(self.cam.renderer.camera((x, y, z), target, fov=self.cam.fov_arg), np.float32))
        rows = np.stack(rows).reshape(2 * self.num, -1)
        b = self.cam.batch
        rows = np.concatenate([rows, np.repeat(rows[-1:], b - len(rows), 0)]) if len(rows) < b else rows
        self.cam.renderer.set_cameras(rows)
        self.cam.renderer.render()
        d = np.array(self.cam.renderer.outputs[self.cam._depth].read())[: 2 * self.num]
        return np.nan_to_num(d, nan=0.0, posinf=0.0)

    def vector(self, pose, roll, pitch):
        n = self.num
        v = self.sim.state["linear_velocity"]
        w = self.sim.state["angular_velocity"]
        c, s = np.cos(pose[:, 3]), np.sin(pose[:, 3])
        vb = np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], 1)
        base = np.concatenate([vb, w[:, 2:3], roll[:, None], pitch[:, None], (pose[:, 2:3] - self.floor),
                               self.has_plan[:, None].astype(np.float32)], 1)
        wps = np.zeros((n, N_WP, 5), np.float32)
        d = np.zeros(n)
        dyaw = np.zeros(n)
        for i in range(n):
            pt, pp = self.plans[i]
            now = plan_at(pt, pp, self.t[i])
            d[i] = np.linalg.norm(now[:3] - pose[i, :3])
            dyaw[i] = abs(wrap(now[3] - pose[i, 3]))
            if not self.has_plan[i]:
                continue
            q = plan_at(pt, pp, self.t[i] + WP_DT * np.arange(1, N_WP + 1))
            dx, dy = q[:, 0] - pose[i, 0], q[:, 1] - pose[i, 1]
            wps[i, :, 0] = c[i] * dx + s[i] * dy
            wps[i, :, 1] = -s[i] * dx + c[i] * dy
            wps[i, :, 2] = q[:, 2] - pose[i, 2]
            wps[i, :, 3] = np.sin(q[:, 3] - pose[i, 3])
            wps[i, :, 4] = np.cos(q[:, 3] - pose[i, 3])
        vec = np.concatenate([base, wps.reshape(n, -1)], 1).astype(np.float32)
        return vec, (d, dyaw)


if __name__ == "__main__":
    import time

    env = FlightEnv(13, num=64)
    print("floor", env.floor, "ceiling", env.ceiling, "center", env.cam.center, "half", env.cam.half)
    starts = env.cam.find_starts(np.random.default_rng(0), count=8)
    print(len(starts), "starts")
    st = np.array([starts[i % len(starts)]["pose"] for i in range(64)], np.float32)
    plan_t = np.linspace(0, 5, 39)
    plans = []
    for p in st:
        rel = np.stack([plan_t * 0.8, 0 * plan_t, 0 * plan_t, 0 * plan_t], 1)
        c, s = math.cos(p[3]), math.sin(p[3])
        w = np.stack([p[0] + c * rel[:, 0], p[1] + s * rel[:, 0], p[2] + rel[:, 2], p[3] + rel[:, 3]], 1)
        plans.append((plan_t, w))
    env.reset(np.arange(64), st, plans)
    obs, r, done, info = env.observe()
    print("obs", obs["image"].shape, obs["vector"].shape, "reward", r[:4], "crash", done.sum())
    t0 = time.time()
    a = np.zeros((64, 4), np.float32)
    a[:, 0] = np.linspace(0, 1, 64)
    for k in range(30):
        obs, r, done, info = env.step(a)
    dt = (time.time() - t0) / 30
    print(f"{dt * 1000:.1f} ms / step for 64 drones; crashes {done.sum()}, reward mean {r.mean():.3f}")
    print("pose drift", np.round(info["pose"][::16] - st[::16], 2))
    print("clear_h", np.round(info["clear_h"][::8], 2), "clear_v", np.round(info["clear_v"][::8], 2))
    import imageio.v2 as imageio
    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    grid = np.concatenate([np.concatenate([obs["image"][i, ..., :3], np.repeat(obs["image"][i, ..., 3:], 3, -1)], 0)
                           for i in range(0, 64, 8)], 1)
    imageio.imwrite(os.path.join(HERE, "outputs", "env_obs.png"), grid)
