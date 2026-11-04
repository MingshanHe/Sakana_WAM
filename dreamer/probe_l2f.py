"""Probe L2F position control (RAPTOR) in HyperDrone: hover, then track a moving setpoint."""
import math, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "follow"))
import scenes  # noqa: F401  (PATH fix for CMake)
import numpy as np
from hyperdrone import dynamics

n = 8
t0 = time.time()
sim = dynamics.Sim(n, parameters={"dynamics": {"dynamics": {"model": "crazyflie"}}}, seed=0, device="cpu", control="position")
print("build", time.time() - t0, "dt", sim.dt, "act", sim.action_dim, sim.action_low, sim.action_high, "obs", sim.observation_dim)
sim.reset()
print("state keys", {k: sim.state[k][0] for k in sim.state.keys()})
p0 = sim.state["position"].copy()
# each drone: different constant velocity command, integrated into a carrot setpoint
vel = np.zeros((n, 4), np.float32)
vel[1, 0] = 1.0; vel[2, 1] = 1.0; vel[3, 2] = 0.5; vel[4, 3] = 1.0; vel[5, 0] = 1.5; vel[6, 0] = -1.0; vel[7, 2] = -0.5
target = np.concatenate([p0, np.zeros((n, 1))], 1).astype(np.float32)
steps = int(3.0 / sim.dt)
log = []
t0 = time.time()
for k in range(steps):
    target[:, :3] += vel[:, :3] * sim.dt
    target[:, 3] += vel[:, 3] * sim.dt
    sim.step(target)
    if k % int(0.5 / sim.dt) == 0:
        log.append((k * sim.dt, sim.state["position"].copy(), sim.state["linear_velocity"].copy()))
print("sim time", time.time() - t0, "for", steps, "steps")
for t, p, v in log:
    print(f"t={t:.2f}", np.round(p - p0, 2).tolist()[:8])
q = sim.state["orientation"]
yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
print("final err", np.round(sim.state["position"] - target[:, :3], 3).tolist())
print("yaw", np.round(yaw, 2), "target yaw", np.round(target[:, 3], 2))
