"""Reference plans for the flight env: gen2pose teacher plans of Wan2.2 imaginations, and synthetic primitives.

A plan is (plan_t (T,), plan_p (T, 4) world x, y, z, yaw). Teacher plans are the 39 DiffWAM slots over 5 s.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "diffwam", "data", "samples")
SAMPLES_EVAL = os.path.join(HERE, "..", "diffwam", "data", "samples_eval")
SLOT_T = np.arange(39) * 5.0 / 38.0
MOTIONS = ("forward", "slow_forward", "turn_left", "turn_right", "pan_left", "pan_right", "ascend", "descend",
           "approach_stop", "strafe_left", "strafe_right", "hover")


def rel_to_world(start, rel):
    """(T, 4) poses in the start's yaw frame (FLU) -> world."""
    c, s = math.cos(start[3]), math.sin(start[3])
    x = start[0] + c * rel[:, 0] - s * rel[:, 1]
    y = start[1] + s * rel[:, 0] + c * rel[:, 1]
    return np.stack([x, y, start[2] + rel[:, 2], start[3] + rel[:, 3]], 1).astype(np.float32)


def plausible(rel):
    speed = np.linalg.norm(np.diff(rel[:, :3], axis=0), axis=1) / (5.0 / 38.0)
    return bool(speed.max() < 2.5 and np.linalg.norm(rel[-1, :3]) < 7.0)


def load_teacher(root=SAMPLES):
    """Teacher plans: list of dicts id, house, motion, start (4,), rel (39, 4), plausible."""
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.exists(os.path.join(d, "teacher.npz")):
            continue
        meta = json.load(open(os.path.join(d, "sample.json")))
        rel = np.load(os.path.join(d, "teacher.npz"))["pose"].astype(np.float32)
        out.append({"id": meta["id"], "house": meta["house"], "motion": meta["motion"],
                    "start": np.array(meta["pose"], np.float32), "rel": rel, "plausible": plausible(rel)})
    return out


def synthetic_rel(rng: np.random.Generator, motion: str | None = None, duration: float = 5.0):
    """A smooth primitive in the start frame, sampled on SLOT_T-like spacing over `duration` s."""
    motion = motion or rng.choice(MOTIONS, p=_P)
    t = np.linspace(0, duration, int(round(duration / (5.0 / 38.0))) + 1)
    ramp = np.clip(t / rng.uniform(0.4, 1.2), 0, 1)
    v_f = v_l = v_u = w = np.zeros_like(t)
    if motion in ("forward", "turn_left", "turn_right", "ascend", "descend"):
        v_f = rng.uniform(0.5, 1.4) * ramp
    if motion == "slow_forward":
        v_f = rng.uniform(0.2, 0.5) * ramp
    if motion in ("turn_left", "turn_right"):
        w = (1 if motion == "turn_left" else -1) * rng.uniform(0.2, 0.7) * ramp
    if motion in ("pan_left", "pan_right"):
        w = (1 if motion == "pan_left" else -1) * rng.uniform(0.25, 0.9) * ramp
        v_f = rng.uniform(0, 0.2) * ramp
    if motion in ("ascend", "descend"):
        v_u = (1 if motion == "ascend" else -1) * rng.uniform(0.1, 0.35) * ramp
    if motion == "approach_stop":
        stop = rng.uniform(2.0, 4.0)
        v_f = rng.uniform(0.5, 1.2) * ramp * np.clip((stop - t) / 0.8, 0, 1)
    if motion in ("strafe_left", "strafe_right"):
        v_l = (1 if motion == "strafe_left" else -1) * rng.uniform(0.3, 0.8) * ramp
    # mild random curvature / height wobble on everything but hover
    if motion != "hover":
        w = w + rng.normal(0, 0.08) * ramp
        v_u = v_u + rng.normal(0, 0.05) * ramp
    dt = np.diff(t, prepend=0.0)
    yaw = np.cumsum(w * dt)
    x = np.cumsum((np.cos(yaw) * v_f - np.sin(yaw) * v_l) * dt)
    y = np.cumsum((np.sin(yaw) * v_f + np.cos(yaw) * v_l) * dt)
    z = np.cumsum(v_u * dt)
    return t.astype(np.float32), np.stack([x, y, z, yaw], 1).astype(np.float32), str(motion)


_W = {"forward": 3, "slow_forward": 1.5, "turn_left": 2, "turn_right": 2, "pan_left": 1.5, "pan_right": 1.5,
      "ascend": 1.5, "descend": 1.5, "approach_stop": 2, "strafe_left": 1, "strafe_right": 1, "hover": 0.5}
_P = np.array([_W[m] for m in MOTIONS]) / sum(_W.values())


def instruction_prior(teacher, motion):
    """Mean training teacher trajectory of an instruction type (the 'instruction prior' baseline plan)."""
    rels = [p["rel"] for p in teacher if p["motion"] == motion and p["plausible"]]
    return np.mean(rels, 0).astype(np.float32) if rels else np.zeros((39, 4), np.float32)
