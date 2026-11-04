"""Collect DreamerV3 training data in HyperDrone (env: conda `sakana_hyperdrone`).

Same dataset definition as diffwam/: the 80 ProcTHOR training houses of diffwam/data/samples, their take-off poses
and their gen2pose teacher plans (Wan2.2 imaginations). Evaluation houses (diffwam/data/samples_eval) are never used.
Per house, `--num` drones fly in parallel for `--steps` control steps (10 Hz); a drone that crashes or times out is
reset to a new start + plan. Plans: the house's teacher plans (at their own take-off pose) or synthetic primitives at
take-off poses found in the house; some episodes switch to a new plan mid-flight (as replanning will).

Behaviour policies (per episode):
  track   pure-pursuit on the plan with OU noise (sigma 0.05-0.5) and, sometimes, a 1-2 s random-command excursion
  random  OU random commands
  safe    the tracker plus reactive avoidance from the onboard depth (slow down, turn to the freer side, keep off
          floor / ceiling); round 1+ only
The output is one stream per drone, in DreamerV3 replay format (image, vector, action, reward, is_first, is_terminal)
plus pose / track error / plan-source columns for analysis: data/replay/<split>/h<house>_r<round>/<key>.npy.

    python collect.py --houses 13 14 --steps 200        # or --shard i --nshards n for all 80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from env import ACT_SCALE, CTRL_DT, IMG, FlightEnv, decode_depth, plan_at, wrap  # noqa: E402
from plans import SLOT_T, load_teacher, rel_to_world, synthetic_rel  # noqa: E402

VAL_HOUSES_EVERY = 10
MIX = [(0.8, 0.2, 0.0), (0.45, 0.2, 0.35)]  # per round: track, random, safe-track  # every 10th training house (sorted) is a world-model validation house


class Behaviour:
    def __init__(self, num, rng):
        self.rng = rng
        self.kind = np.zeros(num, int)  # 0 track, 1 random
        self.sigma = np.zeros(num)
        self.look = np.full(num, 0.5)
        self.ou = np.zeros((num, 4))
        self.excursion = np.zeros(num, int)  # remaining steps of a random excursion

    mix = np.array([0.8, 0.2, 0.0])  # probabilities of track / random / safe-track per episode

    def reset(self, i):
        r = self.rng
        self.kind[i] = r.choice(3, p=self.mix)
        self.sigma[i] = r.uniform(0.05, 0.5) if self.kind[i] == 0 else r.uniform(0.3, 0.8)
        self.look[i] = r.uniform(0.3, 0.8)
        self.ou[i] = 0
        self.excursion[i] = 0

    def act(self, env, obs=None):
        pose, _, _ = env.pose()
        if obs is not None:  # reactive avoidance for the "safe" tracker (kind 2), from the onboard depth
            d = decode_depth(obs["image"][..., 3])
            band = d[:, IMG // 3: 2 * IMG // 3]
            front = np.percentile(band[:, :, IMG // 3: 2 * IMG // 3].reshape(env.num, -1), 5, axis=1)
            left = np.median(band[:, :, : IMG // 3].reshape(env.num, -1), axis=1)
            right = np.median(band[:, :, 2 * IMG // 3:].reshape(env.num, -1), axis=1)
        n = env.num
        a = np.zeros((n, 4), np.float32)
        self.ou += 0.15 * (-self.ou) + self.sigma[:, None] * self.rng.normal(size=(n, 4)) * np.sqrt(0.3)
        start_exc = (self.kind == 0) & (self.excursion == 0) & (self.rng.random(n) < 0.01)
        self.excursion[start_exc] = self.rng.integers(10, 20, start_exc.sum())
        for i in range(n):
            pt, pp = env.plans[i]
            if self.kind[i] == 1 or self.excursion[i] > 0:
                a[i] = np.clip(self.ou[i] * 1.5, -1, 1)
                self.excursion[i] = max(0, self.excursion[i] - 1)
                continue
            q = plan_at(pt, pp, env.t[i] + self.look[i])
            dx, dy, dz = q[0] - pose[i, 0], q[1] - pose[i, 1], q[2] - pose[i, 2]
            c, s = np.cos(pose[i, 3]), np.sin(pose[i, 3])
            v = np.array([c * dx + s * dy, -s * dx + c * dy, dz, wrap(q[3] - pose[i, 3])]) / self.look[i]
            a[i] = np.clip(v / ACT_SCALE + self.ou[i], -1, 1)
            if self.kind[i] == 2 and obs is not None:
                slow = np.clip((front[i] - 0.5) / 0.8, 0, 1)
                a[i, 0] = min(a[i, 0], slow) if front[i] < 1.3 else a[i, 0]
                if front[i] < 1.3:
                    a[i, 3] = np.clip(a[i, 3] + (0.8 if left[i] > right[i] else -0.8) * (1 - slow), -1, 1)
                z = pose[i, 2]
                if z > env.ceiling - 0.6:
                    a[i, 2] = min(a[i, 2], -0.2)
                if z < env.floor + 0.5:
                    a[i, 2] = max(a[i, 2], 0.2)
        return a


def house_list():
    teacher = load_teacher()
    houses = sorted({p["house"] for p in teacher})
    return teacher, houses


def split_of(house, houses):
    return "val" if houses.index(house) % VAL_HOUSES_EVERY == VAL_HOUSES_EVERY - 1 else "train"


def collect_house(house, teacher, out_path, num, steps, seed):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    env = FlightEnv(house, num=num, seed=seed)
    mine = [p for p in teacher if p["house"] == house and p["plausible"]]
    found = [np.array(s["pose"], np.float32) for s in env.cam.find_starts(rng, count=16)]
    take_off = [p["start"] for p in mine] + found
    beh = Behaviour(num, rng)
    beh.mix = np.array(MIX[min(seed // 1000, len(MIX) - 1)], float)
    ep_len = np.zeros(num, int)
    ep_max = np.zeros(num, int)
    replan_at = np.full(num, -1)
    source = np.zeros(num, int)  # 0 teacher, 1 synthetic
    motion = np.array([""] * num, dtype=object)

    def new_episode(idx):
        starts, plans = [], []
        for i in idx:
            if mine and rng.random() < 0.45:
                p = mine[rng.integers(len(mine))]
                start = p["start"].copy()
                start[:3] += rng.normal(0, [0.15, 0.15, 0.05])
                start[3] += rng.normal(0, 0.1)
                rel, pt, source[i], motion[i] = p["rel"], SLOT_T, 0, p["motion"]
                world = rel_to_world(start, rel)
            else:
                start = take_off[rng.integers(len(take_off))].copy()
                start[:3] += rng.normal(0, [0.3, 0.3, 0.1])
                start[2] = np.clip(start[2], env.floor + 0.6, env.ceiling - 0.6)
                start[3] = rng.uniform(-np.pi, np.pi) if rng.random() < 0.25 else start[3] + rng.normal(0, 0.3)
                pt, rel, motion[i] = synthetic_rel(rng)
                source[i] = 1
                world = rel_to_world(start, rel)
            starts.append(start)
            plans.append((pt, world))
            beh.reset(i)
            ep_len[i] = 0
            ep_max[i] = rng.integers(60, 101)
            replan_at[i] = rng.integers(25, 50) if rng.random() < 0.3 else -1
        env.reset(np.asarray(idx), np.stack(starts).astype(np.float32), plans)

    new_episode(np.arange(num))
    obs, rew, term, info = env.observe()
    first = np.ones(num, bool)
    keys = {"image": [], "vector": [], "action": [], "reward": [], "is_first": [], "is_terminal": [], "pose": [],
            "track_d": [], "source": [], "behaviour": [], "clear_h": [], "clear_v": []}
    crashes = episodes = 0
    for k in range(steps):
        a = beh.act(env, obs)
        end = term | (ep_len >= ep_max)
        a[end] = 0
        for name, v in (("image", obs["image"]), ("vector", obs["vector"]), ("action", a), ("reward", rew),
                        ("is_first", first.copy()), ("is_terminal", term.copy()), ("pose", info["pose"]),
                        ("track_d", info["track_d"]), ("source", source.copy()), ("behaviour", beh.kind.copy()),
                        ("clear_h", info["clear_h"]), ("clear_v", info["clear_v"])):
            keys[name].append(v)
        crashes += int(term.sum())
        episodes += int(end.sum())
        first = end.copy()
        if end.any():
            new_episode(np.flatnonzero(end))
        rp = (~end) & (ep_len == replan_at)
        for i in np.flatnonzero(rp):  # switch to a new synthetic plan from where the drone is now
            pose, _, _ = env.pose()
            pt, rel, motion[i] = synthetic_rel(rng)
            env.set_plan([i], [(pt, rel_to_world(pose[i], rel))])
            source[i] = 2
        ep_len += 1
        obs, rew, term, info = env.step(a)
    data = {k: np.stack(v, 1) for k, v in keys.items()}  # (num, steps, ...)
    tmp = out_path + ".tmp"
    os.makedirs(tmp, exist_ok=True)
    for k, v in data.items():  # one .npy per key, so training can memory-map the streams
        np.save(os.path.join(tmp, k + ".npy"), v)
    os.replace(tmp, out_path)
    stats = {"house": house, "frames": int(num * steps), "episodes_ended": episodes, "crashes": crashes,
             "teacher_plans": len(mine), "reward_mean": float(data["reward"].mean()),
             "track_d_median": float(np.median(data["track_d"])), "seconds": time.time() - t0}
    print(json.dumps(stats), flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--houses", type=int, nargs="*")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--num", type=int, default=64)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--round", type=int, default=0, help="collection round (new seeds, new file names)")
    ap.add_argument("--out", default=os.path.join(HERE, "data", "replay"))
    args = ap.parse_args()
    teacher, houses = house_list()
    todo = args.houses or houses[args.shard::args.nshards]
    for h in todo:
        path = os.path.join(args.out, split_of(h, houses), f"h{h:03d}_r{args.round}")
        if os.path.exists(path):
            continue
        collect_house(h, teacher, path, args.num, args.steps, seed=1000 * args.round + h)


if __name__ == "__main__":
    main()
