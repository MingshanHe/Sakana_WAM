"""Full workflow in HyperDrone with L2F dynamics: Wan2.2 / DiffWAM global plan + DreamerV3 local planner.

Test set: the 20 DiffWAM evaluation samples (10 scenarios x 2 Wan seeds, evaluation houses never used in training).
Each arm flies `--repeats` drones per sample for `--seconds` s (plans cover 5 s, then hold their last pose):

  teacher+track     slow path: full Wan video -> gen2pose -> pure-pursuit tracker (no Dreamer)
  diffwam+track     fast path: DiffWAM features -> readout head -> pure-pursuit tracker (no Dreamer)
  prior+dreamer     instruction-prior plan (mean training trajectory of the instruction type) -> Dreamer actor
  diffwam+dreamer   DiffWAM plan -> Dreamer actor (the combined workflow)
  diffwam+dreamer+mpc  the same, with the world-model shield (best of K candidate actions by imagined return)
  openfly+...       OpenFly-Agent v1 (vla/openfly_server.py) as the global planner, called every --openfly-every s;
                    each discrete action becomes a --openfly-step-time s plan from the current pose
  hover+dreamer     control: the plan is "hover at the start"; shows what Dreamer does on its own

Every arm uses the same dynamics, renderer and crash rule (env.py). Motion vs instruction uses the rules of
../follow/analyze_follow.py on the flown path up to the end or the crash.

    python deploy_combined.py --ckpt data/runs/ac200m/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "diffwam"))
sys.path.insert(0, os.path.join(HERE, "..", "follow"))
from env import ACT_SCALE, CTRL_DT, FlightEnv, plan_at, wrap  # noqa: E402
from plans import SAMPLES_EVAL, SLOT_T, instruction_prior, load_teacher, rel_to_world  # noqa: E402
from analyze_follow import FLOWN_OK  # noqa: E402

ARMS = ("teacher+track", "diffwam+track", "prior+dreamer", "diffwam+dreamer", "diffwam+dreamer+mpc")
OPENFLY_ARMS = ("openfly+track", "openfly+dreamer", "openfly+dreamer+mpc")  # OpenFly-Agent v1 as the global planner
EVAL_TO_TRAIN_MOTION = {"forward": "forward", "hallway": "forward", "doorway": "forward", "approach": "forward",
                        "approach+stop": "approach_stop", "turn left": "turn_left", "turn right": "turn_right",
                        "ascend": "ascend", "descend": "descend", "strafe right": "strafe_right"}


def diffwam_plans(head_ckpt):
    """DiffWAM head predictions for the evaluation samples (relative (39, 4)) and the head latency."""
    from train_head import ReadoutHead, batchify, load_split, to_poses

    ck = torch.load(head_ckpt, weights_only=False)
    model = ReadoutHead(k=len(ck["evals"])).cuda().eval()
    model.load_state_dict(ck["model"])
    out = {}
    for s in load_split(SAMPLES_EVAL, ck["evals"]):
        z, a0, geo, s0, _ = batchify([s])
        with torch.no_grad():
            out[s["id"]] = {"rel": to_poses(model(z, a0, geo), s0)[0].cpu().numpy(), "teacher": s["pose"]}
    return out


def track_action(env, i, look=0.5):
    pose, _, _ = env.pose()
    pt, pp = env.plans[i]
    q = plan_at(pt, pp, env.t[i] + look)
    dx, dy, dz = q[0] - pose[i, 0], q[1] - pose[i, 1], q[2] - pose[i, 2]
    c, s = math.cos(pose[i, 3]), math.sin(pose[i, 3])
    v = np.array([c * dx + s * dy, -s * dx + c * dy, dz, wrap(q[3] - pose[i, 3])]) / look
    return np.clip(v / ACT_SCALE, -1, 1)


def summarize(start, end):
    c, s = math.cos(start[3]), math.sin(start[3])
    d = end[:3] - start[:3]
    return {"forward_m": float(d[0] * c + d[1] * s), "left_m": float(-d[0] * s + d[1] * c), "up_m": float(d[2]),
            "yaw_deg": float(math.degrees(wrap(end[3] - start[3])))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--head", default=os.path.join(HERE, "..", "diffwam", "data", "head", "diffwam.pt"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out", default="")
    ap.add_argument("--max-houses", type=int, default=0)
    ap.add_argument("--replan-every", type=float, default=0.0,
                    help="seconds between replans (0 = one plan from the start frame). diffwam arms ask wan_server.py "
                         "for features of the current onboard view; the plan takes effect after the measured Wan "
                         "latency (simulated time). prior arms restart the instruction prior from the current pose.")
    ap.add_argument("--openfly-every", type=float, default=1.0,
                    help="openfly arms: seconds between OpenFly-Agent calls (first call at t = 0)")
    ap.add_argument("--openfly-step-time", type=float, default=1.0,
                    help="seconds over which one OpenFly action (e.g. forward 0.5 m, turn 30 deg) is spread as a plan")
    ap.add_argument("--openfly-scale", type=float, default=1 / 6, help="OpenFly 3 m outdoor unit -> 3 * scale m")
    ap.add_argument("--openfly-prompt", choices=("wan", "nav"), default="wan")
    ap.add_argument("--vla-root", default=os.path.join(HERE, "data", "vla_requests"))
    ap.add_argument("--wan-latency", type=float, default=0.0,
                    help="simulated Wan latency in s (0 = use the measured time of each request)")
    ap.add_argument("--wan-root", default=os.path.join(HERE, "data", "wan_requests"))
    ap.add_argument("--video", action="store_true", help="save onboard videos of the first repeat")
    args = ap.parse_args()
    from policy import DreamerPolicy

    tag = os.path.basename(os.path.dirname(args.ckpt)) + (f"_replan{args.replan_every:g}" if args.replan_every else "")
    use_openfly = any(a.startswith("openfly") for a in args.arms)
    if use_openfly:
        sys.path.append(os.path.join(HERE, "..", "vla"))
        from eval_zeroshot import ACTION_NAMES, NAV, ask_openfly, openfly_move

        tag += f"_openfly{args.openfly_every:g}s_{args.openfly_prompt}"
        while not os.path.exists(os.path.join(args.vla_root, "READY")):
            print("waiting for vla/openfly_server.py ...", flush=True)
            time.sleep(10)
    out_dir = args.out or os.path.join(HERE, "data", "deploy", tag)
    replanner = None
    if args.replan_every:
        from replan import Replanner

        replanner = Replanner(args.head, args.wan_root)
        while not replanner.server_ready():
            print("waiting for wan_server.py ...", flush=True)
            time.sleep(10)
    os.makedirs(out_dir, exist_ok=True)
    samples = json.load(open(os.path.join(SAMPLES_EVAL, "index.json")))
    teacher_train = load_teacher()
    dw = diffwam_plans(args.head)
    steps = int(round(args.seconds / CTRL_DT))
    houses = sorted({s["house"] for s in samples})[: args.max_houses or None]
    rows = []
    for house in houses:
        here = [s for s in samples if s["house"] == house and s["id"] in dw]
        jobs = [(s, arm, r) for s in here for arm in args.arms for r in range(args.repeats)]
        n = len(jobs)
        env = FlightEnv(house, num=n, seed=house)
        pol = DreamerPolicy(args.ckpt, num=n)
        starts, plans = [], []
        for s, arm, r in jobs:
            start = np.array(s["pose"], np.float32)
            if arm.startswith("openfly") or arm.startswith("hover"):
                rel = np.zeros((39, 4), np.float32)  # hover (control) / until OpenFly's first action arrives
            elif arm.startswith("teacher"):
                rel = dw[s["id"]]["teacher"]
            elif arm.startswith("diffwam"):
                rel = dw[s["id"]]["rel"]
            else:
                rel = instruction_prior(teacher_train, EVAL_TO_TRAIN_MOTION[s["motion"]])
            starts.append(start)
            plans.append((SLOT_T, rel_to_world(start, rel)))
        env.reset(np.arange(n), np.stack(starts), plans)
        obs, rew, term, info = env.observe()
        first = np.ones(n, bool)
        crashed = np.zeros(n, bool)
        crash_t = np.full(n, np.nan)
        end_pose = info["pose"].copy()
        track = [[] for _ in range(n)]
        paths = [[p] for p in info["pose"]]
        frames = [[] for _ in range(n)]
        for i, (s, arm, r) in enumerate(jobs):
            s.setdefault("instruction", "")
        uses_dreamer = np.array(["dreamer" in arm for _, arm, _ in jobs])
        mpc = np.array(["mpc" in arm for _, arm, _ in jobs])
        dreamer_ms = []
        pending = {}  # drone -> (activation step, world plan, time offset)
        wan_s = []
        big_cam = None
        is_of = np.array([arm.startswith("openfly") for _, arm, _ in jobs])
        of_every = int(round(args.openfly_every / CTRL_DT))
        of_hist = [[] for _ in range(n)]
        of_done = np.zeros(n, bool)
        of_actions = [[] for _ in range(n)]
        of_s = []
        if replanner is not None or use_openfly:
            from scenes import HouseCamera

            big_cam = HouseCamera(house, batch=8)
        every = int(round(args.replan_every / CTRL_DT)) if args.replan_every else 0
        for k in range(steps):
            if every and k and k % every == 0:
                pose_now, _, _ = env.pose()
                reqs = []
                for i, (s, arm, r) in enumerate(jobs):
                    if crashed[i] or arm.startswith(("teacher", "openfly", "hover")):
                        continue
                    if arm.startswith("prior"):
                        rel = instruction_prior(teacher_train, EVAL_TO_TRAIN_MOTION[s["motion"]])
                        env.plans[i] = (SLOT_T, rel_to_world(pose_now[i], rel))
                        env.t[i] = 0.0
                        continue
                    rgb, depth = big_cam.render(pose_now[i:i + 1])
                    d = replanner.request(f"{s['id']}_{arm.replace('+', '-')}_r{r}_k{k}", rgb[0], depth[0],
                                          s["instruction"], s["seed"])
                    reqs.append((i, d, depth[0], pose_now[i].copy()))
                for i, d, depth, p0 in reqs:
                    feats = replanner.wait(d)
                    rel = replanner.plan(feats, depth)
                    lat = args.wan_latency or float(feats["seconds"])
                    wan_s.append(lat)
                    pending[i] = (k + int(math.ceil(lat / CTRL_DT)), (SLOT_T, rel_to_world(p0, rel)), lat)
            if use_openfly and k % of_every == 0:
                pose_now, _, _ = env.pose()
                for i in np.flatnonzero(is_of & ~crashed & ~of_done):
                    s, arm, r = jobs[i]
                    rgb, _ = big_cam.render(pose_now[i:i + 1])
                    of_hist[i].append(rgb[0])
                    hist = [of_hist[i][max(0, len(of_hist[i]) - 1 - j)] for j in range(3)]
                    text = s["instruction"] if args.openfly_prompt == "wan" else NAV[s["id"][:3]]
                    out = ask_openfly(args.vla_root, f"{s['id']}_{arm.replace('+', '-')}_r{r}_k{k}", hist, text)
                    of_actions[i].append(ACTION_NAMES[out["id"]])
                    of_s.append(out["seconds"])
                    p0 = pose_now[i].astype(float)
                    if out["id"] == 0:  # stop: hold here for the rest of the flight
                        of_done[i] = True
                        target = p0
                    else:
                        target = openfly_move(p0, out["id"], args.openfly_scale)
                        target[3] = p0[3] + math.atan2(math.sin(target[3] - p0[3]), math.cos(target[3] - p0[3]))
                    pt = np.array([0.0, args.openfly_step_time, args.openfly_step_time + 10.0])
                    pp = np.stack([p0, target, target]).astype(np.float32)
                    pending[i] = (k + int(math.ceil(out["seconds"] / CTRL_DT)), (pt, pp), float(out["seconds"]))
            for i in [i for i, v in pending.items() if v[0] <= k]:
                _, plan, lat = pending.pop(i)
                env.plans[i] = plan
                env.t[i] = lat
            a = np.zeros((n, 4), np.float32)
            for i, (s, arm, r) in enumerate(jobs):
                if not uses_dreamer[i]:
                    a[i] = track_action(env, i)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            a_d = pol.act(obs, first)
            t1 = time.perf_counter()
            a_m = None
            if mpc.any():  # the shield plans from the state just filtered by pol.act
                with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                    a_m = pol._shield(torch.as_tensor(a_d, device="cuda")).float().cpu().numpy()
            t2 = time.perf_counter()
            dreamer_ms.append(((t1 - t0) * 1000, (t2 - t1) * 1000))
            a[uses_dreamer] = a_d[uses_dreamer]
            if a_m is not None:
                a[mpc] = a_m[mpc]
            a[crashed] = 0
            pol.prev_a = torch.as_tensor(a, device="cuda")  # the RSSM conditions on what was really flown
            first[:] = False
            obs, rew, term, info = env.step(a)
            new = term & ~crashed
            crash_t[new] = (k + 1) * CTRL_DT
            end_pose[~crashed] = info["pose"][~crashed]
            crashed |= term
            for i in range(n):
                if not crashed[i] or new[i]:
                    track[i].append(float(info["track_d"][i]))
                    paths[i].append(info["pose"][i])
                if args.video and jobs[i][2] == 0 and (not crashed[i] or new[i]):
                    frames[i].append(obs["image"][i, ..., :3])
        for i, (s, arm, r) in enumerate(jobs):
            summ = summarize(np.array(s["pose"]), end_pose[i])
            yaws = np.array(paths[i])[:, 3]
            summ["yaw_deg"] = float(math.degrees(np.sum(wrap(np.diff(yaws)))))  # accumulated, not wrapped
            ok = bool(FLOWN_OK[s["motion"]](summ))
            path_len = float(np.sum(np.linalg.norm(np.diff(np.array(paths[i])[:, :3], axis=0), axis=1)))
            y = summ["yaw_deg"]
            sane = {"turn left": 15 < y <= 120, "turn right": -120 <= y < -15, "strafe right": abs(y) <= 45}.get(
                s["motion"], abs(y) <= 90)
            rows.append({"id": s["id"], "motion": s["motion"], "arm": arm, "repeat": r, **summ, "flown_ok": ok,
                         "success_sane": bool(ok and not crashed[i] and sane),
                         "path_m": path_len, "openfly_actions": of_actions[i] or None,
                         "collision": bool(crashed[i]), "crash_t": None if np.isnan(crash_t[i]) else float(crash_t[i]),
                         "success": ok and not crashed[i], "track_err_m": float(np.mean(track[i]))})
            if args.video and r == 0 and frames[i]:
                import imageio.v2 as imageio
                imageio.mimwrite(os.path.join(out_dir, f"{s['id']}_{arm}.mp4"),
                                 np.stack(frames[i]).repeat(4, 1).repeat(4, 2), fps=10, macro_block_size=1)
        ms = np.array(dreamer_ms)
        print(json.dumps({"house": house, "drones": n, "dreamer_ms": float(ms[:, 0].mean()),
                          "shield_ms": float(ms[:, 1].mean()), "wan_s": float(np.mean(wan_s)) if wan_s else None,
                          "openfly_s": float(np.median(of_s)) if of_s else None}),
              flush=True)
        for arm in args.arms:
            rs = [x for x in rows if x["arm"] == arm and x["id"] in {s["id"] for s in here}]
            print(f"  {arm:22s} ok {sum(x['flown_ok'] for x in rs)}/{len(rs)} crash {sum(x['collision'] for x in rs)} "
                  f"success {sum(x['success'] for x in rs)} (sane yaw {sum(x['success_sane'] for x in rs)})", flush=True)
        del env, pol
        torch.cuda.empty_cache()
    summary = {}
    for arm in args.arms:
        rs = [x for x in rows if x["arm"] == arm]
        summary[arm] = {"n": len(rs), "flown_ok": sum(x["flown_ok"] for x in rs),
                        "collisions": sum(x["collision"] for x in rs), "success": sum(x["success"] for x in rs),
                        "success_sane": sum(x["success_sane"] for x in rs),
                        "track_err_m": float(np.mean([x["track_err_m"] for x in rs])),
                        "path_m": float(np.mean([x["path_m"] for x in rs])),
                        "crash_t_mean": float(np.mean([x["crash_t"] for x in rs if x["crash_t"] is not None]))
                        if any(x["crash_t"] is not None for x in rs) else None}
    json.dump({"summary": summary, "runs": rows, "ckpt": args.ckpt}, open(os.path.join(out_dir, "deploy.json"), "w"),
              indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
