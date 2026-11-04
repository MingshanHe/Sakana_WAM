"""Figures for the Dreamer local-planner report (env: conda `sakana_hyperdrone`; the open-loop figure needs a GPU).

    python make_figures.py --run quick3 --out ../results/wan2.2-ti2v-5b/dreamer_local_planner
Writes overview.png, openloop.png, flights.png, results.png, training.png and copies a few flight videos.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SAMPLES_EVAL = os.path.join(HERE, "..", "diffwam", "data", "samples_eval")
ARM_LABEL = {"teacher+track": "teacher plan + tracker", "diffwam+track": "DiffWAM plan + tracker",
             "prior+dreamer": "instruction prior + Dreamer", "diffwam+dreamer": "DiffWAM plan + Dreamer",
             "diffwam+dreamer+mpc": "DiffWAM + Dreamer + shield"}
COLORS = {"teacher+track": "#9aa5b1", "diffwam+track": "#5b7083", "prior+dreamer": "#f2b880",
          "diffwam+dreamer": "#d1495b", "diffwam+dreamer+mpc": "#8e2c48"}


def box(ax, xy, w, h, text, fc, fontsize=9):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="#333", lw=1))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def arrow(ax, a, b, text="", color="#333"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.4))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.03, text, ha="center", fontsize=7.5, color=color)


def overview(out, deploy, sid="s05_ascend_seed0"):
    """Pipeline schematic with real images + the headline numbers."""
    fig = plt.figure(figsize=(14, 6.8))
    ax = fig.add_axes([0, 0.33, 1, 0.67])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    i0 = iio.imread(os.path.join(SAMPLES_EVAL, sid, "I0.png"))
    slots = np.load(os.path.join(SAMPLES_EVAL, sid, "slots.npz"))["frames"]
    ax.imshow(i0, extent=(0.01, 0.13, 0.52, 0.86))
    ax.text(0.07, 0.89, "onboard frame I0\n+ instruction", ha="center", fontsize=9)
    xs = [0.18, 0.345, 0.51, 0.675, 0.84]
    w, y0, h = 0.125, 0.58, 0.22
    labels = ["Wan2.2-TI2V-5B\n4 denoising steps\n(blocks 9/15/21)", "DiffWAM\nreadout head\n1.6M params",
              "global plan\n38 poses / 5 s\n(where to go)", "DreamerV3\nlocal planner\n207M params, 10 Hz\n(how to fly)",
              "L2F Crazyflie\nRAPTOR position\ncontroller"]
    fcs = ["#e8eef7", "#e8eef7", "#fdf3e1", "#fbe3e6", "#e9f5ec"]
    for x0, lab, fc in zip(xs, labels, fcs):
        box(ax, (x0, y0), w, h, lab, fc, fontsize=8.5)
    arrow(ax, (0.135, 0.69), (xs[0], 0.69))
    for a, b, t in zip(xs[:-1], xs[1:], ["features", "", "waypoints", "v, yaw rate"]):
        arrow(ax, (a + w, 0.69), (b, 0.69))
        if t:
            ax.text((a + w + b) / 2, 0.83, t, ha="center", fontsize=7.5)
    cx_d, cx_e = xs[3] + w / 2, xs[4] + w / 2
    arrow(ax, (cx_e, y0), (cx_e, 0.45), color="#2a7f62")
    arrow(ax, (cx_e, 0.45), (cx_d + 0.01, 0.45), color="#2a7f62")
    arrow(ax, (cx_d + 0.01, 0.45), (cx_d + 0.01, y0), color="#2a7f62")
    ax.text((cx_d + cx_e) / 2 + 0.005, 0.47, "64x64 RGB-D + state", ha="center", fontsize=7.5, color="#2a7f62")
    arrow(ax, (cx_d - 0.01, 0.40), (0.07, 0.40), color="#7a4ea3")
    arrow(ax, (cx_e, 0.40), (cx_d - 0.01, 0.40), color="#7a4ea3")
    arrow(ax, (cx_e, 0.45), (cx_e, 0.40), color="#7a4ea3")
    arrow(ax, (0.07, 0.40), (0.07, 0.52), color="#7a4ea3")
    ax.text(0.40, 0.355, "replan every 3 s: new I0 -> Wan server (~3-4 s latency); Dreamer keeps flying the old plan meanwhile",
            ha="center", fontsize=8.5, color="#7a4ea3")
    for k, j in enumerate((0, 19, 38)):
        ax.imshow(slots[j], extent=(0.17 + 0.105 * k, 0.27 + 0.105 * k, 0.03, 0.27))
    ax.text(0.33, 0.29, "Wan imagined rollout, 0 / 2.5 / 5 s (ascend: it flies into the ceiling)", ha="center", fontsize=8.5)
    flight = os.path.join(HERE, "data", "deploy", "quick3_ac", f"{sid}_diffwam+dreamer.mp4")
    if os.path.exists(flight):
        f = iio.mimread(flight)
        for k, j in enumerate((0, len(f) // 2, len(f) - 1)):
            ax.imshow(f[j], extent=(0.62 + 0.12 * k, 0.72 + 0.12 * k, 0.03, 0.27))
        ax.text(0.80, 0.29, "Dreamer flight, 0 / 3 / 6 s: climbs 0.3 m, no crash (tracker: ceiling crash)", ha="center", fontsize=8.5)
    ax.set_aspect("auto")
    # headline bars
    ax2 = fig.add_axes([0.07, 0.09, 0.4, 0.19])
    arms = list(deploy["summary"])
    s = deploy["summary"]
    x = np.arange(len(arms))
    ax2.bar(x - 0.2, [s[a]["success"] for a in arms], 0.4, color=[COLORS[a] for a in arms], label="success")
    ax2.bar(x + 0.2, [s[a]["collisions"] for a in arms], 0.4, color="#333", alpha=0.6, label="crashes")
    ax2.set_xticks(x, [ARM_LABEL[a].replace(" + ", "\n+ ") for a in arms], fontsize=7)
    ax2.set_ylabel("of 20 test flights", fontsize=8)
    ax2.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.32), ncol=2, frameon=False)
    ax2.set_ylim(0, 14)
    ax2.set_title("one plan from I0 (quick check: 10% data, short training)", fontsize=9)
    ax3 = fig.add_axes([0.55, 0.09, 0.4, 0.19])
    ax3.bar(x, [s[a]["track_err_m"] for a in arms], color=[COLORS[a] for a in arms])
    ax3.set_xticks(x, [ARM_LABEL[a].replace(" + ", "\n+ ") for a in arms], fontsize=7)
    ax3.set_ylabel("mean distance to plan (m)", fontsize=8)
    ax3.set_title("plan tracking error", fontsize=9)
    fig.savefig(os.path.join(out, "overview.png"), dpi=130)
    plt.close(fig)


def openloop(out, ckpt, val_dir, ctx=16, horizon=32, rows=3):
    import torch

    from policy import DreamerPolicy

    pol = DreamerPolicy(ckpt, num=rows, compile=False)
    wm = pol.wm
    img = np.load(os.path.join(val_dir, "image.npy"), mmap_mode="r")
    vec = np.load(os.path.join(val_dir, "vector.npy"), mmap_mode="r")
    act = np.load(os.path.join(val_dir, "action.npy"), mmap_mode="r")
    first = np.load(os.path.join(val_dir, "is_first.npy"), mmap_mode="r")
    picks = []
    for i in range(img.shape[0]):  # chunks with no episode boundary inside
        for k in range(0, img.shape[1] - ctx - horizon, 7):
            if not first[i, k + 1:k + ctx + horizon].any():
                picks.append((i, k))
                break
        if len(picks) == rows:
            break
    show = [0, 4, 8, 16, 24, 31]
    fig, axes = plt.subplots(2 * rows, len(show) + 1, figsize=(2 * (len(show) + 1), 4.1 * rows))
    with torch.no_grad():
        for r, (i, k) in enumerate(picks):
            sl = slice(k, k + ctx + horizon)
            im = torch.as_tensor(np.array(img[i, sl]))[None].cuda()
            ve = torch.as_tensor(np.array(vec[i, sl]))[None].cuda().float()
            ac = torch.as_tensor(np.array(act[i, sl]))[None].cuda().float()
            isf = torch.zeros(1, ctx, dtype=torch.bool, device="cuda")
            isf[:, 0] = True
            tok = wm.enc(im[0, :ctx], ve[0, :ctx])[None]
            _, (state, prev) = wm.rssm.observe(tok, ac[:, :ctx], isf)
            preds = []
            for h in range(horizon):
                a = ac[:, ctx + h - 1] if h else prev
                state = wm.rssm.img_step(state, a)
                p, _ = wm.dec(wm.feat(state))
                preds.append(((p[0].permute(1, 2, 0).float().cpu().numpy() + 0.5).clip(0, 1) * 255).astype(np.uint8))
            ax = axes[2 * r, 0]
            ax.imshow(np.array(img[i, k + ctx - 1, ..., :3]))
            ax.set_title("last context", fontsize=8)
            axes[2 * r + 1, 0].axis("off")
            for c, h in enumerate(show):
                axes[2 * r, c + 1].imshow(np.array(img[i, k + ctx + h, ..., :3]))
                axes[2 * r + 1, c + 1].imshow(preds[h][..., :3])
                axes[2 * r, c + 1].set_title(f"real +{(h + 1) / 10:.1f} s", fontsize=8)
                axes[2 * r + 1, c + 1].set_title(f"imagined +{(h + 1) / 10:.1f} s", fontsize=8)
    for a in axes.flat:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("World model on held-out houses: 1.6 s of context, then 3.2 s imagined under the logged actions",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "openloop.png"), dpi=110)
    plt.close(fig)


def flights(out, deploy_dir, cases):
    d = json.load(open(os.path.join(deploy_dir, "deploy.json")))
    r = {(x["id"], x["arm"]): x for x in d["runs"]}
    arms = ["diffwam+track", "diffwam+dreamer"]
    fig, axes = plt.subplots(len(cases) * 2, 1, figsize=(12, 1.75 * 2 * len(cases)))
    for c, sid in enumerate(cases):
        for j, arm in enumerate(arms):
            f = iio.mimread(os.path.join(deploy_dir, f"{sid}_{arm}.mp4"))
            idx = np.linspace(0, len(f) - 1, 8).astype(int)
            strip = np.concatenate([f[i][::2, ::2] for i in idx], 1)
            ax = axes[2 * c + j]
            ax.imshow(strip)
            ax.set_xticks([])
            ax.set_yticks([])
            x = r[(sid, arm)]
            tag = "CRASH" if x["collision"] else "ok"
            ax.set_ylabel(ARM_LABEL[arm].replace(" + ", "\n+ "), fontsize=7, rotation=0, ha="right", va="center")
            ax.set_title(f"{sid}: {tag}, forward {x['forward_m']:.2f} m, up {x['up_m']:.2f} m, yaw {x['yaw_deg']:.0f} deg"
                         f"{' (crash at %.1f s)' % x['crash_t'] if x['crash_t'] else ''}", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "flights.png"), dpi=110)
    plt.close(fig)


def results(out, single, replan):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, d, title in ((axes[0], single, "one plan from I0 (20 test flights)"),
                         (axes[1], replan, "replan every 3 s with live Wan (3 houses, 6 flights)")):
        s = d["summary"]
        arms = list(s)
        x = np.arange(len(arms))
        for k, (key, lab) in enumerate((("flown_ok", "correct motion"), ("success", "success"), ("collisions", "crashes"))):
            ax.bar(x + (k - 1) * 0.27, [s[a][key] for a in arms], 0.27, label=lab,
                   color=["#4c78a8", "#59a14f", "#e15759"][k])
        ax.set_xticks(x, [ARM_LABEL[a].replace(" + ", "\n+ ") for a in arms], fontsize=7.5)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
        ax.set_ylim(0, max(s[a]["n"] for a in arms) * 0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "results.png"), dpi=120)
    plt.close(fig)


def training(out, run):
    def rows(tag):
        p = os.path.join(HERE, "data", "runs", tag, "log.jsonl")
        tr, ev = [], []
        for line in open(p):
            j = json.loads(line)
            (ev.append(j["eval"]) if "eval" in j else tr.append(j))
        return tr, ev

    wtr, wev = rows(f"{run}_wm")
    atr, aev = rows(f"{run}_ac")
    off = wtr[-1]["step"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot([r["step"] for r in wtr], [r["img_mse_px"] for r in wtr], label="train")
    axes[0].plot([off + r["step"] for r in atr], [r["img_mse_px"] for r in atr], color="C0")
    axes[0].plot([r["step"] for r in wev] + [off + r["step"] for r in aev],
                 [r["val_img_mse_px"] for r in wev + aev], "o-", label="val (held-out houses)")
    axes[0].plot([r["step"] for r in wev] + [off + r["step"] for r in aev],
                 [r["ol_img_mse_px"] for r in wev + aev], "s--", label="val open-loop 3.2 s")
    axes[0].axvline(off, color="#999", ls=":")
    axes[0].set_title("image MSE / pixel", fontsize=9)
    axes[0].legend(fontsize=7)
    axes[1].plot([r["step"] for r in wtr], [r["rew"] for r in wtr], label="reward loss")
    axes[1].plot([r["step"] for r in wtr], [r["vec"] for r in wtr], label="state+plan loss")
    axes[1].plot([off + r["step"] for r in atr], [r["rew"] for r in atr], color="C0")
    axes[1].plot([off + r["step"] for r in atr], [r["vec"] for r in atr], color="C1")
    axes[1].axvline(off, color="#999", ls=":")
    axes[1].set_title("world-model heads (train)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[2].plot([off + r["step"] for r in atr], [r["imag_rew"] for r in atr], label="imagined reward / step")
    axes[2].plot([off + r["step"] for r in atr], [r["imag_con"] for r in atr], label="imagined continue")
    axes[2].set_title("actor-critic in imagination", fontsize=9)
    axes[2].legend(fontsize=7)
    for a in axes:
        a.set_xlabel("update step", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "training.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="quick3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    single_dir = os.path.join(HERE, "data", "deploy", f"{args.run}_ac")
    single = json.load(open(os.path.join(single_dir, "deploy.json")))
    replan = json.load(open(os.path.join(HERE, "data", "deploy", f"{args.run}_ac_replan3", "deploy.json")))
    overview(args.out, single)
    flights(args.out, single_dir, ["s05_ascend_seed0", "s03_doorway_seed1", "s07_turn_right_seed0"])
    results(args.out, single, replan)
    training(args.out, args.run)
    if not args.no_gpu:
        openloop(args.out, os.path.join(HERE, "data", "runs", f"{args.run}_ac", "last.pt"),
                 os.path.join(HERE, "data", "replay", "val", "h052_r1"))
    vids = os.path.join(args.out, "flights")
    os.makedirs(vids, exist_ok=True)
    for sid in ("s05_ascend_seed0", "s03_doorway_seed1", "s07_turn_right_seed0"):
        for arm in ("diffwam+track", "diffwam+dreamer"):
            shutil.copy(os.path.join(single_dir, f"{sid}_{arm}.mp4"), vids)
    for name in ("deploy.json",):
        shutil.copy(os.path.join(single_dir, name), os.path.join(args.out, "deploy_single.json"))
    shutil.copy(os.path.join(HERE, "data", "deploy", f"{args.run}_ac_replan3", "deploy.json"),
                os.path.join(args.out, "deploy_replan3.json"))
    print("figures in", args.out)


if __name__ == "__main__":
    main()
