"""Wan2.2 DiffWAM feature server for closed-loop replanning (env: conda `sakana_dreamer`).

The flight process (env `sakana_hyperdrone`) cannot import Wan2.2 (diffusers main lives in the other env), so the two
talk through a directory, like the rest of the repo:
  <root>/<req_id>/request.json   {"instruction", "seed"}       written by replan.request()
  <root>/<req_id>/I0.png         current onboard frame, 832 x 480
  <root>/<req_id>/feats.pt       DiffWAM features (same format as diffwam/wan_worker.py, no video), written here
The server keeps the pipeline and the PCA in memory and serves requests in arrival order.

    python wan_server.py --root ../dreamer/data/wan_requests
"""

import argparse
import glob
import json
import os
import sys
import time

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "wan"), os.path.join(HERE, "..", "follow"), os.path.join(HERE, "..", "diffwam")]
from sweep_imagine import NEGATIVE  # noqa: E402
from wan_features import WAN_LAYERS, WanFeatureExtractor, latent_slot_times, load_pipeline, resample_slots  # noqa: E402
from wan_worker import project  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, "data", "wan_requests"))
    ap.add_argument("--steps", type=int, default=30, help="scheduler steps (only 0-3 are run)")
    ap.add_argument("--idle-exit", type=float, default=3600)
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    pca = torch.load(os.path.join(HERE, "..", "diffwam", "data", "feature_pca.pt"))
    pipe = load_pipeline()
    open(os.path.join(args.root, "READY"), "w").write(str(os.getpid()))
    print("wan server ready", flush=True)
    last = time.time()
    while time.time() - last < args.idle_exit:
        todo = sorted((os.path.getmtime(p), os.path.dirname(p)) for p in glob.glob(os.path.join(args.root, "*", "request.json"))
                      if not os.path.exists(os.path.join(os.path.dirname(p), "feats.pt")))
        if not todo:
            time.sleep(0.05)
            continue
        d = todo[0][1]
        req = json.load(open(os.path.join(d, "request.json")))
        image = Image.open(os.path.join(d, "I0.png")).convert("RGB")
        kw = dict(image=image, prompt=req["instruction"], negative_prompt=NEGATIVE, height=480, width=832,
                  num_frames=121, num_inference_steps=args.steps, guidance_scale=5.0)
        t0 = time.perf_counter()
        with WanFeatureExtractor(pipe.transformer, WAN_LAYERS, (0, 1, 2, 3), passes_per_step=2) as ext:
            ext.run(lambda: pipe(**kw, generator=torch.Generator("cuda").manual_seed(int(req["seed"])), output_type="latent"))
        res = ext.result()
        z = res["Z"]
        a0 = project(z[:, :, :1], pca)
        z39 = resample_slots(project(z, pca).float(), latent_slot_times(z.shape[2]), torch.linspace(0, 5, 39), dim=2).half()
        seconds = time.perf_counter() - t0
        torch.save({"Z": z39, "A0": a0, "seconds": seconds}, os.path.join(d, "feats.tmp"))
        os.replace(os.path.join(d, "feats.tmp"), os.path.join(d, "feats.pt"))
        print(json.dumps({"req": os.path.basename(d), "seconds": round(seconds, 2)}), flush=True)
        last = time.time()


if __name__ == "__main__":
    main()
