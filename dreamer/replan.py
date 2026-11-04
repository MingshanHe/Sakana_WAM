"""Client side of closed-loop replanning (env `sakana_hyperdrone`): ask wan_server.py for DiffWAM features of the
current onboard frame, then turn them into a 5 s plan with the DiffWAM readout head (same pooling and depth scale as
diffwam/train_head.py)."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "diffwam"))
from train_head import POOL, ReadoutHead, to_poses  # noqa: E402


class Replanner:
    def __init__(self, head_ckpt, root=os.path.join(HERE, "data", "wan_requests")):
        ck = torch.load(head_ckpt, weights_only=False)
        self.evals = list(ck["evals"])
        self.model = ReadoutHead(k=len(self.evals)).cuda().eval()
        self.model.load_state_dict(ck["model"])
        self.root = root
        os.makedirs(root, exist_ok=True)

    def server_ready(self):
        return os.path.exists(os.path.join(self.root, "READY"))

    def request(self, req_id, rgb, depth, instruction, seed):
        d = os.path.join(self.root, req_id)
        os.makedirs(d, exist_ok=True)
        from PIL import Image

        Image.fromarray(rgb).save(os.path.join(d, "I0.png"))
        np.save(os.path.join(d, "I0_depth.npy"), depth.astype(np.float32))
        tmp = os.path.join(d, "request.tmp")
        json.dump({"instruction": instruction, "seed": int(seed)}, open(tmp, "w"))
        os.replace(tmp, os.path.join(d, "request.json"))
        return d

    def wait(self, d, timeout=600):
        path = os.path.join(d, "feats.pt")
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > timeout:
                raise TimeoutError(d)
            time.sleep(0.05)
        for _ in range(100):  # NFS: the file may appear before it is complete
            try:
                return torch.load(path)
            except Exception:
                time.sleep(0.1)
        return torch.load(path)

    @torch.no_grad()
    def plan(self, feats, depth):
        """feats.pt dict + I0 depth (480, 832) -> relative plan (39, 4) in the frame of the request pose."""
        pool = lambda x: F.adaptive_avg_pool2d(x.float().flatten(0, 2).permute(0, 3, 1, 2), POOL)  # noqa: E731
        z = pool(feats["Z"]).reshape(*feats["Z"].shape[:3], 128, *POOL).permute(0, 1, 2, 4, 5, 3)[self.evals]
        a0 = pool(feats["A0"]).reshape(*feats["A0"].shape[:3], 128, *POOL).permute(0, 1, 2, 4, 5, 3)[self.evals]
        valid = depth[(depth > 0.3) & (depth < 30)]
        s0 = float(np.clip(np.median(valid), 0.5, 15)) if len(valid) else 1.0
        h, w = depth.shape
        regions = [depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3], depth[:, : w // 3], depth[:, 2 * w // 3:],
                   depth[: h // 3], depth[2 * h // 3:]]
        geo = np.array([np.median(r[r > 0]) if (r > 0).any() else 0.0 for r in regions], np.float32) / s0
        zt = z.half().float()[None].cuda()
        a0t = a0.half().float()[None].cuda()
        out = self.model(zt, a0t, torch.tensor(geo[None]).cuda())
        return to_poses(out, torch.tensor([s0]).cuda())[0].cpu().numpy()
