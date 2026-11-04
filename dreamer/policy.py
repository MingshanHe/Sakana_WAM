"""DreamerV3 local planner at deployment: filters onboard observations with the RSSM and outputs velocity commands.

    pol = DreamerPolicy("data/runs/ac200m/last.pt", num=30)
    a = pol.act(obs, is_first)                # actor mean
    a = pol.act(obs, is_first, mpc=True)      # + world-model shield: pick the best of K candidate first actions by
                                              #   imagined return (reward + crash) over `mpc_horizon` steps
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dv3.agent import ActorCritic, WorldModel  # noqa: E402
from dv3.nets import TwoHot  # noqa: E402


class DreamerPolicy:
    def __init__(self, ckpt, num, device="cuda", candidates=64, mpc_horizon=10, compile=True):
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        c = ck["config"]
        self.wm = WorldModel(c["img_ch"], c["vec_dim"], c["act_dim"], c["size"]).to(device).eval()
        self.wm.load_state_dict(ck["wm"])
        self.ac = ActorCritic(self.wm.feat_dim, c["act_dim"], self.wm.units).to(device).eval()
        self.ac.load_state_dict(ck["ac"])
        if compile:
            self.wm.rssm.compile()
        self.twohot = TwoHot(device=device)
        self.num, self.device, self.k, self.h = num, device, candidates, mpc_horizon
        self.act_dim = c["act_dim"]
        self.state = self.wm.rssm.initial(num, device)
        self.prev_a = torch.zeros(num, self.act_dim, device=device)

    @torch.no_grad()
    def act(self, obs, is_first, mpc=False):
        img = torch.as_tensor(obs["image"], device=self.device)
        vec = torch.as_tensor(obs["vector"], device=self.device).float()
        first = torch.as_tensor(is_first, device=self.device).bool()
        with torch.autocast("cuda", torch.bfloat16):
            tok = self.wm.enc(img, vec)
            deter, stoch, _, _ = self.wm.rssm._obs_step(self.state["deter"], self.state["stoch"], self.prev_a, tok, first)
            self.state = {"deter": deter.float(), "stoch": stoch.float()}
            feat = self.wm.feat(self.state)
            a = self.ac.actor(feat).mean.clamp(-1, 1)
            if mpc:
                a = self._shield(a)
        self.prev_a = a.float()
        return a.float().cpu().numpy()

    def _shield(self, a_mean):
        """K candidate first actions per drone (actor mean + actor samples + brake / turn primitives); each is followed
        by the actor mean in imagination; score = discounted imagined reward with crash (continue) weighting + value."""
        n, k, h = self.num, self.k, self.h
        feat = self.wm.feat(self.state)
        dist = self.ac.actor(feat)
        cand = dist.sample((k,)).clamp(-1, 1)  # (K, N, A)
        cand[0] = a_mean
        cand[1] = 0.0  # hover / brake
        cand[2] = torch.tensor([0, 0, 0, 0.8], device=self.device)  # turn left in place
        cand[3] = torch.tensor([0, 0, 0, -0.8], device=self.device)
        state = {key: v[None].expand(k, *v.shape).reshape(k * n, *v.shape[1:]) for key, v in self.state.items()}
        act = cand.reshape(k * n, -1)
        ret = torch.zeros(k * n, device=self.device)
        alive = torch.ones(k * n, device=self.device)
        disc = 1.0
        for step in range(h):
            state = self.wm.rssm.img_step(state, act.to(state["deter"].dtype))
            f = self.wm.feat(state)
            r = self.twohot.mean(self.wm.rew(f))
            cont = torch.sigmoid(self.wm.con(f)[..., 0].float())
            ret = ret + disc * alive * r
            alive = alive * cont
            disc *= self.ac.gamma
            act = self.ac.actor(f).mean.clamp(-1, 1)
        ret = ret + disc * alive * self.twohot.mean(self.ac.critic(self.wm.feat(state)))
        best = ret.reshape(k, n).argmax(0)
        return cand[best, torch.arange(n, device=self.device)]
