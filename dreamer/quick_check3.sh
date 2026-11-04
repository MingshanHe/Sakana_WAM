#!/bin/bash
# Quick check, take 2: fixed Gumbel sampler (the stochastic latent was constant). Same small data / short training.
set -e
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH OMP_NUM_THREADS=4
cd "$HOME/sakana_dreamer/.claude/worktrees/follow/dreamer"
echo "== 1. world model"; date
python train.py --tag quick3_wm --max-train-files 16 --steps 3000 --eval-every 1000 --save-every 1000 --resume
echo "== 1b. plan sensitivity"; sed -i "s#data/runs/[a-z0-9_]*/last.pt#data/runs/quick3_wm/last.pt#" outputs/dbg_policy.py
python outputs/dbg_policy.py 2>&1 | grep -v Warn | tail -4
echo "== 2. actor-critic"; date
python train.py --tag quick3_ac --init data/runs/quick3_wm/last.pt --ac --max-train-files 16 --steps 2000 \
  --eval-every 1000 --save-every 1000 --warmup 1 --resume
echo "== 3. combined deployment"; date
python deploy_combined.py --ckpt data/runs/quick3_ac/last.pt --repeats 1 --video

echo "== 4. live Wan replanning every 3 s"; date
( source ~/miniconda3/etc/profile.d/conda.sh; conda activate sakana_dreamer
  python wan_server.py --root data/wan_requests_quick --idle-exit 120 > data/quick_wan_server.log 2>&1 ) &
python deploy_combined.py --ckpt data/runs/quick3_ac/last.pt --repeats 1 --replan-every 3 --seconds 9 \
  --max-houses 3 --arms diffwam+track diffwam+dreamer diffwam+dreamer+mpc --wan-root data/wan_requests_quick
echo "== done (replan)"; date
