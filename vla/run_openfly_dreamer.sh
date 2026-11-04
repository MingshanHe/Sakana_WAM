#!/bin/bash
# OpenFly-Agent v1 (global, every 1 s) + Dreamer (local, 10 Hz) in L2F dynamics, vs DiffWAM arms; 20 test cases x 2 repeats.
cd "$HOME/sakana_dreamer/.claude/worktrees/follow/dreamer"
rm -rf data/vla_requests
PYTHONNOUSERSITE=1 $HOME/miniconda3/envs/sakana_vla/bin/python ../vla/openfly_server.py --root data/vla_requests \
  --idle-exit 300 > data/openfly_server2.log 2>&1 &
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH OMP_NUM_THREADS=4
python deploy_combined.py --ckpt data/runs/quick3_ac/last.pt --repeats 2 --seconds 6 --video \
  --arms diffwam+track diffwam+dreamer openfly+track openfly+dreamer openfly+dreamer+mpc "$@"
