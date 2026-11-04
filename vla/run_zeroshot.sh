#!/bin/bash
# Zero-shot OpenFly-Agent vs DiffWAM on the 20 test cases (one GPU).
cd "$HOME/sakana_dreamer/.claude/worktrees/follow/vla"
rm -rf ../dreamer/data/vla_requests
PYTHONNOUSERSITE=1 $HOME/miniconda3/envs/sakana_vla/bin/python openfly_server.py --root ../dreamer/data/vla_requests \
  --idle-exit 300 > ../dreamer/data/openfly_server.log 2>&1 &
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH OMP_NUM_THREADS=4
python eval_zeroshot.py "$@"
