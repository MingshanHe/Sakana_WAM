#!/bin/bash
# Collect all 80 training houses, 2 rounds, on one GPU (renderer) + CPU dynamics.
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH OMP_NUM_THREADS=4
cd "$HOME/sakana_dreamer/.claude/worktrees/follow/dreamer"
for r in 0 1; do python collect.py --round $r --steps 200; done
