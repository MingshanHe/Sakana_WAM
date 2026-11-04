#!/bin/bash
# World-model phase (size200m), resumable: bash run_wm.sh <tag> <size> <steps>
export PATH=$HOME/miniconda3/envs/sakana_hyperdrone/bin:$PATH OMP_NUM_THREADS=4
cd "$HOME/sakana_dreamer/.claude/worktrees/follow/dreamer"
python train.py --tag ${1:-wm200m} --size ${2:-size200m} --steps ${3:-60000} --resume
