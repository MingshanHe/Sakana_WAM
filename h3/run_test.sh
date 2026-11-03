#!/usr/bin/env bash
# End-to-end H3 test on the repo's test input (assets/test.jpg + assets/test_instruction):
#   1. CPU smoke test of the feature hooks
#   2. DiffWAM features (blocks 15/25/35, evals 0-3, no decode)  -> outputs/test_features.pt + _pca.png
#   3. the whole video for the same request and seed (decoded)   -> outputs/test_video.mp4 + _frames.png
#
# Needs 1 H100 and ~150 GB host RAM (CPU offload). From a login node:
#   srun -p a3 --gres=gpu:1 -c 16 --mem=300G -t 1:30:00 bash ~/sakana_dreamer/h3/run_test.sh
# or inside an existing srun shell:  bash h3/run_test.sh
# Extra arguments are passed to both H3 runs, e.g. `--num-inference-steps 30` or `--seed 1`.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
PY=${PY:-$HOME/miniconda3/envs/sakana_dreamer/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
mkdir -p outputs
REQUEST=(--image assets/test.jpg --prompt-file assets/test_instruction --height 480 --width 832 --num-frames 124 --seed 0)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 1/3 smoke test"
$PY smoke_test_tiny.py | tail -1

echo "== 2/3 DiffWAM features"
$PY h3_features.py "${REQUEST[@]}" "$@" --out outputs/test_features.pt
$PY inspect_features.py outputs/test_features.pt --scale 4 | tail -4

echo "== 3/3 whole video (same request + seed)"
$PY generate_video.py "${REQUEST[@]}" "$@" --out outputs/test_video.mp4
