#!/usr/bin/env bash
# Run feature extraction on a GPU node. Submit with (sbatch spools the script, so use --wrap with an absolute path):
#   sbatch -p a3 --gres=gpu:2 -c 16 --mem=400G -t 2:00:00 -o ~/sakana_dreamer/h3/logs/%j.out \
#     --wrap "bash ~/sakana_dreamer/h3/run_extract.sh --placement multi --image ~/sakana_dreamer/h3/assets/obs.png --prompt '...'"
# One GPU is enough with the default --placement offload (slower: weights stream from host RAM).
set -euo pipefail
cd /home/mingshanhe_sakana_ai/sakana_dreamer/h3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LP_NUM_THREADS=${LP_NUM_THREADS:-8}
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/mingshanhe_sakana_ai/miniconda3/envs/sakana_dreamer/bin/python h3_features.py --out "outputs/features_${SLURM_JOB_ID:-local}.pt" "$@"
