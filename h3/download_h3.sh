#!/usr/bin/env bash
# Download only what the t2va / fl2va workflows load (~143 GB) into the HF cache, skipping transformer_ref/
# (ref2va) and the original FL2VA/ and Ref2VA/ checkpoints (the full repo is ~498 GB).
# Accept the MiniMax H3 Community License on https://huggingface.co/MiniMaxAI/MiniMax-H3 first.
set -euo pipefail
args=()
for pattern in model_index.json modular_model_index.json \
  "transformer/*" "text_encoder/*" "tokenizer/*" "processor/*" \
  "vae/*" "audio_vae/*" "scheduler/*" "audio_scheduler/*"; do
  args+=(--include "$pattern")
done
hf download MiniMaxAI/MiniMax-H3 "${args[@]}" --max-workers 16
