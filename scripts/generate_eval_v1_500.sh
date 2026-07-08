#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

venv/bin/python -m scripts.generate_eval_dataset \
  --sample-size 500 \
  --output data/eval/eval_v1_500.jsonl \
  --generation-model anthropic/claude-sonnet-5=0.5 \
  --generation-model openai/gpt-5.4-mini=0.5 \
  --overwrite
