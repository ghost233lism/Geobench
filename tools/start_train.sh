#!/usr/bin/env bash
set -euo pipefail

pip install -U ms-swift
pip install -U "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention
# 若出现训练缓慢的问题请参考：https://github.com/fla-org/flash-linear-attention/issues/758
# 请使用python3.12: https://github.com/fla-org/flash-linear-attention/issues/121
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation

# causal_conv1d
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# flash-attention
pip install "flash-attn==2.8.3" --no-build-isolation

# deepspeed训练
pip install deepspeed

# vllm (torch2.10) for inference/deployment/RL
pip install -U "vllm==0.17.1"
pip install swanlab

REPO_ROOT=/actual/path/to/Geobench \
MODEL=/actual/path/to/Qwen3.5-9B \
INPUT_JSONL=/actual/path/to/train_data.jsonl \
CHECKPOINT_OUTPUT_ROOT=/actual/path/to/checkpoints \
SKIP_DOMAINS=street \
bash tools/launch_full_domain_train_qwen35_9b_colocate_b12.sh
