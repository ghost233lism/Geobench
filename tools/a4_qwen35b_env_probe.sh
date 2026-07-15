#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/mnt/nas/zhangyiming/geobench/output/a4repair_env/qwen35b_env_$(date +%Y%m%d-%H%M%S).log}"
mkdir -p "$(dirname "${OUT}")"

exec > >(tee -a "${OUT}") 2>&1

section() {
  printf '\n===== %s =====\n' "$*"
}

section "basic"
date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
hostname
id
pwd
uname -a

section "gpu"
nvidia-smi || true

section "conda"
/root/miniconda3/bin/conda env list || true
/root/miniconda3/bin/conda list -n qwen35b || true

section "pip-freeze"
/root/miniconda3/envs/qwen35b/bin/python -m pip freeze || true

section "relevant-env-vars"
env | sort | grep -E '^(CONDA|CUDA|NVIDIA|LD_LIBRARY_PATH|LIBRARY_PATH|PATH=|PYTHONPATH|VLLM|NCCL|TORCH|PYTORCH|TRITON|TRANSFORMERS|HF_|HUGGINGFACE|GEOBENCH|SWIFT|WANDB|SWANLAB|TOKENIZERS|FLASH|CUDA_HOME)=' || true

section "python-runtime"
PYTHONPATH="/mnt/nas/zhangyiming/geobench${PYTHONPATH:+:${PYTHONPATH}}" \
GEOBENCH_PATCH_QWEN35_ZERO3_CONV1D=0 \
/root/miniconda3/envs/qwen35b/bin/python - <<'PY'
import importlib
import importlib.metadata as md
import json
import os
import platform
import sys

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
print("platform:", platform.platform())
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("sys_path:")
for item in sys.path:
    print("  ", item)

package_names = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "vllm",
    "deepspeed",
    "trl",
    "accelerate",
    "peft",
    "datasets",
    "tokenizers",
    "huggingface-hub",
    "triton",
    "xformers",
    "flash-attn",
    "causal-conv1d",
    "flash-linear-attention",
    "fla-core",
    "tilelang",
    "qwen-vl-utils",
    "wandb",
    "swanlab",
    "modelscope",
]
print("\nmetadata_versions:")
for name in package_names:
    try:
        print(f"  {name}=={md.version(name)}")
    except Exception as exc:
        print(f"  {name}: <missing> ({type(exc).__name__})")

module_names = [
    "torch",
    "transformers",
    "vllm",
    "deepspeed",
    "trl",
    "accelerate",
    "peft",
    "datasets",
    "tokenizers",
    "huggingface_hub",
    "triton",
    "xformers",
    "flash_attn",
    "causal_conv1d",
    "fla",
    "tilelang",
    "qwen_vl_utils",
    "swift",
    "wandb",
    "swanlab",
]
print("\nmodule_imports:")
for name in module_names:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        path = getattr(module, "__file__", None)
        print(f"  {name}: ok version={version} file={path}")
    except Exception as exc:
        print(f"  {name}: error {type(exc).__name__}: {exc}")

print("\ntorch_cuda:")
try:
    import torch

    print("  torch_version:", torch.__version__)
    print("  torch_cuda:", torch.version.cuda)
    print("  cuda_available:", torch.cuda.is_available())
    print("  device_count:", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"  device[{idx}]: name={props.name}, capability={props.major}.{props.minor}, "
            f"total_memory={props.total_memory}"
        )
except Exception as exc:
    print(f"  torch cuda error {type(exc).__name__}: {exc}")

print("\nqwen35_config:")
try:
    from transformers import AutoConfig

    model_dir = "/mnt/nas/zhangyiming/geobench/models/Qwen3.5-9B"
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print("  config_class:", config.__class__)
    print("  model_type:", getattr(config, "model_type", None))
    text_config = getattr(config, "text_config", None)
    print("  text_model_type:", getattr(text_config, "model_type", None))
    print("  architectures:", getattr(config, "architectures", None))
except Exception as exc:
    print(f"  config error {type(exc).__name__}: {exc}")

print("\nqwen35_modeling:")
try:
    from transformers.models.qwen3_5 import modeling_qwen3_5

    print("  modeling_file:", modeling_qwen3_5.__file__)
    print("  causal_conv1d_fn:", modeling_qwen3_5.causal_conv1d_fn)
    print("  causal_conv1d_update:", modeling_qwen3_5.causal_conv1d_update)
    print("  is_fast_path_available:", getattr(modeling_qwen3_5, "is_fast_path_available", None))
except Exception as exc:
    print(f"  qwen35 modeling error {type(exc).__name__}: {exc}")

print("\nselected_env:")
keys = [
    "CONDA_PREFIX",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "PYTHONPATH",
    "PATH",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_ENGINE_ITERATION_TIMEOUT_S",
    "PYTORCH_CUDA_ALLOC_CONF",
    "NCCL_SOCKET_IFNAME",
    "NCCL_IB_HCA",
]
for key in keys:
    print(f"  {key}={os.environ.get(key)}")
PY

section "done"
printf 'OUTPUT=%s\n' "${OUT}"
