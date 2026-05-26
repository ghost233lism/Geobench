cd /nfs/sunboyuan/Geobench/ms-swift
pip install -U ms-swift
pip install -U "transformers==5.2.*" "qwen_vl_utils>=0.0.14" peft liger-kernel
pip install -U "flash-linear-attention>=0.4.2" --no-build-isolation
pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation
pip install "flash-attn==2.8.3" --no-build-isolation
pip install deepspeed
pip install -U "vllm>=0.17.0"
pip install -U "transformers==5.2.*"
pip install swanlab

SWANLAB_API_KEY="${SWANLAB_API_KEY:-9gYXBe3ZrP0uJEhOMIFmU}"
swanlab login --api-key "${SWANLAB_API_KEY}"

bash tools/run_geogrpo_A100.sh
