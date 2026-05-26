PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /nfs/sunboyuan/model/Qwen3.5-4B \
    --enable_thinking false \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.80 \
    --vllm_max_num_seqs 8 \
    --vllm_max_model_len 8192 \
    --vllm_enforce_eager true \
    --stream true
