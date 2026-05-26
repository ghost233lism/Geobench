start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/group_geo_selector.py \
  --selection-jsonl /nfs/sunboyuan/Geobench/dataset/data_train/street/street_train_grpo_no_mean_product/selection.jsonl \
  --annotation-jsonl /nfs/sunboyuan/Geobench/dataset/meta_data_train/ground_surface.jsonl \
  --output-dir /nfs/sunboyuan/Geobench/dataset/data_train/ground_surface/ground_train \
  --keep-count 1 \
  --random-seed 42 \
  --models Qwen3.5-9B,gemma-3-12b-it,InternVL3_5-8B \
  --disable-model-thinking \
  --gpu-memory-utilization 0.8 \
  --samples-per-model 8,8,8 \
  --temperature 0.9 \
  --top-p 0.9 \
  --cuda-visible-devices 0,1,2,3 \
  --infer-batch-size 128 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --reward-mode latlon_geoscore \
  --max-model-len 40000

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"
