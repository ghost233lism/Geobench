start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /nfs/sunboyuan/Geobench/dataset/data_train/indoor/images_train_10000 \
  --annotation-file /nfs/sunboyuan/Geobench/dataset/meta_data_train/indoor_train.jsonl \
  --output-dir /nfs/sunboyuan/Geobench/dataset/data_train/indoor/indoor_train \
  --models Qwen3.5-9B,gemma-3-12b-it,InternVL3_5-8B \
  --disable-model-thinking \
  --gpu-memory-utilization 0.8 \
  --samples-per-model 8,8,8 \
  --temperature 0.9 \
  --top-p 0.9 \
  --tau-sigma 0.6 \
  --cuda-visible-devices 0,1 \
  --infer-batch-size 32 \
  --tensor-parallel-size 2 \
  --reward-mode latlon_geoscore \
  --max-model-len 40000 \
  --variance-mode exp_std \
  --keep-count 5000

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"