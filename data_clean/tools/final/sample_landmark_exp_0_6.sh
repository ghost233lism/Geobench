start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /nfs/sunboyuan/Geobench/dataset/data_train/landmark/pic \
  --annotation-file /nfs/sunboyuan/Geobench/dataset/meta_data_train/landmark_train.jsonl \
  --output-dir /nfs/sunboyuan/Geobench/dataset/data_train/landmark/train_landmark \
  --models gemma-3-12b-it,InternVL3_5-8B,Qwen3.5-9B \
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
  --variance-mode exp_std \
  --max-images 10009 \
  --keep-count 5000

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"