start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /nfs/sunboyuan/Geobench/dataset/data_train/shape/train_sample_500 \
  --annotation-file /nfs/sunboyuan/Geobench/dataset/meta_data_train/gadm_shape_train_500.jsonl \
  --output-dir /nfs/sunboyuan/Geobench/dataset/data_train/shape/train_latlon_geoscore \
  --models InternVL3_5-8B,gemma-3-12b-it,Qwen3.5-9B,Qwen3-VL-8B-Instruct \
  --disable-model-thinking \
  --gpu-memory-utilization 0.8 \
  --samples-per-model 4,4,4,4 \
  --temperature 0.8 \
  --top-p 0.8 \
  --cuda-visible-devices 0,1 \
  --tensor-parallel-size 2 \
  --reward-mode latlon_geoscore \
  --variance-mode percentile \
  --max-images 30 \
  --keep-count 20

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"