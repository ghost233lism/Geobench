start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/geo_data_selector.py \
  --data-dir /nfs/sunboyuan/Geobench/dataset/data_train/street/train_pic \
  --annotation-file /nfs/sunboyuan/Geobench/dataset/meta_data_train/street_train_25k.jsonl \
  --output-dir /nfs/sunboyuan/Geobench/dataset/data_train/street/street_train \
  --models Qwen3.5-9B,gemma-3-12b-it,InternVL3_5-8B \
  --disable-model-thinking \
  --gpu-memory-utilization 0.8 \
  --samples-per-model 8,8,8 \
  --temperature 0.9 \
  --top-p 0.9 \
  --tau-sigma 0.6 \
  --cuda-visible-devices 0,1,2,3 \
  --infer-batch-size 128 \
  --tensor-parallel-size 2 \
  --data-parallel-size 2 \
  --reward-mode latlon_geoscore \
  --max-model-len 40000 \
  --variance-mode exp_std \
  --keep-count 15000

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"