start_time=$(date +%s)

python /nfs/sunboyuan/Geobench/ms-swift/data_clean/text_latlon_benchmark.py \
  --input-file /nfs/sunboyuan/Geobench/dataset/test_lat_lon.json \
  --output-dir /nfs/sunboyuan/Geobench/dataset/test_lat_lon_benchmark \
  --models InternVL3_5-8B,gemma-3-12b-it,Qwen3.5-9B,Qwen3-VL-8B-Instruct,Qwen3_8b \
  --disable-model-thinking \
  --gpu-memory-utilization 0.8 \
  --infer-batch-size 128 \
  --max-num-seqs 128 \
  --max-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --cuda-visible-devices 0,1 \
  --tensor-parallel-size 2 \
  --max-samples 1000

end_time=$(date +%s)
duration=$((end_time - start_time))
echo "任务完成，用时: ${duration} 秒"
