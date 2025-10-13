CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8129 \
  --gpu-memory-utilization 0.15 \
  --tensor-parallel-size 2 \
  --max-model-len 16384