CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8128 \
  --gpu-memory-utilization 0.1 \
  --tensor-parallel-size 1 \