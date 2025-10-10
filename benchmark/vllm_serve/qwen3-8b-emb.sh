CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-Embedding-8B \
  --host 0.0.0.0 \
  --port 8128 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 2 \
  --max-model-len 32768