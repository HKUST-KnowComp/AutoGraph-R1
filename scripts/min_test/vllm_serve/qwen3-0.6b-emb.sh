export NCCL_P2P_LEVEL=NVL
vllm serve Qwen/Qwen3-Embedding-0.6B \
  --host 0.0.0.0 \
  --port 8128 \
  --gpu-memory-utilization 0.1 \
  --tensor-parallel-size 1 \