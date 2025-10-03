export NCCL_P2P_LEVEL=NVL
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8129 \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 1 \
  --max-model-len 32768