#!/bin/bash

# --- Configuration ---
# Variable for training datasets

# Variable fo

VLLM_EMB_SCRIPT="scripts/vllm_serve/qwen3-0.6b-emb.sh"
VLLM_MAIN_SCRIPT="scripts/vllm_serve/qwen2.5-7b-vllm.sh"
RAG_SERVER_SCRIPT="autograph-r1/rag_server/rag_server.py"
TRAINING_SCRIPT="scripts/autograph-r1/run_qwen2.5-7b_instruct_graph_construct.sh"

GPU_FOR_EMB=0,1,2,3 # (set the same 4 in slurm)
GPU_FOR_MAIN=0,1,2,3 # (set the same 4 in slurm)
GPU_FOR_VERL=4,5,6,7 # (set the another 4 in slurm)

# --- Start vLLM Embedding Server ---
echo "Starting vLLM Embedding Server..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_EMB nohup bash "$VLLM_EMB_SCRIPT" > vllm_emb_server.log 2>&1 &
VLLM_EMB_PID=$!
echo "vLLM Embedding Server PID: $VLLM_EMB_PID"

# --- Start vLLM Main Server ---
echo "Starting vLLM Main Server..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_MAIN nohup bash "$VLLM_MAIN_SCRIPT" > vllm_main_server.log 2>&1 &
VLLM_MAIN_PID=$!
echo "vLLM Main Server PID: $VLLM_MAIN_PID"

# --- Wait for Both Servers to Initialize ---
echo "Waiting for vLLM Embedding Server and Main Server to initialize..."
while true; do
    EMB_READY=false
    MAIN_READY=false

    if grep -q "INFO:     Application startup complete." vllm_emb_server.log; then
        EMB_READY=true
    else
        echo "Still waiting for vLLM Embedding Server to start..."
    fi

    if grep -q "INFO:     Application startup complete." vllm_main_server.log; then
        MAIN_READY=true
    else
        echo "Still waiting for vLLM Main Server to start..."
    fi

    if [ "$EMB_READY" = true ] && [ "$MAIN_READY" = true ]; then
        echo "Both vLLM Embedding Server and Main Server have started successfully."
        break
    fi

    sleep 5  # Check every 5 seconds
done

# --- Extract Server PIDs ---
VLLM_EMB_SERVER_PID=$(grep "INFO:     Started server process" vllm_emb_server.log | awk '{print $NF}' | tr -d '[]')
VLLM_MAIN_SERVER_PID=$(grep "INFO:     Started server process" vllm_main_server.log | awk '{print $NF}' | tr -d '[]')
echo "vLLM Embedding Server Process PID: $VLLM_EMB_SERVER_PID"
echo "vLLM Main Server Process PID: $VLLM_MAIN_SERVER_PID"

# --- Start RAG Server ---
echo "Starting RAG Server..."
nohup python3 "$RAG_SERVER_SCRIPT" > vllm_rag_server.log 2>&1 &
RAG_SERVER_PID=$!
echo "RAG Server PID: $RAG_SERVER_PID"

# --- Wait for RAG Server to Initialize ---
echo "Waiting for RAG Server to initialize..."
while true; do
    if grep -q "INFO:     Application startup complete." vllm_rag_server.log; then
        echo "RAG Server has started successfully."
        break
    fi
    echo "Still waiting for RAG Server to start..."
    sleep 5  # Check every 5 seconds
done

# --- Extract RAG Server PID ---
RAG_SERVER_PROCESS_PID=$(grep "INFO:     Started server process" vllm_rag_server.log | awk '{print $NF}' | tr -d '[]')
echo "RAG Server Process PID: $RAG_SERVER_PROCESS_PID"

# --- Run Training Job ---
echo "Starting Training Job..."
CUDA_VISIBLE_DEVICES=$GPU_FOR_VERL bash "$TRAINING_SCRIPT"
TRAINING_EXIT_CODE=$?

# --- Shut Down Servers ---
echo "Shutting down servers..."
kill -SIGINT $VLLM_EMB_SERVER_PID 2>/dev/null || true
kill -SIGINT $VLLM_MAIN_SERVER_PID 2>/dev/null || true
kill -SIGINT $RAG_SERVER_PROCESS_PID 2>/dev/null || true
echo "Waiting for all servers to shutdown..."
MAX_WAIT=300  # 5 minutes in seconds
WAITED=0
while true; do
    EMB_SHUTDOWN=false
    MAIN_SHUTDOWN=false
    RAG_SHUTDOWN=false

    if ! ps -p $VLLM_EMB_SERVER_PID > /dev/null; then
        EMB_SHUTDOWN=true
    else
        echo "Still waiting for vLLM Embedding Server to shutdown..."
    fi

    if ! ps -p $VLLM_MAIN_SERVER_PID > /dev/null; then
        MAIN_SHUTDOWN=true
    else
        echo "Still waiting for vLLM Main Server to shutdown..."
    fi

    if ! ps -p $RAG_SERVER_PROCESS_PID > /dev/null; then
        RAG_SHUTDOWN=true
    else
        echo "Still waiting for RAG Server to shutdown..."
    fi

    if [ "$EMB_SHUTDOWN" = true ] && [ "$MAIN_SHUTDOWN" = true ] && [ "$RAG_SHUTDOWN" = true ]; then
        echo "All servers have shut down successfully."
        break
    fi

    WAITED=$((WAITED + 5))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "Timeout: Servers did not shutdown within 5 minutes. Forcing kill..."
        kill -9 $VLLM_EMB_SERVER_PID 2>/dev/null || true
        kill -9 $VLLM_MAIN_SERVER_PID 2>/dev/null || true
        kill -9 $RAG_SERVER_PROCESS_PID 2>/dev/null || true
        break
    fi
    sleep 5  # Check every 5 seconds
done

# --- Check Training Exit Code ---
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $TRAINING_EXIT_CODE."
    exit $TRAINING_EXIT_CODE
fi