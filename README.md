# AutoGraph-R1

**Directly Optimizing Knowledge Graph Construction for RAG using Reinforcement Learning**

<!-- [![Paper](https://img.shields.io/badge/paper-ARXIV_ID-B31B1B.svg)](https://arxiv.org/abs/YOUR_PAPER_ID) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

The effectiveness of Retrieval-Augmented Generation (RAG) is often hindered by a fundamental disconnect: the Knowledge Graph (KG) construction process is decoupled from the downstream task it's meant to serve. This results in suboptimal graph structures that don't maximize performance.

**AutoGraph-R1** is the first framework to bridge this gap. We directly optimize KG construction for task performance by framing it as a Reinforcement Learning (RL) problem. An LLM "constructor" is trained as a policy agent, receiving rewards based on the generated graph's functional utility in a live RAG pipeline. This approach shifts the paradigm from building intrinsically "good" graphs to building demonstrably **"useful"** ones.

![AutoGraph-R1 Overview](image/autograph-r1.png)

## Key Features

-   **RL-Optimized KG Construction:** Trains an LLM to build graphs that are verifiably useful for a downstream RAG task.
-   **Task-Aware Reward Functions:** Includes two novel reward functions to optimize graphs as either direct knowledge carriers or as powerful knowledge indices.
-   **Two-Stage Pipeline:** A clear separation between the graph constructor training stage and the inference/benchmarking stage.
-   **Reproducible Benchmarking:** Provides scripts to reproduce our results and evaluate custom-built knowledge graphs on multiple QA benchmarks.

## Setup Instructions

This guide covers the environment setup for both the training and inference stages. All packages should be installed in the same environment.

### Step 1: System Prerequisites (CUDA)

The training and inference stages require a system with an NVIDIA GPU and a compatible CUDA toolkit.

-   **Install CUDA**: Install the appropriate CUDA and cuDNN version for your GPU.
    -   Refer to the [NVIDIA CUDA Toolkit documentation](https://developer.nvidia.com/cuda-downloads) for official installation instructions.

-   **Verify Installation**: Check your CUDA version by running:
    ```bash
    nvcc --version
    ```

### Step 2: Install Core Dependencies

Install the core libraries for deep learning and the RL agent loop.

-   **PyTorch and Transformers**
    Ensure compatibility with your CUDA version. Our code was tested with:
    -   **PyTorch:** `v2.7.1+cu126` (refer to [previous versions](https://pytorch.org/get-started/previous-versions/) for your specific CUDA build)
    -   **Transformers:** `v4.53.3`

    ```bash
    # Example for CUDA 12.6 - adjust for your system
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    pip install transformers==4.53.3
    ```

-   **VeRL (for the RL agent loop)**
    Our modifications are based on `v0.5.0.dev0`.
    -   Install VeRL by following the official [VeRL installation guide](https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-from-custom-environment).
    -   > **Note:** A detailed agent loop setup tutorial using VeRL is available [here](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/703711904b3f69a187068916b29264c310f056cc/rlhf/verl/multi-turn/tool_examples/agent_loop.md) (in Chinese).

### Step 3: Install Inference Dependencies

For the inference stage, an additional package is required for the KG creation pipeline.

-   **Atlas-RAG**
    We use `v0.0.4.post2` or newer. Install it in the same environment:
    ```bash
    pip install "atlas-rag>=0.0.4.post2"
    ```


## Running the Pipeline

The AutoGraph-R1 pipeline consists of a training stage and an inference stage.

### Step 0: Initial Configuration

Before running any script, you must configure the API endpoints for your language models. These models will be served using `vllm`.

Edit the `config.ini` file in `autograph/rag_server/` to match the ports you will use to serve your models. The defaults align with our provided scripts.

```ini
[vllm]
URL = http://0.0.0.0:8129/v1
KEY = EMPTY

[vllm_emb]
URL = http://0.0.0.0:8128/v1
KEY = EMPTY
```

### Stage 1: Training the Graph Constructor

This stage uses RL to fine-tune an LLM to build effective knowledge graphs.

> **Hardware Note:** The following scripts are configured for 2xH100 GPUs. You may need to adjust `gpu_memory_utilization` in the scripts and the `CUDA_VISIBLE_DEVICES` environment variable for your specific hardware.

**1. Launch the LLM API Servers**

First, launch the language models that will act as the environment (generator) and the embedding model for the RL loop. Open two separate terminal sessions for these.

-   **Terminal 1: Launch Embedding Model Server:**
    ```bash
    bash scripts/vllm_serve/qwen3-0.6b-emb.sh
    ```
-   **Terminal 2: Launch Generator Model Server (For 3B model):**
    ```bash
    bash scripts/vllm_serve/qwen2.5-7b-vllm.sh
    ```

**2. Run the Training Script**

In a third terminal, run the RL training loop. Choose one of the following scripts based on the desired reward function.

-   **To train with the Graph Retriever reward (graph as a knowledge carrier):**
    ```bash
    # For a 3B parameter agent
    bash scripts/autograph-r1/run_qwen2.5-3b_instruct_graph.sh

    # For a 7B parameter agent (ensure generator server is not running)
    bash scripts/autograph-r1/run_qwen2.5-7b_instruct_graph.sh
    ```

-   **To train with the Graph-Based Text Retriever reward (graph as a knowledge index):**
    ```bash
    # For a 3B parameter agent
    bash scripts/autograph-r1/run_qwen2.5-3b_instruct_with_distract-iterative-hipporag-2.sh
    
    # For a 7B parameter agent
    bash scripts/autograph-r1/run_qwen2.5-7b-instruct_with_distract-iterative-hipporag-2.sh
    ```

### Stage 2: Inference and Benchmarking

Once you have a trained graph constructor, you can use it to build a knowledge graph from a corpus and benchmark its performance.

**1. Knowledge Graph Construction**

Use your fine-tuned model to extract a KG from a text corpus. Edit the script to point to your model and data.

-   **Arguments**: Pass the `model_name` (the path to your fine-tuned model checkpoint) and other parameters inside the script or via the command line.
-   **Run the script:**
    ```bash
    python benchmark/autograph/custom_kg_extraction.py
    ```
-   **Output**: The constructed knowledge graph will be saved to the specified output directory.
-   **For argument details, please refer to the script.**

**2. RAG Benchmarking**

Evaluate the performance of the generated KG using our benchmarking scripts. Ensure the model endpoints and KG paths in the scripts are correctly set.

-   **Method 1: Graph Retriever Benchmark:**
    ```bash
    python benchmark/autograph/benchmarking_graph.py
    ```

-   **Method 2: Graph-Based Text Retriever Benchmark:**
    ```bash
    python benchmark/autograph/benchmarking_text.py
    ```

<!-- ## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@misc{your_paper_id_here,
      title={Building effective knowledge graphs for Retrieval-Augmented Generation}, 
      author={First Author and Second Author and Third Author},
      year={2024},
      eprint={YOUR_ARXIV_ID_HERE},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``` -->