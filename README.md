# AutoGraph-R1

**AutoGraph-R1** is a framework designed for constructing and utilizing knowledge graphs (KGs) to enhance retrieval-augmented generation (RAG). It comprises two main stages:

1. **Graph Constructor Training Stage**: Trains a model for KG construction from text corpora.
2. **Inference Stage**: Utilizes the trained model to build KGs and perform RAG benchmarking.

This README provides detailed instructions for environment setup, dependency installation, and execution of both stages.

## Project Overview

AutoGraph-R1 optimizes knowledge graph (KG) construction for retrieval-augmented generation (RAG) using Reinforcement Learning (RL). It bridges the gap between KG creation and its application, training a large language model (LLM) constructor to generate graphs that boost task performance. By approaching graph generation as a policy learning challenge, AutoGraph-R1 leverages novel task-aware reward functions, allowing graphs to serve effectively as both knowledge carriers and indices, enhancing question answering (QA) tasks. For a detailed overview, refer to the project diagram:

![AutoGraph-R1 Overview](image/autograph-r1.png)

## Setup Instructions

### 1. Environment Setup for Graph Constructor Training

The training stage requires a CUDA environment and the installation of VeRL, PyTorch, and Transformers.

#### Step 1: Install CUDA

- Install the appropriate CUDA and cudnn version for your GPU. Refer to the [NVIDIA CUDA Toolkit documentation](https://developer.nvidia.com/cuda-downloads) for installation guidance.
- Verify the installation with:
  ```bash
  nvcc --version
  ```

#### Step 2: Install VeRL

VeRL is utilized for the agent loop in the training stage. Follow these steps:

- Install VeRL in accordance with your CUDA version. Refer to the [VeRL installation guide](https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-from-custom-environment).
- For detailed setup instructions, see this [tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/703711904b3f69a187068916b29264c310f056cc/rlhf/verl/multi-turn/tool_examples/agent_loop.md) (Note: This tutorial is in Chinese).

Our VeRL modifications are based on version 0.5.0.dev0.

#### Step 3: Install PyTorch and Transformers

Ensure that PyTorch and Transformers are compatible with your CUDA version. You may need to reinstall if discrepancies arise.

- For PyTorch, our testing environment is v2.7.1+cu12.6. [Installation instructions](https://pytorch.org/get-started/previous-versions/).
- For Transformers, we use version 4.53.3.

Ensure compatibility by checking the [PyTorch compatibility tool](https://pytorch.org/get-started/locally/).

### 2. Environment Setup for Inference Stage

The inference stage employs an additional package, [AutoSchemaKG](https://github.com/HKUST-KnowComp/AutoSchemaKG), for custom prompt KG creation pipeline.

#### Install Atlas-RAG

In the same Python environment, install the `atlas-rag` package:

```bash
pip install atlas-rag
```

We use version >=v0.0.4.post2.

## Running the Training Stage

**Note**: These scripts assume the use of 2xH100 GPUs. Adjust `gpu_memory_utilization` and `CUDA_VISIBLE_DEVICES` based on your hardware.

Edit `config.ini` in `autograph/rag_server/` with:

```ini
[vllm]
URL = http://0.0.0.0:8129/v1
KEY = EMPTY
[vllm_emb]
URL = http://0.0.0.0:8128/v1
KEY = EMPTY
```

URLs align with ports set in `qwen3-0.6b-emb.sh` and `qwen2.5-7b-vllm.sh` in `autograph-r1/scripts/vllm_serve`.

### For 3B Model Training

**Deploy Embedding Model:**

```bash
bash scripts/vllm_serve/qwen3-0.6b-emb.sh
```

**Deploy Generator Model:**

```bash
bash scripts/vllm_serve/qwen2.5-7b-vllm.sh
```

For Graph Retriever:

```bash
bash scripts/autograph-r1/run_qwen2.5-3b_instruct_graph.sh
```

For Graph-Based Text Retriever:

```bash
bash scripts/autograph-r1/run_qwen2.5-3b_instruct_with_distract-iterative-hipporag-2.sh
```

### For 7B Model Training

**Deploy Embedding Model:**

```bash
bash scripts/vllm_serve/qwen3-0.6b-emb.sh
```

For Graph Retriever:

```bash
bash scripts/autograph-r1/run_qwen2.5-7b_instruct_graph.sh
```

For Graph-Based Text Retriever:

```bash
bash scripts/autograph-r1/run_qwen2.5-7b-instruct_with_distract-iterative-hipporag-2.sh
```

## Running the Inference Stage

The inference stage includes constructing a knowledge graph (KG) using a fine-tuned language model and RAG benchmarking.

### 1. Knowledge Graph Construction

To construct a knowledge graph from a general corpus, run:

```bash
python benchmark/autograph/custom_kg_extraction.py
```

- **Input**: Specify the `model_name` in the script or via command-line arguments (see the script for details).
- **Output**: The script generates a knowledge graph stored in the specified output directory.

### 2. RAG Benchmarking

The benchmarking process evaluates two methods: graph retriever and graph-based text retriever. Update the URL/port in scripts as needed.

#### 2.1 Graph Retriever

Run the graph retriever benchmark:

```bash
python benchmark/autograph/benchmarking_graph.py
```

#### 2.2 Graph-Based Text Retriever

Run the text retriever benchmark:

```bash
python benchmark/autograph/benchmarking_text.py
```