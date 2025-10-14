# AutoGraph-R1
[![GitHub stars](https://img.shields.io/github/stars/YOUR_REPO/AutoGraph-R1?style=for-the-badge&logo=github&logoColor=white&color=a29bfe&label=stars)](https://github.com/YOUR_REPO/AutoGraph-R1)
[![arXiv](https://img.shields.io/badge/arXiv-SOON-74b9ff?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/YOUR_PAPER_ID)
[![Python](https://img.shields.io/badge/Python-3.10%2B-0984e3?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](https://opensource.org/licenses/MIT)


### 🤔 **Is Your RAG Pipeline *Really* Optimized?**
**Reinforcement Learning • Task-Aware KG Construction • GraphRAG**

*✨Shifting from building "good" graphs to building demonstrably **"useful"** ones✨*


## 🚀 TL;DR
The effectiveness of Graph Retrieval-Augmented Generation (GraphRAG) is often hindered by a fundamental disconnect: the Knowledge Graph (KG) construction process is decoupled from the downstream task it's meant to serve. **AutoGraph-R1** is the first framework to bridge this gap by framing KG construction as a Reinforcement Learning (RL) problem. An LLM "constructor" agent is trained with rewards based on the generated graph's functional utility in a live RAG pipeline, directly optimizing for task performance.

### 🎯 **Key Features**
- **🤖 RL-Optimized KG Construction:** Trains an LLM to build graphs that are verifiably useful for a downstream RAG task.
- **📈 Task-Aware Reward Functions:** Includes two novel reward functions to optimize graphs as either direct knowledge carriers or as powerful knowledge indices.
- **🔗 Two-Stage Pipeline:** A clear separation between the graph constructor training stage and the inference/benchmarking stage.
- **🔬 Reproducible Benchmarking:** Provides scripts to reproduce our results and evaluate custom-built knowledge graphs on multiple QA benchmarks.


<div align="center">
  <figure>
    <img src="image/autograph-r1.png" alt="AutoGraph-R1 Overview" style="max-width: 100%; height: auto;">
    <br>
    <figcaption><em>Quick Overview of the AutoGraph-R1 Framework.</em></figcaption>
  </figure>
</div>

## 📋 Table of Contents
- [🚀 Get Started](#-get-started)
  - [1. System Prerequisites (CUDA) ⚙️](#1-system-prerequisites-cuda-⚙️)
  - [2. Install Core Dependencies 📦](#2-install-core-dependencies-)
  - [3. Install Inference Dependencies 🔍](#3-install-inference-dependencies-)
- [🧪 Running the Pipeline](#-running-the-pipeline)
  - [0. Initial Configuration 🛠️](#0-initial-configuration-️)
  - [Stage 1: Training the Graph Constructor 🏋️](#stage-1-training-the-graph-constructor-️)
  - [Stage 2: Inference and Benchmarking 📊](#stage-2-inference-and-benchmarking-)
- [🌟 Citation](#-citation)
- [📞 Contacts](#-contacts)


## 🚀 Get Started
This guide covers the environment setup for both the training and inference stages. All packages should be installed in the same environment.

### 1. System Prerequisites (CUDA) ⚙️
The training and inference stages require a system with an NVIDIA GPU and a compatible CUDA toolkit.
- **Install CUDA**: Install the appropriate CUDA and cuDNN version for your GPU.
- Refer to the [NVIDIA CUDA Toolkit documentation](https://developer.nvidia.com/cuda-12-6-0-download-archive) (CUDA 12.6 was installed for VeRL) for official installation instructions.
- **Verify Installation**: Check your CUDA version by running:
    ```bash
    nvcc --version
    ```

### 2. Install Core Dependencies 📦
Install the core libraries for deep learning and the RL agent loop.
- **PyTorch and Transformers**
  Ensure compatibility with your CUDA version. Our code was tested with:
  - **PyTorch:** `v2.7.1+cu126` (refer to [previous versions](https://pytorch.org/get-started/previous-versions/) for your specific CUDA build)
  - **Transformers:** `v4.53.3`

  ```bash
  # Example for CUDA 12.6 - adjust for your system
  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  pip install transformers==4.53.3
  ```
- **VeRL (for the RL agent loop)**
  Our modifications are based on `v0.5.0.dev0`.
  - Install VeRL by following the official [VeRL installation guide](https://verl.readthedocs.io/en/v0.5.x/start/install.html#install-from-custom-environment).
  - > **Note:** A detailed agent loop setup tutorial using VeRL is available [here](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/703711904b3f69a187068916b29264c310f056cc/rlhf/verl/multi-turn/tool_examples/agent_loop.md) (in Chinese).

### 3. Install Inference Dependencies 🔍
For the inference stage, an additional package is required for the KG creation pipeline.
- **Atlas-RAG**
  We use `v0.0.4.post2` or newer. Install it in the same environment:
  ```bash
  pip install "atlas-rag>=0.0.4.post2"
  ```
## 🧪 Running the Pipeline
The AutoGraph-R1 pipeline consists of a training stage and an inference stage.

### 0. Initial Configuration 🛠️
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

### Stage 1: Training the Graph Constructor 🏋️
This stage uses RL to fine-tune an LLM to build effective knowledge graphs.

> **Hardware Note:** The following scripts are configured for 2xH100 GPUs. You may need to adjust `gpu_memory_utilization` in the scripts and the `CUDA_VISIBLE_DEVICES` environment variable for your specific hardware.

**1. Launch the LLM API Servers**

First, launch the language models that will act as the environment (generator) and the embedding model for the RL loop. Open two separate terminal sessions for these.
- **Terminal 1: Launch Embedding Model Server:**
  ```bash
  bash scripts/vllm_serve/qwen3-0.6b-emb.sh
  ```
- **Terminal 2: Launch Generator Model Server (For 3B model):**
  ```bash
  bash scripts/vllm_serve/qwen2.5-7b-vllm.sh
  ```

**2. Run the Training Script**

In a third terminal, run the RL training loop. Choose one of the following scripts based on the desired reward function.

- **To train with the Graph Retriever reward (graph as a knowledge carrier):**
  ```bash
  # For a 3B parameter agent
  bash scripts/autograph-r1/run_qwen2.5-3b_instruct_graph.sh

  # For a 7B parameter agent (ensure generator server is not running)
  bash scripts/autograph-r1/run_qwen2.5-7b_instruct_graph.sh
  ```

- **To train with the Graph-Based Text Retriever reward (graph as a knowledge index):**
  ```bash
  # For a 3B parameter agent
  bash scripts/autograph-r1/run_qwen2.5-3b_instruct_with_distract-iterative-hipporag-2.sh
  # For a 7B parameter agent
  bash scripts/autograph-r1/run_qwen2.5-7b-instruct_with_distract-iterative-hipporag-2.sh
  ```

### Stage 2: Inference and Benchmarking 📊
Once you have a trained graph constructor, you can use it to build a knowledge graph from a corpus and benchmark its performance.

**1. Knowledge Graph Construction**

Use your fine-tuned model to extract a KG from a text corpus. Edit the script to point to your model and data.
- **Arguments**: Pass the `model_name` (the path to your fine-tuned model checkpoint) and other parameters inside the script or via the command line.
- **Run the script:**
  ```bash
  python benchmark/autograph/custom_kg_extraction.py
  ```
- **Output**: The constructed knowledge graph will be saved to the specified output directory.
- **For argument details, please refer to the script.**

**2. RAG Benchmarking**

Evaluate the performance of the generated KG using our benchmarking scripts. Ensure the model endpoints and KG paths in the scripts are correctly set.
- **Method 1: Graph Retriever Benchmark:**
  ```bash
  python benchmark/autograph/benchmarking_graph.py
  ```

- **Method 2: Graph-Based Text Retriever Benchmark:**
  ```bash
  python benchmark/autograph/benchmarking_text.py
  ```

## 🌟 Citation
If you use AutoGraph-R1 in your research, please cite our paper:
```
@misc{YOUR_CITATION_KEY,
      title={Directly Optimizing Knowledge Graph Construction for RAG using Reinforcement Learning}, 
      author={Author One and Author Two and Author Three},
      year={2025},
      eprint={YOUR_PAPER_ID},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/YOUR_PAPER_ID}, 
}
```