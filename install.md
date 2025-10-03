## RL-setup
In order to install atlas-rag with gpu, it is recommended for you to first install pytorch-gpu with cuda and faiss-gpu, then you can run pip insatll atlas-rag to install necessary packages.

As faiss-gpu only support CUDA 11.4 and 12.1 for now. so,
1. Set up clean environment
```bash
conda remove -n atlas-rag-gpu-test --all
conda create -n atlas-rag-gpu-test python=3.10 pip
conda activate atlas-rag-gpu-test
```
2. Install pytorch 
(cuda 12.1 for verl v0.0.1 & vllm 0.6.3, which search-R1 used)
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
(cuda 12.4 for most updated verl)
``` bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
https://pytorch.org/get-started/locally/

3. For faiss-gpu (For cuda 11.8 and 12.1)
```bash
conda install -c pytorch -c nvidia faiss-gpu
```
For faiss-gpu (For cuda <= 12.5)
```bash
conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5'
```

Pip install dev_requirement (TODO to include smooth installation)
```bash
pip install -r dev_requirement.txt
```

4. Set up for verl
```bash
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
git clone verl for set up of verl
```
git clone https://github.com/volcengine/verl.git
```
cd to verl, then (for fsdp)
``` shell
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

finally
```
cd verl
pip install --no-deps -e .
```

If there exist error

replace manually 
```python
from apex import amp 
```
to
```python
from torch.cuda import amp
```
https://www.cnblogs.com/AikN/p/18893668

## Install atlas-rag through pip
```bash
pip install atlas-rag
```