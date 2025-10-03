#!/bin/bash
#SBATCH --job-name=auto_graph_training_7B_combined
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --partition=large
#SBATCH --account=awskgraph
#SBATCH --output=slurm_combined_%A_%a.out
#SBATCH --error=slurm_combined_%A_%a.err

srun --ntasks=1 \
    --container-writable \
    --container-remap-root \
    --no-container-mount-home \
    --container-mounts /home/httsangaj/project/autograph-r1:/workspace/autograph-r1 \
    --container-image /project/awskgraph/VeRL/verl.sqsh \
    --exclusive \
    --output task_%j_%t.log \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl && cd /workspace/autograph-r1 && nvidia-smi" &

srun --ntasks=1 \
    --container-writable \
    --container-remap-root \
    --no-container-mount-home \
    --container-mounts /home/httsangaj/project/autograph-r1:/workspace/autograph-r1 \
    --container-image /project/awskgraph/VeRL/verl.sqsh \
    --exclusive \
    --output task_%j_%t.log \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl && cd /workspace/autograph-r1 && nvidia-smi" &
wait