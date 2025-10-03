#!/bin/bash
#SBATCH --job-name=auto_graph_training_7B
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=awskgraph     
#SBATCH --output=slurm_%j.out     
#SBATCH --error=slurm_%j.err        

# Run the job submission script
srun --container-writable \
    --container-remap-root \
    --no-container-mount-home \
    --container-mounts /home/httsangaj/project/autograph-r1:/workspace/autograph-r1 \
    --container-image /project/awskgraph/VeRL/verl.sqsh \
    bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate verl &&  pip list && cd /workspace/autograph-r1 && bash submit_training_job_min.sh"