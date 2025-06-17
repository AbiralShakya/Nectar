#!/bin/bash
#SBATCH --job-name=adaptive-moe
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-3

TOP_K_LIST=(1 2 4 8)
NUM_EXPERTS_LIST=(4 8 16 32)

TOP_K=${TOP_K_LIST[$SLURM_ARRAY_TASK_ID]}
NUM_EXPERTS=${NUM_EXPERTS_LIST[$SLURM_ARRAY_TASK_ID]}

# Use conda directly from user environment
source ~/.bashrc
conda activate topological_ml

echo "Hostname: $(hostname)"
echo "Running with top_k=$TOP_K, num_experts=$NUM_EXPERTS"
nvidia-smi
which python

python /home/as0714/hardware_efficient_ml/adaptive_moe_rev4nograph.py \
    --top_k $TOP_K \
    --num_experts $NUM_EXPERTS \
