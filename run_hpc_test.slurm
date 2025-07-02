#!/bin/bash
#SBATCH --job-name=ttt_kcm_test
#SBATCH --output=ttt_kcm_test_%j.out
#SBATCH --error=ttt_kcm_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4          


# Load modules (adjust for your cluster)
module purge
module load anaconda3/2024.10 
module load cudatoolkit/12.6

source ~/.bashrc
conda activate topological_ml

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Navigate to the correct project directory
PROJECT_ROOT_DIR="/home/as0714/hardware_efficient_ml"
cd ${PROJECT_ROOT_DIR}

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $PWD"
echo "Testing: TTT + Kernel Cost Model Integration"

# Verify the script exists
if [ ! -f "Hardware_MoE/hpc_single_gpu_test.py" ]; then
    echo "ERROR: hpc_single_gpu_test.py not found in Hardware_MoE/"
    ls -la Hardware_MoE/
    exit 1
fi

# Run the enhanced test with kernel cost model integration
python Hardware_MoE/hpc_single_gpu_test.py \
    --d_model 256 \
    --num_layers 4 \
    --batch_size 8 \
    --seq_length 512 \
    --num_runs 30 \
    --test_moe \
    --num_experts 4 \
    --top_k 2 \
    --ttt_chunk_size 512 \
    --ttt_update_frequency 128 \
    --energy_aware_lr 1e-4 \
    --muon_enabled \
    --output_dir "Hardware_MoE/results/hpc_kcm_test_${SLURM_JOB_ID}"

echo "Job completed successfully!"
echo "Check results in: Hardware_MoE/results/hpc_kcm_test_${SLURM_JOB_ID}/" 