#!/bin/bash
#SBATCH --job-name=ttt_energy_aware_triton_test
#SBATCH --output=ttt_energy_aware_triton_test_%j.out
#SBATCH --error=ttt_energy_aware_triton_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

module purge
module load anaconda3/2024.10
module load cudatoolkit/12.6

# ensure HF_HOME is set in your ~/.bashrc
source ~/.bashrc
conda activate topological_ml

export CUDA_VISIBLE_DEVICES=0

cd /scratch/gpfs/as0714/hardware_efficient_ml
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $PWD"

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/hpc_kcm_test_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

# Install Triton if not already installed
echo "Checking Triton installation..."
python -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "Installing Triton..."
    pip install triton-nightly
}

# Test multiple lambda_energy values to find optimal energy-accuracy trade-off
echo "Testing different energy penalty values..."

# Test 1: Higher energy penalty
echo "=== Testing lambda_energy=0.1 ==="
python src/experiments/run_distilgpt2_moe_ttt_triton.py \
    --lambda_energy 0.1 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --ttt_every 10 \
    --num_batches 50 \
    --num_epochs 2 \
    --benchmark \
| tee "$RESULTS_DIR/triton_moe_lambda_0.1.log"

# Test 2: Very high energy penalty
echo "=== Testing lambda_energy=1.0 ==="
python src/experiments/run_distilgpt2_moe_ttt_triton.py \
    --lambda_energy 1.0 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --ttt_every 10 \
    --num_batches 50 \
    --num_epochs 2 \
    --benchmark \
| tee "$RESULTS_DIR/triton_moe_lambda_1.0.log"

# Test 3: Original value for comparison
echo "=== Testing lambda_energy=0.01 (original) ==="
python src/experiments/run_distilgpt2_moe_ttt_triton.py \
    --lambda_energy 0.01 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --ttt_every 10 \
    --num_batches 50 \
    --num_epochs 2 \
    --benchmark \
| tee "$RESULTS_DIR/triton_moe_lambda_0.01.log"

echo "Job completed successfully!"
echo "Check results in: $RESULTS_DIR"
echo "Compare energy savings across different lambda_energy values" 