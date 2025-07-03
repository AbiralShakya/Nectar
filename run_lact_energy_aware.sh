#!/bin/bash
#SBATCH --job-name=lact_energy_aware_test
#SBATCH --output=lact_energy_aware_test_%j.out
#SBATCH --error=lact_energy_aware_test_%j.err
#SBATCH --time=02:00:00
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

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/lact_energy_test_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

# Install Triton if not already installed
echo "Checking Triton installation..."
python -c "import triton; print(f'Triton version: {triton.__version__}')" || {
    echo "Installing Triton..."
    pip install triton-nightly
}

echo "Running LaCT Energy-Aware MoE Experiments"
echo "Testing different chunk sizes and lambda values..."

# Test 1: Small chunk size (baseline comparison)
echo "=== Testing chunk_size=100, lambda_energy=0.01 ==="
python src/experiments/run_lact_energy_aware.py \
    --lambda_energy 0.01 \
    --chunk_size 100 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --num_batches 100 \
    --num_epochs 2 \
| tee "$RESULTS_DIR/lact_chunk100_lambda0.01.log"

# Test 2: Medium chunk size
echo "=== Testing chunk_size=500, lambda_energy=0.01 ==="
python src/experiments/run_lact_energy_aware.py \
    --lambda_energy 0.01 \
    --chunk_size 500 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --num_batches 100 \
    --num_epochs 2 \
| tee "$RESULTS_DIR/lact_chunk500_lambda0.01.log"

# Test 3: Large chunk size (Zhang et al. style)
echo "=== Testing chunk_size=1000, lambda_energy=0.01 ==="
python src/experiments/run_lact_energy_aware.py \
    --lambda_energy 0.01 \
    --chunk_size 1000 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --num_batches 100 \
    --num_epochs 2 \
| tee "$RESULTS_DIR/lact_chunk1000_lambda0.01.log"

# Test 4: Very large chunk size
echo "=== Testing chunk_size=2000, lambda_energy=0.01 ==="
python src/experiments/run_lact_energy_aware.py \
    --lambda_energy 0.01 \
    --chunk_size 2000 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --num_batches 100 \
    --num_epochs 2 \
| tee "$RESULTS_DIR/lact_chunk2000_lambda0.01.log"

# Test 5: Different lambda with optimal chunk size
echo "=== Testing chunk_size=1000, lambda_energy=0.05 ==="
python src/experiments/run_lact_energy_aware.py \
    --lambda_energy 0.05 \
    --chunk_size 1000 \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 8 \
    --seq_length 64 \
    --num_batches 100 \
    --num_epochs 2 \
| tee "$RESULTS_DIR/lact_chunk1000_lambda0.05.log"

echo "Job completed successfully!"
echo "Check results in: $RESULTS_DIR"
echo "Key metrics to analyze:"
echo "1. GPU utilization improvement with larger chunks"
echo "2. Energy savings vs accuracy trade-off"
echo "3. LaCT update frequency and stability"
echo "4. Muon update effectiveness" 