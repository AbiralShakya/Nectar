#!/bin/bash
#SBATCH --job-name=ttt_energy_aware_test
#SBATCH --output=ttt_energy_aware_test_%j.out
#SBATCH --error=ttt_energy_aware_test_%j.err
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

export CUDA_VISIBLE_DEVICES=0

# Navigate to the correct project directory
PROJECT_ROOT_DIR="/home/as0714/hardware_efficient_ml"
cd ${PROJECT_ROOT_DIR}

export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $PWD"
echo "Testing: EnergyAwareTTTRouter"

# Create results directory
RESULTS_DIR="/home/as0714/hardware_efficient_ml/results/hpc_kcm_test_${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Run the test script and redirect output to the results directory
python src/experiments/test_energy_aware_ttt.py | tee "$RESULTS_DIR/test_energy_aware_ttt.log"

# Move any unittest output or results to the results directory (if generated)
# Example: mv test_output.json "$RESULTS_DIR/" 2>/dev/null

echo "Job completed successfully!"
echo "Check results in: /home/as0714/hardware_efficient_ml/results/hpc_kcm_test_${SLURM_JOB_ID}/" 