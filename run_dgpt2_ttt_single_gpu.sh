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

# run with correct --results_dir and pipe to tee
  
# python src/experiments/run_energy_aware_ttt.py \
#   --results_dir    $RESULTS_DIR \
#   --lambda_balance 0.10 \
#   --lambda_entropy 0.01 \
#   --lambda_kl      0.01 \
#   --router_temperature 1.0 \
#   --batch_size     8 \
#   --seq_length     64 \
#   --num_experts    8 \
#   --num_batches    200 \
#   --num_epochs     4 \
# | tee "$RESULTS_DIR/test_energy_aware_ttt.log"

python src/experiments/run_distilgpt2_moe_ttt.py --lambda_energy 0.01 --num_experts 16 --num_batches 200 --num_epochs 5

echo "Job completed successfully!"
echo "Check results in: $RESULTS_DIR"