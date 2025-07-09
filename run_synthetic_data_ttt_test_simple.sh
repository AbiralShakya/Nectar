#!/bin/bash
#SBATCH --job-name=synthetic_ttt_energy_thermal_test
#SBATCH --output=synthetic_ttt_energy_thermal_test_%j.out
#SBATCH --error=synthetic_ttt_energy_thermal_test_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

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

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/synthetic_ttt_test_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

echo "=== Synthetic Data TTT Energy/Thermal Routing Test ==="
echo "Testing MoE + TTT system with synthetic data to validate routing factors"
echo "Results will be saved to: $RESULTS_DIR"

# Test 1: Energy-Aware TTT with Inline Synthetic Data (Core Test)
echo "=== Step 1: Energy-Aware TTT Routing Test ==="

# Test different lambda_energy values with inline synthetic data
for lambda_energy in 0.001 0.01 0.05 0.1 0.2; do
    echo "=== Testing lambda_energy=$lambda_energy ==="
    python src/experiments/test_energy_aware_ttt_synthetic_inline.py \
        --lambda_energy $lambda_energy \
        --num_experts 16 \
        --moe_top_k 2 \
        --batch_size 8 \
        --seq_length 64 \
        --d_model 768 \
        --num_batches 100 \
        --num_epochs 3 \
        --enable_thermal_awareness \
        --enable_noise_injection \
        --noise_level 0.05 \
        --error_margin 0.1 \
        --output_file "$RESULTS_DIR/energy_aware_ttt_lambda_${lambda_energy}.json" \
        --benchmark \
    | tee "$RESULTS_DIR/energy_aware_ttt_lambda_${lambda_energy}.log"
done

# Test 2: Thermal-Aware Routing Test
echo "=== Step 2: Thermal-Aware Routing Test ==="
python src/experiments/test_thermal_aware_routing_simple.py \
    --num_gpus 1 \
    --num_experts 16 \
    --batch_size 16 \
    --seq_length 128 \
    --d_model 768 \
    --num_batches 50 \
    --thermal_scenarios "normal,hot,imbalanced,cool" \
    --memory_pressure_levels "0.3,0.5,0.7,0.9" \
    --output_file "$RESULTS_DIR/thermal_aware_routing_results.json" \
    --save_plots \
| tee "$RESULTS_DIR/thermal_aware_routing.log"

# Generate comprehensive summary report
echo "=== Step 3: Generating Summary Report ==="
python src/experiments/generate_synthetic_test_report.py \
    --results_dir "$RESULTS_DIR" \
    --output_file "$RESULTS_DIR/synthetic_test_summary.json" \
    --generate_plots \
| tee "$RESULTS_DIR/summary_report.log"

echo "=== Synthetic Data TTT Test Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo ""
echo "Key files generated:"
echo "  - energy_aware_ttt_lambda_*.json: Energy-aware routing results"
echo "  - thermal_aware_routing_results.json: Thermal routing analysis"
echo "  - synthetic_test_summary.json: Overall summary report"
echo ""
echo "Check the logs for detailed performance metrics and routing factor validation." 