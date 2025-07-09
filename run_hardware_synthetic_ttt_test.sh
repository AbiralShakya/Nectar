#!/bin/bash
#SBATCH --job-name=hardware_synthetic_ttt_test
#SBATCH --output=hardware_synthetic_ttt_test_%j.out
#SBATCH --error=hardware_synthetic_ttt_test_%j.err
#SBATCH --time=02:00:00
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

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/hardware_synthetic_ttt_test_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

echo "=== Hardware + Synthetic Stress TTT Test ==="
echo "Testing MoE + TTT system with REAL hardware metrics + controlled synthetic stress"
echo "Results will be saved to: $RESULTS_DIR"

# Install pynvml if not available
echo "=== Step 1: Checking Hardware Monitoring Setup ==="
python -c "import pynvml" 2>/dev/null || {
    echo "Installing pynvml for hardware monitoring..."
    pip install pynvml
}

# Test 1: Energy-Aware TTT with Real Hardware + Synthetic Stress
echo "=== Step 2: Energy-Aware TTT with Hardware Metrics + Synthetic Stress ==="

# Test different lambda_energy values with real hardware + synthetic stress
for lambda_energy in 0.001 0.01 0.05 0.1 0.2; do
    echo "=== Testing lambda_energy=$lambda_energy ==="
    python src/experiments/test_energy_aware_ttt_hardware_synthetic.py \
        --lambda_energy $lambda_energy \
        --num_experts 16 \
        --moe_top_k 2 \
        --batch_size 8 \
        --seq_length 64 \
        --d_model 768 \
        --num_batches 200 \
        --num_epochs 3 \
        --enable_thermal_awareness \
        --enable_noise_injection \
        --noise_level 0.05 \
        --error_margin 0.1 \
        --output_file "$RESULTS_DIR/hardware_synthetic_ttt_lambda_${lambda_energy}.json" \
        --benchmark \
    | tee "$RESULTS_DIR/hardware_synthetic_ttt_lambda_${lambda_energy}.log"
done

# Test 2: Thermal-Aware Routing with Real Hardware
echo "=== Step 3: Thermal-Aware Routing with Real Hardware ==="
python src/experiments/test_thermal_aware_routing_hardware.py \
    --num_gpus 1 \
    --num_experts 16 \
    --batch_size 16 \
    --seq_length 128 \
    --d_model 768 \
    --num_batches 100 \
    --thermal_scenarios "normal,hot,imbalanced,cool" \
    --memory_pressure_levels "0.3,0.5,0.7,0.9" \
    --output_file "$RESULTS_DIR/thermal_aware_routing_hardware_results.json" \
    --save_plots \
| tee "$RESULTS_DIR/thermal_aware_routing_hardware.log"

# Test 3: Comprehensive Hardware Validation
echo "=== Step 4: Comprehensive Hardware Validation ==="
python src/experiments/validate_hardware_ttt.py \
    --num_experts 16 \
    --moe_top_k 2 \
    --batch_size 16 \
    --seq_length 128 \
    --d_model 768 \
    --num_batches 150 \
    --num_epochs 5 \
    --lambda_energy 0.05 \
    --enable_thermal_awareness \
    --enable_noise_injection \
    --noise_level 0.03 \
    --error_margin 0.08 \
    --output_file "$RESULTS_DIR/hardware_validation_results.json" \
    --save_plots \
| tee "$RESULTS_DIR/hardware_validation.log"

# Generate comprehensive summary report
echo "=== Step 5: Generating Hardware + Synthetic Stress Summary Report ==="
python src/experiments/generate_hardware_synthetic_report.py \
    --results_dir "$RESULTS_DIR" \
    --output_file "$RESULTS_DIR/hardware_synthetic_summary.json" \
    --generate_plots \
| tee "$RESULTS_DIR/summary_report.log"

echo "=== Hardware + Synthetic Stress TTT Test Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo ""
echo "Key files generated:"
echo "  - hardware_synthetic_ttt_lambda_*.json: Hardware + synthetic stress results"
echo "  - thermal_aware_routing_hardware_results.json: Real thermal routing analysis"
echo "  - hardware_validation_results.json: Comprehensive hardware validation"
echo "  - hardware_synthetic_summary.json: Overall summary report"
echo ""
echo "Check the logs for detailed performance metrics and routing adaptation analysis."
echo "Look for 'Adaptation Score' values to see how well the router responded to synthetic stress." 