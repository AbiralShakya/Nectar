#!/bin/bash
#SBATCH --job-name=hardware_synthetic_ttt_test_simple
#SBATCH --output=hardware_synthetic_ttt_test_simple_%j.out
#SBATCH --error=hardware_synthetic_ttt_test_simple_%j.err
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

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/hardware_synthetic_ttt_test_simple_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

echo "=== Hardware + Synthetic Stress TTT Test (Simplified) ==="
echo "Testing MoE + TTT system with REAL hardware metrics + controlled synthetic stress"
echo "Results will be saved to: $RESULTS_DIR"

# Install pynvml if not available
echo "=== Step 1: Checking Hardware Monitoring Setup ==="
python -c "import pynvml" 2>/dev/null || {
    echo "Installing pynvml for hardware monitoring..."
    pip install pynvml
}

# Test 1: Energy-Aware TTT with Real Hardware + Synthetic Stress (Inline)
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
        --num_batches 100 \
        --num_epochs 3 \
        --enable_thermal_awareness \
        --enable_noise_injection \
        --noise_level 0.05 \
        --error_margin 0.1 \
        --output_file "$RESULTS_DIR/hardware_synthetic_ttt_lambda_${lambda_energy}.json" \
        --benchmark \
    | tee "$RESULTS_DIR/hardware_synthetic_ttt_lambda_${lambda_energy}.log"
done

# Test 2: Simple Demo Script (if the above fails)
echo "=== Step 3: Running Simple Demo Script ==="
python test_hardware_synthetic_demo.py 2>&1 | tee "$RESULTS_DIR/hardware_demo.log"

# Generate simple summary report
echo "=== Step 4: Generating Simple Summary Report ==="
python -c "
import json
import glob
import os

results_dir = '$RESULTS_DIR'
summary = {
    'test_config': {
        'hardware_metrics_used': True,
        'synthetic_stress_scenarios': ['energy_imbalance', 'thermal_imbalance', 'combined_stress'],
        'lambda_energy_values': [0.001, 0.01, 0.05, 0.1, 0.2]
    },
    'results': {},
    'timestamp': $(date +%s)
}

# Load any available results
for result_file in glob.glob(os.path.join(results_dir, '*.json')):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            filename = os.path.basename(result_file)
            summary['results'][filename] = data
    except:
        pass

# Save summary
with open(os.path.join(results_dir, 'simple_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Simple summary saved to:', os.path.join(results_dir, 'simple_summary.json'))
" | tee "$RESULTS_DIR/summary_report.log"

echo "=== Hardware + Synthetic Stress TTT Test Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo ""
echo "Key files generated:"
echo "  - hardware_synthetic_ttt_lambda_*.json: Hardware + synthetic stress results"
echo "  - hardware_demo.log: Simple demo output"
echo "  - simple_summary.json: Summary report"
echo ""
echo "Check the logs for detailed performance metrics and routing adaptation analysis."
echo "Look for 'Adaptation Score' values to see how well the router responded to synthetic stress." 