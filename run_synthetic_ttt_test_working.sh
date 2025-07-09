#!/bin/bash
#SBATCH --job-name=synthetic_ttt_test_working
#SBATCH --output=synthetic_ttt_test_working_%j.out
#SBATCH --error=synthetic_ttt_test_working_%j.err
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

RESULTS_DIR=/scratch/gpfs/as0714/hardware_efficient_ml/results/synthetic_ttt_test_working_${SLURM_JOB_ID}
mkdir -p "$RESULTS_DIR"

echo "=== Synthetic TTT Test (Working Version) ==="
echo "Testing MoE + TTT system with synthetic data and controlled stress scenarios"
echo "Results will be saved to: $RESULTS_DIR"

# Test 1: Energy-Aware TTT with Synthetic Data (Inline)
echo "=== Step 1: Energy-Aware TTT with Synthetic Data ==="

# Test different lambda_energy values with synthetic data
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

# Generate simple summary report
echo "=== Step 2: Generating Summary Report ==="
python -c "
import json
import glob
import os
import numpy as np

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

# Load energy-aware TTT results
for lambda_val in [0.001, 0.01, 0.05, 0.1, 0.2]:
    result_file = os.path.join(results_dir, f'energy_aware_ttt_lambda_{lambda_val}.json')
    if os.path.exists(result_file):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                summary['results'][f'lambda_{lambda_val}'] = data
        except Exception as e:
            print(f'Error loading {result_file}: {e}')
            pass

# Calculate overall performance
if summary['results']:
    energy_savings = []
    accuracy_losses = []
    adaptation_scores = []
    
    for lambda_val, result in summary['results'].items():
        if 'results' in result:
            energy_savings.append(result['results'].get('energy_savings_percent', 0))
            accuracy_losses.append(result['results'].get('accuracy_loss_percent', 0))
            
            # Calculate adaptation score based on expert usage distribution
            expert_usage = result['results'].get('expert_usage_distribution', [])
            if expert_usage:
                # Simple adaptation score: how uniform is the distribution
                expected_uniform = 1.0 / len(expert_usage)
                variance = np.var(expert_usage)
                adaptation_score = 1.0 / (1.0 + variance / (expected_uniform ** 2))
                adaptation_scores.append(adaptation_score)
    
    summary['overall_performance'] = {
        'avg_energy_savings': float(np.mean(energy_savings)) if energy_savings else 0.0,
        'avg_accuracy_loss': float(np.mean(accuracy_losses)) if accuracy_losses else 0.0,
        'avg_adaptation_score': float(np.mean(adaptation_scores)) if adaptation_scores else 0.0,
        'best_lambda_energy': float([0.001, 0.01, 0.05, 0.1, 0.2][np.argmax(energy_savings)]) if energy_savings else 0.05,
        'best_adaptation_lambda': float([0.001, 0.01, 0.05, 0.1, 0.2][np.argmax(adaptation_scores)]) if adaptation_scores else 0.05
    }

# Save summary
with open(os.path.join(results_dir, 'working_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print('Working summary saved to:', os.path.join(results_dir, 'working_summary.json'))

# Print key findings
if 'overall_performance' in summary:
    perf = summary['overall_performance']
    print(f'\\n=== Key Findings ===')
    print(f'Average Energy Savings: {perf[\"avg_energy_savings\"]:.2f}%')
    print(f'Average Accuracy Loss: {perf[\"avg_accuracy_loss\"]:.2f}%')
    print(f'Average Adaptation Score: {perf[\"avg_adaptation_score\"]:.3f}')
    print(f'Best Lambda for Energy: {perf[\"best_lambda_energy\"]}')
    print(f'Best Lambda for Adaptation: {perf[\"best_adaptation_lambda\"]}')
    
    # Grade the performance
    if perf['avg_adaptation_score'] > 0.7:
        grade = 'A'
    elif perf['avg_adaptation_score'] > 0.5:
        grade = 'B'
    elif perf['avg_adaptation_score'] > 0.3:
        grade = 'C'
    else:
        grade = 'D'
    
    print(f'Overall Grade: {grade}')
    
    if perf['avg_energy_savings'] > 0:
        print('✓ System achieved energy savings')
    else:
        print('⚠ System used more energy than baseline')
        
    if perf['avg_adaptation_score'] > 0.5:
        print('✓ Router showed good adaptation to synthetic stress')
    else:
        print('⚠ Router needs tuning for better adaptation')
" | tee "$RESULTS_DIR/summary_report.log"

echo "=== Synthetic TTT Test Complete ==="
echo "All results saved to: $RESULTS_DIR"
echo ""
echo "Key files generated:"
echo "  - energy_aware_ttt_lambda_*.json: Energy-aware routing results"
echo "  - working_summary.json: Overall summary report"
echo ""
echo "Check the logs for detailed performance metrics and routing adaptation analysis."
echo "Look for 'Adaptation Score' values to see how well the router responded to synthetic stress." 