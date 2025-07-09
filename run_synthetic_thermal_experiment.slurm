#!/bin/bash
#SBATCH --job-name=synthetic_thermal_moe
#SBATCH --output=logs/synthetic_thermal_%j.out
#SBATCH --error=logs/synthetic_thermal_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Enable debug output
set -x

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.9

# Set up environment and ensure we're in the right directory
cd /Users/abiralshakya/Documents/Research/Hardware_MoE
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create results directory
mkdir -p results/thermal_experiment
mkdir -p logs

# Print debug info
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run the synthetic thermal experiment
echo "Starting Synthetic Thermal-Aware Energy-Adaptive Routing Experiment"
echo "Timestamp: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run with different lambda values to test energy awareness
for lambda in 0.0 0.01 0.1 0.5 1.0; do
    echo "Running experiment with lambda_energy = $lambda"
    
    # Test if the experiment script can be imported
    echo "Testing import..."
    python -c "import sys; sys.path.insert(0, '.'); from src.experiments.run_synthetic_thermal_experiment_minimal import SyntheticThermalExperimentMinimal; print('Import successful')"
    
    if [ $? -eq 0 ]; then
        echo "Import test passed, running experiment..."
        
        python src/experiments/run_synthetic_thermal_experiment_minimal.py \
            --lambda_energy $lambda \
            --num_experts 8 \
            --moe_top_k 2 \
            --batch_size 8 \
            --seq_length 64 \
            --d_model 768 \
            --num_batches 100 \
            --output_dir "results/thermal_experiment/lambda_${lambda}" \
            --output_file "thermal_experiment_lambda_${lambda}.json"
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed lambda_energy = $lambda"
        else
            echo "❌ Failed lambda_energy = $lambda"
        fi
    else
        echo "❌ Import test failed for lambda_energy = $lambda"
    fi
    
    echo "----------------------------------------"
done

# Run additional test with energy awareness verification
echo "Running energy awareness verification test..."
python test_energy_awareness_real.py \
    --lambda_energy 0.1 \
    --num_batches 50 \
    --output_file "results/thermal_experiment/energy_awareness_test.json"

echo "All experiments completed!"
echo "Final timestamp: $(date)"

# Final status check
echo "=== Final Status Check ==="
echo "Results directory contents:"
ls -la results/thermal_experiment/ 2>/dev/null || echo "Results directory not found"

echo "Summary file contents:"
if [ -f "results/thermal_experiment/experiment_summary.json" ]; then
    cat results/thermal_experiment/experiment_summary.json
else
    echo "Summary file not found"
fi

echo "=== Job completed ==="

# Create summary report
echo "Creating summary report..."
python -c "
import json
import glob
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, '.')

# Collect all results - look in both the main directory and subdirectories
results_files = []
results_dir = 'results/thermal_experiment'

print(f'Looking for results in: {os.path.abspath(results_dir)}')

# Look for files directly in the main directory
if os.path.exists(results_dir):
    print(f'Directory exists, listing contents:')
    for file in os.listdir(results_dir):
        print(f'  {file}')
        if file.endswith('.json'):
            results_files.append(os.path.join(results_dir, file))

# Look for files in subdirectories
subdir_files = glob.glob('results/thermal_experiment/*/*.json', recursive=True)
results_files.extend(subdir_files)

print(f'Found {len(results_files)} result files: {results_files}')

summary = {}

for file_path in results_files:
    try:
        print(f'Reading {file_path}...')
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'results' in data and 'experiment_config' in data:
            lambda_val = data['experiment_config'].get('lambda_energy', 'unknown')
            summary[f'lambda_{lambda_val}'] = {
                'total_energy': data['results']['total_energy'],
                'avg_diversity': data['results']['avg_diversity'],
                'energy_savings_percent': data['results']['energy_savings_percent'],
                'ttt_updates': data['results']['ttt_updates']
            }
            print(f'Successfully processed lambda_{lambda_val}')
        else:
            print(f'Skipping {file_path} - missing required fields')
            print(f'Keys in data: {list(data.keys()) if isinstance(data, dict) else \"not a dict\"}')
    except Exception as e:
        print(f'Error reading {file_path}: {e}')

print(f'Processed {len(summary)} experiments')

# Save summary
summary_file = 'results/thermal_experiment/experiment_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Summary saved to {summary_file}')
print('\\nExperiment Summary:')
for lambda_val, metrics in summary.items():
    print(f'{lambda_val}: Energy={metrics[\"total_energy\"]:.2f}J, '
          f'Diversity={metrics[\"avg_diversity\"]:.3f}, '
          f'Savings={metrics[\"energy_savings_percent\"]:.2f}%')

# Also print the summary JSON for debugging
print('\\nFull summary JSON:')
print(json.dumps(summary, indent=2))
" 