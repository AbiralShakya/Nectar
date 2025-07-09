#!/usr/bin/env python3
"""
Test the full experiment workflow locally.
Runs multiple lambda values and generates summary report.
"""

import subprocess
import json
import os
import glob
from pathlib import Path

def run_experiment(lambda_val, output_dir):
    """Run a single experiment with given lambda value."""
    cmd = [
        "python", "src/experiments/run_synthetic_thermal_experiment_minimal.py",
        "--lambda_energy", str(lambda_val),
        "--num_experts", "8",
        "--moe_top_k", "2",
        "--batch_size", "8",
        "--seq_length", "64",
        "--d_model", "768",
        "--num_batches", "50",  # Smaller for testing
        "--output_dir", output_dir,
        "--output_file", f"thermal_experiment_lambda_{lambda_val}.json"
    ]
    
    print(f"Running experiment with lambda_energy = {lambda_val}")
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print(f"✅ Completed lambda_energy = {lambda_val}")
        return True
    else:
        print(f"❌ Failed lambda_energy = {lambda_val}")
        print(f"Error: {result.stderr}")
        return False

def generate_summary(results_dir):
    """Generate summary report from all experiment results."""
    print("Creating summary report...")
    
    # Collect all results
    results_files = []
    
    # Look for files directly in the directory
    if os.path.exists(results_dir):
        for file in os.listdir(results_dir):
            if file.endswith('.json'):
                results_files.append(os.path.join(results_dir, file))
    
    # Look for files in subdirectories
    subdir_files = glob.glob(f'{results_dir}/*/*.json', recursive=True)
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
        except Exception as e:
            print(f'Error reading {file_path}: {e}')
    
    print(f'Processed {len(summary)} experiments')
    
    # Save summary
    summary_file = os.path.join(results_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'Summary saved to {summary_file}')
    print('\nExperiment Summary:')
    for lambda_val, metrics in summary.items():
        print(f'{lambda_val}: Energy={metrics["total_energy"]:.2f}J, '
              f'Diversity={metrics["avg_diversity"]:.3f}, '
              f'Savings={metrics["energy_savings_percent"]:.2f}%')
    
    return summary

def main():
    print("=== Full Experiment Workflow Test ===")
    
    # Create results directory
    results_dir = "test_full_workflow_results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Test lambda values
    lambda_values = [0.0, 0.01, 0.1, 0.5]
    
    # Run experiments
    successful_experiments = 0
    for lambda_val in lambda_values:
        if run_experiment(lambda_val, results_dir):
            successful_experiments += 1
        print("----------------------------------------")
    
    print(f"Completed {successful_experiments}/{len(lambda_values)} experiments")
    
    # Generate summary
    if successful_experiments > 0:
        summary = generate_summary(results_dir)
        print(f"\n✅ Full workflow test completed successfully!")
        print(f"Results saved to: {results_dir}")
        print(f"Summary: {json.dumps(summary, indent=2)}")
    else:
        print("\n❌ No experiments completed successfully")

if __name__ == "__main__":
    main() 