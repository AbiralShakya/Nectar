#!/usr/bin/env python3
"""
Analyze energy-aware routing results to find optimal lambda value.
"""

import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def extract_metrics_from_log(log_file):
    """Extract metrics from a log file."""
    if not os.path.exists(log_file):
        return None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract lambda value from filename
    lambda_match = re.search(r'lambda_([0-9.]+)', log_file)
    lambda_val = float(lambda_match.group(1)) if lambda_match else None
    
    # Extract final metrics
    lines = content.split('\n')
    final_loss = None
    final_power = None
    final_diversity = None
    
    for line in reversed(lines):
        if '[energy_aware]' in line and 'Batch' in line:
            # Extract final batch metrics
            loss_match = re.search(r'Loss: ([0-9.]+)', line)
            power_match = re.search(r'Power: ([0-9.]+)', line)
            diversity_match = re.search(r'Diversity: ([0-9.]+)', line)
            
            if loss_match:
                final_loss = float(loss_match.group(1))
            if power_match:
                final_power = float(power_match.group(1))
            if diversity_match:
                final_diversity = float(diversity_match.group(1))
            break
    
    return {
        'lambda_energy': lambda_val,
        'final_loss': final_loss,
        'final_power': final_power,
        'final_diversity': final_diversity
    }

def analyze_results(results_dir):
    """Analyze all results in the directory."""
    log_files = glob.glob(os.path.join(results_dir, 'triton_moe_lambda_*.log'))
    
    results = []
    for log_file in log_files:
        metrics = extract_metrics_from_log(log_file)
        if metrics:
            results.append(metrics)
    
    # Sort by lambda value
    results.sort(key=lambda x: x['lambda_energy'])
    
    return results

def plot_analysis(results):
    """Plot the analysis results."""
    if not results:
        print("No results to plot")
        return
    
    df = pd.DataFrame(results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss vs Lambda
    ax1.plot(df['lambda_energy'], df['final_loss'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Lambda Energy')
    ax1.set_ylabel('Final Loss')
    ax1.set_title('Accuracy vs Energy Penalty')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Power vs Lambda
    ax2.plot(df['lambda_energy'], df['final_power'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Lambda Energy')
    ax2.set_ylabel('Final Power (W)')
    ax2.set_title('Power Consumption vs Energy Penalty')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diversity vs Lambda
    ax3.plot(df['lambda_energy'], df['final_diversity'], 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Lambda Energy')
    ax3.set_ylabel('Final Diversity')
    ax3.set_title('Expert Diversity vs Energy Penalty')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Pareto frontier (Loss vs Power)
    ax4.scatter(df['final_power'], df['final_loss'], c=df['lambda_energy'], 
                cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Power Consumption (W)')
    ax4.set_ylabel('Final Loss')
    ax4.set_title('Pareto Frontier: Loss vs Power')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = ax4.scatter(df['final_power'], df['final_loss'], c=df['lambda_energy'], 
                         cmap='viridis', s=100, alpha=0.7)
    plt.colorbar(scatter, ax=ax4, label='Lambda Energy')
    
    plt.tight_layout()
    return fig

def find_optimal_lambda(results):
    """Find the optimal lambda value based on Pareto efficiency."""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    
    # Normalize metrics to [0, 1] range
    loss_norm = (df['final_loss'] - df['final_loss'].min()) / (df['final_loss'].max() - df['final_loss'].min())
    power_norm = (df['final_power'] - df['final_power'].min()) / (df['final_power'].max() - df['final_power'].min())
    
    # Calculate Pareto efficiency score (lower is better)
    df['pareto_score'] = loss_norm + power_norm
    
    # Find the best lambda
    best_idx = df['pareto_score'].idxmin()
    best_lambda = df.loc[best_idx, 'lambda_energy']
    
    print(f"\n=== Pareto Analysis ===")
    print(f"Best lambda_energy: {best_lambda}")
    print(f"Corresponding loss: {df.loc[best_idx, 'final_loss']:.4f}")
    print(f"Corresponding power: {df.loc[best_idx, 'final_power']:.1f}W")
    print(f"Corresponding diversity: {df.loc[best_idx, 'final_diversity']:.3f}")
    
    return best_lambda

def main():
    # Find the most recent results directory
    base_dir = "/scratch/gpfs/as0714/hardware_efficient_ml/results"
    if not os.path.exists(base_dir):
        print(f"Results directory not found: {base_dir}")
        return
    
    # Get the most recent results directory
    result_dirs = glob.glob(os.path.join(base_dir, "hpc_kcm_test_*"))
    if not result_dirs:
        print("No result directories found")
        return
    
    latest_dir = max(result_dirs, key=os.path.getctime)
    print(f"Analyzing results from: {latest_dir}")
    
    # Analyze results
    results = analyze_results(latest_dir)
    
    if not results:
        print("No valid results found")
        return
    
    # Print summary
    print("\n=== Results Summary ===")
    for result in results:
        print(f"Lambda {result['lambda_energy']}: Loss={result['final_loss']:.4f}, "
              f"Power={result['final_power']:.1f}W, Diversity={result['final_diversity']:.3f}")
    
    # Find optimal lambda
    optimal_lambda = find_optimal_lambda(results)
    
    # Plot results
    fig = plot_analysis(results)
    if fig:
        plot_file = os.path.join(latest_dir, 'energy_analysis.png')
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nAnalysis plot saved to: {plot_file}")
        plt.close(fig)

if __name__ == "__main__":
    main() 