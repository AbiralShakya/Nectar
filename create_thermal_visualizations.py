#!/usr/bin/env python3
"""
Create visualizations for synthetic thermal experiment results.
Run this separately if matplotlib is available.
"""

import json
import argparse
from pathlib import Path
import sys

def create_visualizations(results_file: str, output_dir: str):
    """Create visualizations from experiment results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        print("matplotlib or seaborn not available. Skipping visualizations.")
        return
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    config = results['experiment_config']
    time_series = results['time_series_data']
    cold_experts = config['cold_experts']
    hot_experts = config['hot_experts']
    
    # 1. Expert Usage Over Time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    steps = [step['step'] for step in time_series]
    
    # Plot individual expert usage
    for expert_id in range(config['num_experts']):
        usage_trend = results['expert_usage'][expert_id]['usage_trend']
        if expert_id in cold_experts:
            ax1.plot(steps, usage_trend, label=f'Cold Expert {expert_id}', alpha=0.7)
        else:
            ax2.plot(steps, usage_trend, label=f'Hot Expert {expert_id}', alpha=0.7)
    
    ax1.set_title('Cold Expert Usage Over Time')
    ax1.set_ylabel('Usage Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Hot Expert Usage Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Usage Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'expert_usage_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Temperature vs Usage Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Collect data points
    temperatures = []
    usages = []
    colors = []
    
    for expert_id in range(config['num_experts']):
        total_usage = results['expert_usage'][expert_id]['total_usage']
        avg_temp = np.mean([
            step['expert_usage'][expert_id]['temperature'] 
            for step in time_series
        ])
        
        temperatures.append(avg_temp)
        usages.append(total_usage)
        colors.append('blue' if expert_id in cold_experts else 'red')
    
    # Create scatter plot
    scatter = ax.scatter(temperatures, usages, c=colors, s=100, alpha=0.7)
    
    # Add labels
    for i, expert_id in enumerate(range(config['num_experts'])):
        ax.annotate(f'E{expert_id}', (temperatures[i], usages[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Average Temperature (°C)')
    ax.set_ylabel('Total Usage Count')
    ax.set_title('Expert Usage vs Temperature')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Cold Experts (0°C)'),
        Patch(facecolor='red', alpha=0.7, label='Hot Experts (60-80°C)')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_path / 'temperature_vs_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Expert Diversity Growth
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diversity = [step['expert_diversity'] for step in time_series]
    
    ax.plot(steps, diversity, linewidth=2, color='green')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Expert Diversity (Entropy)')
    ax.set_title('Expert Diversity Growth Over Time')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(steps, diversity, 1)
    p = np.poly1d(z)
    ax.plot(steps, p(steps), "--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'diversity_growth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Energy vs Thermal Imbalance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    energy = [step['total_energy'] for step in time_series]
    thermal_imbalance = [step['thermal_imbalance'] for step in time_series]
    
    scatter = ax.scatter(thermal_imbalance, energy, c=range(len(energy)), 
                       cmap='viridis', alpha=0.7)
    
    ax.set_xlabel('Thermal Imbalance')
    ax.set_ylabel('Energy Consumption (J)')
    ax.set_title('Energy vs Thermal Imbalance')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time Step')
    
    plt.tight_layout()
    plt.savefig(output_path / 'energy_vs_thermal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Cold vs Hot Expert Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calculate statistics
    cold_usage = []
    hot_usage = []
    cold_energy = []
    hot_energy = []
    
    for step in time_series:
        cold_step_usage = sum(
            exp['usage_count'] for exp in step['expert_usage'] 
            if exp['expert_id'] in cold_experts
        )
        hot_step_usage = sum(
            exp['usage_count'] for exp in step['expert_usage'] 
            if exp['expert_id'] in hot_experts
        )
        
        cold_step_energy = sum(
            exp['energy_cost'] for exp in step['expert_usage'] 
            if exp['expert_id'] in cold_experts
        )
        hot_step_energy = sum(
            exp['energy_cost'] for exp in step['expert_usage'] 
            if exp['expert_id'] in hot_experts
        )
        
        cold_usage.append(cold_step_usage)
        hot_usage.append(hot_step_usage)
        cold_energy.append(cold_step_energy)
        hot_energy.append(hot_step_energy)
    
    # Plot 1: Usage comparison
    ax1.plot(steps, cold_usage, label='Cold Experts', color='blue')
    ax1.plot(steps, hot_usage, label='Hot Experts', color='red')
    ax1.set_title('Usage Comparison')
    ax1.set_ylabel('Usage Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy comparison
    ax2.plot(steps, cold_energy, label='Cold Experts', color='blue')
    ax2.plot(steps, hot_energy, label='Hot Experts', color='red')
    ax2.set_title('Energy Consumption')
    ax2.set_ylabel('Energy (J)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Usage ratio over time
    total_usage = [c + h for c, h in zip(cold_usage, hot_usage)]
    cold_ratio = [c / t if t > 0 else 0 for c, t in zip(cold_usage, total_usage)]
    hot_ratio = [h / t if t > 0 else 0 for h, t in zip(hot_usage, total_usage)]
    
    ax3.plot(steps, cold_ratio, label='Cold Ratio', color='blue')
    ax3.plot(steps, hot_ratio, label='Hot Ratio', color='red')
    ax3.set_title('Usage Ratio Over Time')
    ax3.set_ylabel('Ratio')
    ax3.set_xlabel('Time Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy efficiency
    cold_efficiency = [c / (c + h) if (c + h) > 0 else 0 for c, h in zip(cold_usage, hot_usage)]
    ax4.plot(steps, cold_efficiency, color='green')
    ax4.set_title('Cold Expert Efficiency')
    ax4.set_ylabel('Efficiency Ratio')
    ax4.set_xlabel('Time Step')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cold_vs_hot_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create visualizations for thermal experiment")
    parser.add_argument("--results_file", type=str, required=True, help="Path to results JSON file")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Output directory for plots")
    
    args = parser.parse_args()
    
    create_visualizations(args.results_file, args.output_dir)

if __name__ == "__main__":
    main() 