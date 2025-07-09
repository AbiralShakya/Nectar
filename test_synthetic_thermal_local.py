#!/usr/bin/env python3
"""
Local test script for synthetic thermal experiment.
Run this first to verify everything works before submitting to SLURM.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments.run_synthetic_thermal_experiment import SyntheticThermalExperiment
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Local test of synthetic thermal experiment")
    parser.add_argument("--lambda_energy", type=float, default=0.1, help="Energy penalty weight")
    parser.add_argument("--num_batches", type=int, default=20, help="Number of batches (small for testing)")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Override some parameters for quick testing
    test_args = argparse.Namespace(
        lambda_energy=args.lambda_energy,
        num_experts=8,
        moe_top_k=2,
        batch_size=4,  # Smaller for testing
        seq_length=32,  # Smaller for testing
        d_model=256,    # Smaller for testing
        num_batches=args.num_batches,
        output_dir=args.output_dir,
        output_file="test_results.json"
    )
    
    print("=== Local Test of Synthetic Thermal Experiment ===")
    print(f"Lambda Energy: {test_args.lambda_energy}")
    print(f"Number of Batches: {test_args.num_batches}")
    print(f"Batch Size: {test_args.batch_size}")
    print(f"Sequence Length: {test_args.seq_length}")
    
    try:
        # Run experiment
        experiment = SyntheticThermalExperiment(test_args)
        results = experiment.run_experiment()
        
        # Save results
        output_path = Path(test_args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / test_args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create basic visualization
        experiment.create_visualizations(results, test_args.output_dir)
        
        # Print summary
        print("\n=== Test Results ===")
        print(f"Total Energy: {results['results']['total_energy']:.2f}J")
        print(f"Average Diversity: {results['results']['avg_diversity']:.3f}")
        print(f"Energy Savings: {results['results']['energy_savings_percent']:.2f}%")
        print(f"TTT Updates: {results['results']['ttt_updates']}")
        
        # Check expert usage
        print("\n=== Expert Usage Summary ===")
        for expert_id in range(test_args.num_experts):
            usage_data = results['expert_usage'][expert_id]
            total_usage = usage_data['total_usage']
            is_cold = usage_data['is_cold']
            is_hot = usage_data['is_hot']
            
            expert_type = "COLD" if is_cold else "HOT" if is_hot else "UNKNOWN"
            print(f"Expert {expert_id} ({expert_type}): {total_usage} total usage")
        
        print(f"\nTest completed successfully!")
        print(f"Results saved to: {output_path / test_args.output_file}")
        print(f"Visualizations saved to: {output_path}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 