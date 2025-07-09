#!/usr/bin/env python3
"""
Minimal test script for synthetic thermal experiment (no external dependencies).
Run this first to verify everything works before submitting to SLURM.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiments.run_synthetic_thermal_experiment_minimal import SyntheticThermalExperimentMinimal
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Minimal test of synthetic thermal experiment")
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
    
    print("=== Minimal Test of Synthetic Thermal Experiment ===")
    print(f"Lambda Energy: {test_args.lambda_energy}")
    print(f"Number of Batches: {test_args.num_batches}")
    print(f"Batch Size: {test_args.batch_size}")
    print(f"Sequence Length: {test_args.seq_length}")
    
    try:
        # Run experiment
        experiment = SyntheticThermalExperimentMinimal(test_args)
        results = experiment.run_experiment()
        
        # Save results
        output_path = Path(test_args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / test_args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        experiment.print_summary(results)
        
        print(f"\nTest completed successfully!")
        print(f"Results saved to: {output_path / test_args.output_file}")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 