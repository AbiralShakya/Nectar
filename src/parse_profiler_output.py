# scripts/parse_profiler_output.py
import os
import sqlite3
import json
import pandas as pd
import argparse
import numpy as np
from typing import Dict, Any
import re # For regular expressions to parse filenames

# Assuming KernelCostModel is in src/moe_models.py
# Make sure your Python path allows this import (e.g., by running from project root)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from moe_models import KernelCostModel # Import the KernelCostModel class


def parse_nsys_sqlite(sqlite_path: str) -> Dict[str, Any]:
    """
    Parses an nsys SQLite database to extract kernel-level metrics
    (latency, energy_joules, temp_impact).
    Assumes a single main operation was profiled per .sqlite file.
    """
    conn = None
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        # 1. Get main kernel information (e.g., GEMM, elementwise ops)
        # We look for the longest running kernel as a proxy for the main operation.
        # This is a simplification; a full parser would analyze the trace.
        cursor.execute("""
            SELECT
                name, start, end,
                (end - start) / 1000000.0 AS duration_ms
            FROM
                CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
            ORDER BY
                duration_ms DESC
            LIMIT 1
        """)
        main_kernel_info = cursor.fetchone()

        if not main_kernel_info:
            print(f"Warning: No main kernel found in {sqlite_path}")
            return {}

        kernel_name, kernel_start_ns, kernel_end_ns, kernel_duration_ms = main_kernel_info

        # 2. Get GPU power and temperature metrics over the kernel's duration
        # Query GPU_METRICS_RAW for power and temperature samples within the kernel's timeframe
        # nsys measures in milliWatts for power.
        
        # Power query
        cursor.execute(f"""
            SELECT
                value
            FROM
                GPU_METRICS_RAW
            WHERE
                timestamp >= {kernel_start_ns} AND timestamp <= {kernel_end_ns}
                AND metric_name = 'gpu__power_usage.power.avg'
        """)
        power_samples = [row[0] / 1000.0 for row in cursor.fetchall()] # Convert mW to W

        # Temperature query
        cursor.execute(f"""
            SELECT
                value
            FROM
                GPU_METRICS_RAW
            WHERE
                timestamp >= {kernel_start_ns} AND timestamp <= {kernel_end_ns}
                AND metric_name = 'gpu__temperature.gpu.avg'
        """)
        temp_samples = [row[0] for row in cursor.fetchall()]

        avg_power_watt = np.mean(power_samples) if power_samples else 0.0
        avg_temp_c = np.mean(temp_samples) if temp_samples else 0.0

        # Calculate energy (Joules)
        energy_joules = avg_power_watt * (kernel_duration_ms / 1000.0) # ms to seconds

        return {
            "kernel_name_in_trace": kernel_name, # The actual CUDA kernel name
            "latency_ms": kernel_duration_ms,
            "energy_joules": energy_joules,
            "temp_impact": avg_temp_c # Using average temp during kernel as impact proxy
        }

    except sqlite3.Error as e:
        print(f"Error processing {sqlite_path}: {e}")
        return {}
    finally:
        if conn:
            conn.close()

def parse_ncu_report(ncu_path: str) -> Dict[str, Any]:
    """
    Parses an NCU report to extract detailed performance counters.
    This is illustrative; a full parser would be more complex.
    Requires 'ncu' command line tool to export to CSV first.
    """
    # NCU reports are not easily parsed programmatically directly from .ncu-rep
    # It's usually exported to CSV: ncu --csv --output-dir <path> --section <section> --page <page> --export <file>.ncu-rep
    # Or, use ncu --import <file>.ncu-rep --export csv -o temp.csv
    
    # For now, this is a placeholder. You'd typically look for metrics like:
    # sm__cycles_elapsed.avg.per_second, dram__bytes_read.sum, flop_count_hp_fma.sum
    print(f"Warning: NCU parsing is a placeholder. Requires manual export to CSV or deeper ncu API integration for {ncu_path}")
    
    # Return dummy data for now
    return {
        "ncu_sm_util": np.random.uniform(0.5, 1.0),
        "ncu_dram_bw": np.random.uniform(500, 1000)
    }


def main():
    parser = argparse.ArgumentParser(description="Parse profiler outputs and build KernelCostModel")
    parser.add_argument("--profile_base_dir", type=str, default="profiling_data_kernels_iter1",
                        help="Base directory where nsys/ncu profiles are stored.")
    parser.add_argument("--output_json", type="str", default="kernel_cost_model.json",
                        help="Output JSON file for the KernelCostModel (will be saved in kernel_cost_models/ directory).")
    args = parser.parse_args()

    nsys_traces_dir = os.path.join(args.profile_base_dir, "nsys_traces")
    ncu_reports_dir = os.path.join(args.profile_base_dir, "ncu_reports") # For future use

    all_op_costs = []

    # Map op_name from script to a cleaner name for KernelCostModel if desired
    # e.g., 'linear_fc1' -> 'Linear_Layer_FC1'
    # This is important for consistency in the KernelCostModel lookup in moe_models.py and routers.py
    op_name_map = {
        "linear_fc1": "linear_fc1",
        "relu": "relu",
        "linear_fc2": "linear_fc2",
        "dequant_unpack_op": "dequant_unpack_op"
    }

    # Iterate through nsys traces and parse them
    for filename in os.listdir(nsys_traces_dir):
        if filename.endswith(".sqlite"):
            match = re.match(r"(\w+)_d(\d+)_b(\d+)", filename.replace(".sqlite", ""))
            if match:
                op_name_raw, d_model_str, batch_size_str = match.groups()
                
                # Check if this op_name is one we care about from our profiler script
                if op_name_raw not in op_name_map:
                    print(f"Skipping unrecognized op_name in filename: {filename}")
                    continue

                op_type = op_name_map[op_name_raw] # Use the cleaned/mapped name
                d_model = int(d_model_str)
                batch_size = int(batch_size_str)

                sqlite_path = os.path.join(nsys_traces_dir, filename)
                print(f"Parsing {sqlite_path}...")
                
                # Parse Nsys data
                nsys_metrics = parse_nsys_sqlite(sqlite_path)

                if nsys_metrics:
                    entry = {
                        "op_type": op_type,
                        "d_model": d_model, # Include d_model for context, even if not key in KCM
                        "batch_size": batch_size,
                        **nsys_metrics # Add all extracted nsys metrics
                    }
                    
                    # Optionally add NCU metrics if enabled and available
                    ncu_filename = filename.replace(".sqlite", ".ncu-rep")
                    ncu_path = os.path.join(ncu_reports_dir, ncu_filename)
                    if os.path.exists(ncu_path):
                        ncu_metrics = parse_ncu_report(ncu_path)
                        entry.update(ncu_metrics)
                    
                    all_op_costs.append(entry)
                else:
                    print(f"Warning: No valid metrics found for {filename}")
            else:
                print(f"Skipping unrecognized file format: {filename}")

    if not all_op_costs:
        print("\nNo kernel cost data extracted. Ensure profiler ran correctly and output .sqlite files.")
        print("Creating dummy cost model for demonstration purposes.")
        
        # Populate with dummy data if no real data was parsed
        dummy_ops = ["linear_fc1", "relu", "linear_fc2", "dequant_unpack_op"]
        dummy_batch_sizes = [1, 32, 128]
        for op in dummy_ops:
            for bs in dummy_batch_sizes:
                all_op_costs.append({
                    "op_type": op,
                    "d_model": args.d_model, # Use the d_model from arguments
                    "batch_size": bs,
                    "kernel_name_in_trace": f"dummy_{op}_kernel",
                    "latency_ms": (np.random.rand() * 1.0 + 0.1) * (bs / 32),
                    "energy_joules": (np.random.rand() * 0.1 + 0.01) * (bs / 32),
                    "temp_impact": (np.random.rand() * 0.05 + 0.005) * (bs / 32)
                })

    # Convert to DataFrame and save to JSON
    cost_df = pd.DataFrame(all_op_costs)
    
    # Ensure output directory exists for the JSON
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'kernel_cost_models')
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, args.output_json)

    cost_df.to_json(output_json_path, orient="records", indent=4)
    print(f"\nKernel Cost Model saved to {output_json_path}")
    print("\n--- Kernel Cost Model Preview (first 5 rows) ---")
    print(cost_df.head())

    # Example usage of the KernelCostModel class (for verification)
    print("\nExample KernelCostModel lookup verification:")
    loaded_cost_model_instance = KernelCostModel(data_path=output_json_path)
    cost = loaded_cost_model_instance.get_cost(op_type="linear_fc1", batch_size=32)
    print(f"Cost for 'linear_fc1' (batch 32): {cost}")
    cost = loaded_cost_model_instance.get_cost(op_type="dequant_unpack_op", batch_size=1)
    print(f"Cost for 'dequant_unpack_op' (batch 1): {cost}")
    cost = loaded_cost_model_instance.get_cost(op_type="relu", batch_size=512)
    print(f"Cost for 'relu' (batch 512): {cost}")


if __name__ == "__main__":
    main()