# scripts/expert_kernel_profiler.py
import torch
import torch.nn as nn
import os
import subprocess
import json
import time
import argparse
import tempfile
import sys # Import sys for checking Python executable path

# --- Configuration for Temporary Script Generation ---
# This dictionary maps a descriptive operation name to its PyTorch code.
# The code will be injected into a temporary Python script.
# Key variables: {d_model_in}, {d_model_out}, {batch_size}
# These will be dynamically replaced by the generator.
OPERATION_TEMPLATES = {
    "linear_fc1": { # Represents the first linear layer in your expert
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=True).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d * 2}
    },
    "relu": { # The ReLU activation after the first linear layer
        "code": """
            model = nn.ReLU().to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d * 2, "d_model_out": lambda d: d * 2} # Input/Output d_model are same for ReLU
    },
    "linear_fc2": { # The second linear layer in your expert
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=True).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d * 2, "d_model_out": lambda d: d}
    },
    # Add a dummy dequantization operation for the QuantizedExpert.
    # This will simulate the unpacking and scaling overhead.
    "dequant_unpack_op": {
        "code": """
            # Simulate a 4-bit packed input (e.g., from QuantizedExpert's weights)
            # This is not a real module, but a sequence of ops.
            # We profile the combined operations.
            packed_input = torch.randint(-8, 7, ({d_model_out}, {d_model_in} // 2), dtype=torch.int8, device=device) # (out_features, in_features/2)
            scales = torch.randn({d_model_out}, 1, dtype=torch.float16, device=device)

            # --- Simulated dequantization logic (matching moe_models.py's QuantizedExpert) ---
            low_nibbles = (packed_input & 0x0F).to(torch.int8)
            high_nibbles = (packed_input >> 4).to(torch.int8)
            low_nibbles = torch.where(low_nibbles > 7, low_nibbles - 16, low_nibbles)
            high_nibbles = torch.where(high_nibbles > 7, high_nibbles - 16, high_nibbles)
            
            unpacked_weights = torch.empty({d_model_out}, {d_model_in}, device=device, dtype=scales.dtype)
            unpacked_weights[:, 0::2] = high_nibbles.float()
            unpacked_weights[:, 1::2] = low_nibbles.float()
            _ = unpacked_weights * scales.to(unpacked_weights.dtype)
            # --- End simulated dequantization logic ---
        """,
        # d_model_in here represents the full input feature dimension for the linear layer
        # (e.g., d_model for fc1, d_model*2 for fc2).
        # d_model_out represents the output feature dimension.
        # This operation is essentially preparing the weights for a linear layer.
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d * 2} # Example for fc1's weight dequant
    }
}

def generate_isolated_script(op_name: str, d_model_base: int, batch_size: int, temp_script_path: str):
    """
    Generates a temporary Python script to run an isolated PyTorch operation.
    """
    template_info = OPERATION_TEMPLATES[op_name]
    
    # Map d_model_base to specific input/output d_model for the operation
    d_model_in = template_info["d_model_mapping"]["d_model_in"](d_model_base)
    d_model_out = template_info["d_model_mapping"]["d_model_out"](d_model_base)
    
    op_code = template_info["code"].format(
        d_model_in=d_model_in,
        d_model_out=d_model_out,
        batch_size=batch_size
    )

    script_content = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F # Needed for F.linear if it's used directly
import os
import sys

# Set CUDA_VISIBLE_DEVICES if specified by nsys/ncu, otherwise use default
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    device = torch.device(f"cuda:{{os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]}}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    print("CUDA not available. Exiting isolated script.")
    sys.exit(1) # Use sys.exit for robust exiting from subprocess

torch.cuda.empty_cache() # Clear cache for cleaner measurements

# Warm-up (important for consistent measurements, especially for GPU kernels)
for _ in range(20): # Increased warm-up iterations
    {op_code}
torch.cuda.synchronize() # Ensure warm-up completes

# Actual run for profiling - only one execution for a clean profile
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
{op_code} # Execute the operation for profiling
end_event.record()
torch.cuda.synchronize() # Crucial: Wait for GPU to finish

# Optional: Print timing for in-script verification. nsys/ncu don't directly use this.
print(f"Operation: {{op_name}}, d_model_base: {{d_model_base}}, batch_size: {{batch_size}}, Latency: {{start_event.elapsed_time(end_event):.4f}} ms")
"""
    with open(temp_script_path, "w") as f:
        f.write(script_content)
    # print(f"Generated temporary script: {temp_script_path}")


def run_profiler(profiler_cmd_base: list, python_script_path: str, output_path: str,
                 profile_type: str, op_name: str, d_model_base: int, batch_size: int):
    """
    Executes a profiler command on the generated Python script.
    """
    profile_filename_base = f"{op_name}_d{d_model_base}_b{batch_size}" # Define it here

    cmd = profiler_cmd_base + [
        f"--output={output_path}",
        "python", python_script_path
    ]
    
    # Check if the output file already exists. If yes, skip to save time during debugging.
    # For nsys, it adds .qdrep; for ncu, it adds .ncu-rep
    final_output_file = os.path.join(os.path.dirname(output_path), profile_filename_base + (".qdrep" if profile_type == "nsys" else ".ncu-rep")) # Correct path construction

    if os.path.exists(final_output_file):
        print(f"Skipping {profile_type} for {profile_filename_base}: Output file already exists.")
        return

    print(f"Running {profile_type} for {op_name} (d_model_base: {d_model_base}, batch_size: {batch_size})")
    
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        print(f"ERROR: {profile_type} failed for {op_name} (d_model_base: {d_model_base}, batch_size: {batch_size}).")
        print(f"Stderr:\n{process.stderr}")
        print(f"Stdout:\n{process.stdout}") # Ncu often puts useful info here
    else:
        print(f"{profile_type} completed for {op_name}. Output: {final_output_file}") # Print the correct final path

def main():
    parser = argparse.ArgumentParser(description="Offline Expert Kernel Profiler")
    parser.add_argument("--d_model", type=int, default=64, help="Base dimension for model embeddings.")
    parser.add_argument("--profile_base_dir", type=str, default="profiling_data", help="Base directory for profiles.")
    parser.add_argument("--skip_ncu", action="store_true", help="Skip ncu profiling (nsys is sufficient for initial pass).")
    args = parser.parse_args()

    # Define operation types corresponding to your MoE expert's internal structure
    # 'dequant_unpack_op' is added to simulate and profile the dequantization overhead
    op_types_to_profile = ["linear_fc1", "relu", "linear_fc2", "dequant_unpack_op"]

    # Define various input token counts (batch sizes) to profile
    input_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] # Wider range for better cost model

    nsys_dir = os.path.join(args.profile_base_dir, "nsys_traces")
    ncu_dir = os.path.join(args.profile_base_dir, "ncu_reports")
    os.makedirs(nsys_dir, exist_ok=True)
    os.makedirs(ncu_dir, exist_ok=True)

    # Base commands for profilers
    nsys_cmd_base = ["nsys", "profile", "--force-overwrite", "--export=sqlite", "--stats=true", "--trace-gpu-metrics=true"]
    ncu_cmd_base = ["ncu", "--set", "full", "--target-processes", "all"] # --target-processes all is important for python subprocesses

    print(f"Starting kernel profiling with d_model_base={args.d_model}...")

    for op_name in op_types_to_profile:
        for batch_size in input_batch_sizes:
            # Create a unique temporary file path for the script
            # Ensure proper cleanup using try-finally
            temp_script_path = None # Initialize to None
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp_script_file:
                    temp_script_path = tmp_script_file.name
                
                generate_isolated_script(op_name, args.d_model, batch_size, temp_script_path)
                
                # The output path for profilers is constructed here
                profile_filename_base = f"{op_name}_d{args.d_model}_b{batch_size}"
                
                # Run Nsys
                nsys_output_path = os.path.join(nsys_dir, profile_filename_base)
                run_profiler(nsys_cmd_base, temp_script_path, nsys_output_path, "nsys", op_name, args.d_model, batch_size)
                
                # Run Ncu (optional)
                if not args.skip_ncu:
                    ncu_output_path = os.path.join(ncu_dir, profile_filename_base)
                    run_profiler(ncu_cmd_base, temp_script_path, ncu_output_path, "ncu", op_name, args.d_model, batch_size)
                
                # Small delay to prevent profiler conflicts or resource exhaustion
                time.sleep(0.1) 
            finally:
                # Always attempt to clean up the temporary script
                if temp_script_path and os.path.exists(temp_script_path):
                    os.remove(temp_script_path)

    print("\n--- Iteration 1: Offline Kernel Profiling Complete ---")
    print(f"Raw nsys traces in: {nsys_dir}")
    print(f"Raw ncu reports (if enabled) in: {ncu_dir}")
    print("\nNext: Inspect these files and work on `parse_profiler_output.py` in Iteration 2.")

if __name__ == "__main__":
    main()