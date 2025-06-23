# python3 expert_kernel_profiler --d_model 4096 --profile_base_dir profiling_data_for_tests --skip_ncu for A100 profiling

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import subprocess
import json
import time
import argparse
import tempfile
import sys

# Define operation templates and their d_model mappings
# d_model_in and d_model_out are functions that take the base d_model and return
# the specific input/output dimensions for that operation.
OPERATION_TEMPLATES = {
    "linear_fc1": { # Represents the first linear layer in a generic expert (not specifically SwiGLU)
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=True).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d * 2}
    },
    "relu": { # A generic ReLU activation
        "code": """
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = F.relu(dummy_input) # Use F.relu for consistent profiling of the operation itself
        """,
        "d_model_mapping": {"d_model_in": lambda d: d * 2, "d_model_out": lambda d: d * 2} # Input/Output d_model are same for ReLU
    },
    "linear_fc2": { # The second linear layer in a generic expert
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=True).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d * 2, "d_model_out": lambda d: d}
    },
    "dequant_unpack_op": {
        "code": """
            # This simulates the unpacking and scaling overhead of a 4-bit quantized weight
            # (out_features, in_features) corresponds to weight dimensions
            # For this op, d_model_out is like out_features, d_model_in is like in_features
            # The input `packed_input` represents the packed quantized weights
            packed_input = torch.randint(-8, 7, ({d_model_out}, {d_model_in} // 2), dtype=torch.int8, device=device)
            scales = torch.randn({d_model_out}, 1, dtype=torch.float16, device=device)

            # --- Simulated dequantization logic (matching moe_models.py's QuantizedExpert) ---
            low_nibbles = (packed_input & 0x0F).to(torch.int8)
            high_nibbles = (packed_input >> 4).to(torch.int8)
            
            # Convert to signed 4-bit values (handle 2's complement for values > 7)
            low_nibbles = torch.where(low_nibbles > 7, low_nibbles - 16, low_nibbles)
            high_nibbles = torch.where(high_nibbles > 7, high_nibbles - 16, high_nibbles)
            
            # Unpack into full float/half precision weights
            unpacked_weights = torch.empty({d_model_out}, {d_model_in}, device=device, dtype=scales.dtype)
            unpacked_weights[:, 0::2] = high_nibbles.float()
            unpacked_weights[:, 1::2] = low_nibbles.float()
            
            _ = unpacked_weights * scales.to(unpacked_weights.dtype) # Apply scales
            # --- End simulated dequantization logic ---
        """,
        # d_model_in here represents the full d_model for the original (unquantized) linear layer's input dim
        # d_model_out is its output dim. This maps to the overall size of the weight being unpacked.
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: int(8 * d / 3)}, # Using SwiGLU hidden dim as an example for d_model_out
    },
    # --- Operations specific to SwiGLU FFN experts ---
    "ffn_gate": {
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=False).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: int(8 * d / 3)}
    },
    "ffn_up": {
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=False).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: int(8 * d / 3)}
    },
    "ffn_down": {
        "code": """
            model = nn.Linear({d_model_in}, {d_model_out}, bias=False).to(device)
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device)
            _ = model(dummy_input)
        """,
        "d_model_mapping": {"d_model_in": lambda d: int(8 * d / 3), "d_model_out": lambda d: d}
    },
    "silu_gelu": { # Profile common activation functions (SiLU for SwiGLU)
        "code": """
            hidden_dim = int(8 * {d_model_base_ph} / 3) # Use d_model_base_ph for this calculation
            dummy_input = torch.randn({batch_size}, hidden_dim, device=device)
            _ = F.silu(dummy_input)
        """,
        # d_model_in and d_model_out for mapping are for the hidden dimension of the activation
        "d_model_mapping": {"d_model_in": lambda d: int(8 * d / 3), "d_model_out": lambda d: int(8 * d / 3)}
    },
    # --- Quantization operations (simplified for profiling) ---
    "quantize_w8a16": { # Simulates a quantization step (e.g., converting FP16 to INT8)
        "code": """
            dummy_input = torch.randn({batch_size}, {d_model_in}, device=device).half()
            _ = dummy_input.to(torch.int8) # Simulate data conversion
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d}
    },
    "dequantize_w8a16": { # Simulates a dequantization step (e.g., converting INT8 back to FP16)
        "code": """
            dummy_input = torch.randint(-128, 127, ({batch_size}, {d_model_in}), dtype=torch.int8, device=device)
            _ = dummy_input.to(torch.float16) # Simulate data conversion back
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d}
    },
    # --- LaCT-specific operations (placeholders for now, requires more complex isolation) ---
    "lact_fw_forward": { # Represents the forward pass of SwiGLUFastWeightNet (fW(Q))
        "code": """
            # Simulate the forward pass of SwiGLUFastWeightNet
            # d_model_in = input_dim (d_model), d_model_out = output_dim (d_model)
            # fast_weight_dim = int(d_model_in * 0.25) # Example ratio from MoEConfig
            
            d_model = {d_model_in}
            fast_weight_dim = int(d_model * 0.25) 

            w1 = torch.randn(fast_weight_dim, d_model, device=device)
            w3 = torch.randn(fast_weight_dim, d_model, device=device)
            w2 = torch.randn(d_model, fast_weight_dim, device=device)

            x = torch.randn({batch_size}, d_model, device=device)

            hidden_gate = F.linear(x, w1)
            hidden_up = F.linear(x, w3)
            activated = F.silu(hidden_gate) * hidden_up
            _ = F.linear(activated, w2)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d}
    },
    "lact_fw_update_loss_grad": { # Represents the gradient computation for fast weights
        "code": """
            # Simulate the gradient computation (backward pass of the loss)
            # This is complex as it involves autograd.grad.
            # For profiling, we need to ensure the graph is built and gradients are computed.
            d_model = {d_model_in}
            fast_weight_dim = int(d_model * 0.25) 

            w1 = torch.randn(fast_weight_dim, d_model, device=device, requires_grad=True)
            w3 = torch.randn(fast_weight_dim, d_model, device=device, requires_grad=True)
            w2 = torch.randn(d_model, fast_weight_dim, device=device, requires_grad=True)

            k = torch.randn({batch_size}, d_model, device=device)
            v = torch.randn({batch_size}, d_model, device=device)

            # Forward pass through fast weight net with k
            gate_before_act = F.linear(k, w1)
            hidden_before_gate = F.linear(k, w3)
            hidden = F.silu(gate_before_act) * hidden_before_gate
            fW_k = F.linear(hidden, w2)

            # Negative dot product loss
            chunk_loss = (-(fW_k * v).sum(dim=-1)).mean()
            
            # Compute gradients
            _ = torch.autograd.grad(chunk_loss, [w1, w2, w3], retain_graph=False)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d}
    },
    "lact_fw_optimizer_step": { # Represents the optimizer step and normalization
        "code": """
            # Simulate optimizer step and normalization of fast weights
            d_model = {d_model_in}
            fast_weight_dim = int(d_model * 0.25) 

            w1 = torch.randn(fast_weight_dim, d_model, device=device, requires_grad=True)
            w2 = torch.randn(d_model, fast_weight_dim, device=device, requires_grad=True)
            w3 = torch.randn(fast_weight_dim, d_model, device=device, requires_grad=True)

            # Assign dummy gradients for the optimizer to work
            w1.grad = torch.randn_like(w1)
            w2.grad = torch.randn_like(w2)
            w3.grad = torch.randn_like(w3)

            optimizer = torch.optim.AdamW([w1, w2, w3], lr=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

            optimizer.zero_grad() # Just to be safe, though dummy grads are assigned
            optimizer.step()
            scheduler.step()

            # Simulate L2 normalization (without_grad, as in LaCTMoEExpert)
            with torch.no_grad():
                if w1.norm(p=2, dim=0).sum() > 0: # Sum the norms to get a scalar boolean
                    w1.data = F.normalize(w1.data, p=2, dim=0)
                if w2.norm(p=2, dim=0).sum() > 0:
                    w2.data = F.normalize(w2.data, p=2, dim=0)
                if w3.norm(p=2, dim=0).sum() > 0:
                    w3.data = F.normalize(w3.data, p=2, dim=0)
        """,
        "d_model_mapping": {"d_model_in": lambda d: d, "d_model_out": lambda d: d}
    },
}

def generate_isolated_script(op_name: str, d_model_base: int, batch_size: int, temp_script_path: str):
    """
    Generates a temporary Python script to run an isolated PyTorch operation.
    """
    template_info = OPERATION_TEMPLATES[op_name]
    
    # Map d_model_base to specific input/output d_model for the operation
    # Use 'd' as the lambda variable name for clarity with original mapping
    d_model_in = template_info["d_model_mapping"]["d_model_in"](d_model_base)
    d_model_out = template_info["d_model_mapping"]["d_model_out"](d_model_base)

    # Special handling for silu_gelu's input size
    if op_name == "silu_gelu":
        # hidden_dim is calculated within the template code, use d_model_base_ph
        # to ensure it's calculated based on the correct d_model for the profile run
        op_code = template_info["code"].format(
            batch_size=batch_size,
            d_model_base_ph=d_model_base # Pass the base d_model for internal calc
        )
    else:
        op_code = template_info["code"].format(
            d_model_in=d_model_in,
            d_model_out=d_model_out,
            batch_size=batch_size
        )

    script_content = f"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# These placeholders will be replaced by actual values by the profiler script
op_name = "{op_name}"
d_model_base = {d_model_base}
batch_size = {batch_size}

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

# Print timing and metadata for parser to extract
print(f"Operation: {{op_name}}, D_model_base: {{d_model_base}}, Batch_size: {{batch_size}}, Latency: {{start_event.elapsed_time(end_event):.4f}} ms")
"""
    with open(temp_script_path, "w") as f:
        f.write(script_content)
    # print(f"Generated temporary script: {temp_script_path}")


def run_profiler(profiler_cmd_base: list, python_script_path: str, output_path: str,
                 profile_type: str, op_name: str, d_model_base: int, batch_size: int):
    """
    Executes a profiler command on the generated Python script.
    """
    profile_filename_base = f"{op_name}_d{d_model_base}_b{batch_size}"

    cmd = profiler_cmd_base + [
        f"--output={output_path}",
        "python", python_script_path
    ]
    
    # Check if the output file already exists. If yes, skip to save time during debugging.
    final_output_file = os.path.join(os.path.dirname(output_path), profile_filename_base + (".qdrep" if profile_type == "nsys" else ".ncu-rep"))

    if os.path.exists(final_output_file):
        print(f"Skipping {profile_type} for {profile_filename_base}: Output file already exists.")
        return

    print(f"Running {profile_type} for {op_name} (d_model_base: {d_model_base}, batch_size: {batch_size})")
    
    process = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if process.returncode != 0:
        print(f"ERROR: {profile_type} failed for {op_name} (d_model_base: {d_model_base}, batch_size: {batch_size}).")
        print(f"Stderr:\n{process.stderr}")
        print(f"Stdout:\n{process.stdout}")
    else:
        print(f"{profile_type} completed for {op_name}. Output: {final_output_file}")

def main():
    parser = argparse.ArgumentParser(description="Offline Expert Kernel Profiler")
    parser.add_argument("--d_model", type=int, default=4096, help="Base dimension for model embeddings.")
    parser.add_argument("--profile_base_dir", type=str, default="profiling_data", help="Base directory for profiles.")
    parser.add_argument("--skip_ncu", action="store_true", help="Skip ncu profiling (nsys is sufficient for initial pass).")
    args = parser.parse_args()

    op_types_to_profile = [
        "linear_fc1", "relu", "linear_fc2", "dequant_unpack_op",
        "ffn_gate", "ffn_up", "ffn_down", "silu_gelu",
        "quantize_w8a16", "dequantize_w8a16",
        "lact_fw_forward", "lact_fw_update_loss_grad", "lact_fw_optimizer_step"
    ]

    # Define various input token counts (batch sizes) to profile
    input_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] # Wider range for better cost model

    nsys_dir = os.path.join(args.profile_base_dir, "nsys_traces")
    ncu_dir = os.path.join(args.profile_base_dir, "ncu_reports")
    os.makedirs(nsys_dir, exist_ok=True)
    os.makedirs(ncu_dir, exist_ok=True)

    # Base commands for profilers
    # REMOVED --trace-gpu-metrics=true because it causes errors with older nsys versions.
    # For full GPU hardware counter metrics, a newer Nsight Systems (2023.1+) is required.
    nsys_cmd_base = ["nsys", "profile", "--force-overwrite", "true", "--export=sqlite", "--stats=true"]

    ncu_cmd_base = ["ncu", "--set", "full", "--target-processes", "all"]

    print(f"Starting kernel profiling with d_model_base={args.d_model}...")

    for op_name in op_types_to_profile:
        for batch_size in input_batch_sizes:
            temp_script_path = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as tmp_script_file:
                    temp_script_path = tmp_script_file.name
                
                generate_isolated_script(op_name, args.d_model, batch_size, temp_script_path)
                
                profile_filename_base = f"{op_name}_d{args.d_model}_b{batch_size}"
                
                # Run Nsys
                nsys_output_path = os.path.join(nsys_dir, profile_filename_base)
                run_profiler(nsys_cmd_base, temp_script_path, nsys_output_path, "nsys", op_name, args.d_model, batch_size)
                
                # Run Ncu (optional)
                if not args.skip_ncu:
                    ncu_output_path = os.path.join(ncu_dir, profile_filename_base)
                    run_profiler(ncu_cmd_base, temp_script_path, ncu_output_path, "ncu", op_name, args.d_model, batch_size)
                
                time.sleep(0.1) 
            finally:
                if temp_script_path and os.path.exists(temp_script_path):
                    os.remove(temp_script_path)

    print("\n--- Iteration 1: Offline Kernel Profiling Complete ---")
    print(f"Raw nsys traces in: {nsys_dir}")
    print(f"Raw ncu reports (if enabled) in: {ncu_dir}")
    print("\nNext: Run `python src/experiments/parse_profiler_output.py` to process these profiles into a KCM.")

if __name__ == "__main__":
    main()