# scripts/run_experiment.py
import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import time
from moe_models import MoETransformerBlock, compute_energy_loss
from kernelcostmodel import KernelCostModel
# Ensure src directory is in PYTHONPATH for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import all necessary components
from moe_models import MoETransformerBlock, compute_energy_loss, KernelCostModel
from routers import RoutingStrategy, GpuSystemMonitor
from data_utils import DataLoaderManager
from metrics_logger import MetricsLogger
from routers import Gpu


def run_experiment_phase(args, model: MoETransformerBlock, dataloader_manager: DataLoaderManager,
                         gpu_monitor: GpuSystemMonitor, metrics_logger: MetricsLogger,
                         current_phase_strategies: list, expert_type: str,
                         kernel_cost_model: KernelCostModel, epoch_offset: int = 0):
    """
    Runs a single phase of the experiment (e.g., Baseline, TTHA).
    """
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Determine which workloads to use for this phase
    workload_types = ["standard", "high_complexity", "small_batch", "large_batch"]

    print(f"\n--- Starting Experiment Phase for Expert Type: {expert_type} ---")
    
    # Cache baseline latency for TTHA if running non-baseline strategy later
    # This should ideally be determined from a short run of the 'baseline' strategy with a typical workload
    if model.moe_layer.router.strategy != RoutingStrategy.BASELINE:
        # If not already set, use a dummy or compute a quick baseline
        if getattr(model.moe_layer.router, 'base_latency_for_penalty', 0.0) == 0.0:
            print("Warning: base_latency_for_penalty not set. Running a quick baseline to estimate...")
            temp_model = MoETransformerBlock(
                args.d_model, args.num_experts, args.top_k,
                kernel_cost_model, gpu_monitor,
                routing_strategy=RoutingStrategy.BASELINE.value,
                expert_type=expert_type,
                quantization_bits=args.quantization_bits
            ).to(device)
            temp_dataloader = dataloader_manager.get_workload("standard", args.batch_size, args.num_samples_per_workload // 10)
            avg_latency = 0.0
            num_batches = 0
            for x, _ in temp_dataloader:
                x = x.to(device)
                start_time = time.perf_counter()
                _, _, _ = temp_model(x)
                avg_latency += (time.perf_counter() - start_time) * 1000.0
                num_batches += 1
            model.moe_layer.router.base_latency_for_penalty = avg_latency / num_batches if num_batches > 0 else 50.0
            print(f"Estimated base_latency_for_penalty: {model.moe_layer.router.base_latency_for_penalty:.2f} ms")


    for strategy_enum in current_phase_strategies:
        strategy_name = strategy_enum.value
        print(f"\nRunning with strategy: {strategy_name}")
        model.moe_layer.router.strategy = strategy_enum # Set the router's strategy

        for workload_type in workload_types:
            print(f"  Running workload: {workload_type}")
            dataloader = dataloader_manager.get_workload(workload_type, args.batch_size, args.num_samples_per_workload)
            
            # Reset expert cumulative timings for each new strategy/workload run
            for expert in model.moe_layer.experts:
                if hasattr(expert, 'expert_cumulative_timings_ms'):
                    expert.expert_cumulative_timings_ms = {} # Should be on MoELayer, not expert itself
            model.moe_layer.expert_cumulative_timings_ms = {} # Reset MoE Layer's cumulative timings

            # Torch profiler (for detailed trace analysis of a subset of batches)
            profile_name = f"{expert_type}_{workload_type}_{strategy_name}"
            # Profile only a few batches at the start of each run
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1), # Profile one repetition
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.profile_dir, profile_name)),
                record_shapes=True,
                with_stack=True
            ) as prof:
                for epoch in range(epoch_offset, epoch_offset + args.epochs):
                    for batch_idx, (x, y) in enumerate(dataloader):
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()

                        batch_start_time = time.perf_counter()
                        output, aux_loss, metrics = model(x) # num_tokens is automatically x.shape[0]
                        batch_end_time = time.perf_counter()

                        inference_latency_ms = (batch_end_time - batch_start_time) * 1000.0
                        throughput_qps = x.size(0) / (batch_end_time - batch_start_time)
                        throughput_tokens_per_sec = x.size(0) * x.size(1) / (batch_end_time - batch_start_time) # Assuming d_model is token size

                        task_loss = criterion(output, y)
                        selected_indices = metrics["top_k_indices"]
                        top_k_probs = metrics["top_k_probs"] # Make sure this is captured
                        
                        # Calculate energy loss using the KernelCostModel
                        energy_loss = compute_energy_loss(selected_indices, top_k_probs, kernel_cost_model)
                        loss = task_loss + energy_loss + aux_loss # Total loss for optimization

                        loss.backward()
                        optimizer.step()

                        # --- TTHA Update Step (if applicable) ---
                        if strategy_enum == RoutingStrategy.KERNEL_AWARE_TTHA:
                            observed_metrics = gpu_monitor.get_current_stats()
                            observed_metrics["inference_latency_ms"] = inference_latency_ms
                            observed_metrics["throughput_tokens_per_sec"] = throughput_tokens_per_sec # For throughput bonus
                            
                            ttha_loss_components = model.moe_layer.router.update_ttha(
                                observed_metrics,
                                target_power=args.ttha_target_power,
                                target_temp=args.ttha_target_temp,
                                latency_penalty_weight=args.ttha_latency_penalty_weight
                            )
                        else:
                            ttha_loss_components = {} # Empty dict if not TTHA

                        # Collect all data for logging
                        gpu_stats = gpu_monitor.get_current_stats()
                        log_data = {
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch,
                            "batch": batch_idx,
                            "workload_type": workload_type,
                            "strategy": strategy_name,
                            "expert_type": expert_type,
                            "loss": loss.item(),
                            "task_loss": task_loss.item(),
                            "aux_loss": aux_loss.item(),
                            "energy_loss": energy_loss.item(),
                            "inference_latency_ms": inference_latency_ms,
                            "throughput_qps": throughput_qps,
                            "gpu_temperature_c": gpu_stats['temperature'],
                            "gpu_power_watt": gpu_stats['power_watt'],
                            "gpu_thermal_state": gpu_stats['thermal_state'],
                            "gpu_utilization_percent": gpu_stats['gpu_utilization_percent'],
                            "memory_utilization_percent": gpu_stats['memory_utilization_percent'],
                            "expert_usage_counts": metrics["expert_usage_current"].tolist(),
                            "expert_batch_timings_ms": metrics["expert_batch_timings_ms"],
                            "expert_cumulative_timings_ms": metrics["expert_cumulative_timings_ms"],
                            "ttha_history": model.moe_layer.router.ttha_history # Pass full history for logging
                        }
                        metrics_logger.log(log_data)

                        if batch_idx % 10 == 0:
                            print(f"[{expert_type}/{strategy_name}/{workload_type}] Epoch {epoch+1} Batch {batch_idx}: "
                                  f"Loss: {loss.item():.4f}, Task: {task_loss.item():.4f}, Energy: {energy_loss.item():.4f}, "
                                  f"Temp: {gpu_stats['temperature']:.1f}°C, Power: {gpu_stats['power_watt']:.1f}W, "
                                  f"Latency: {inference_latency_ms:.2f}ms")
                            if ttha_loss_components:
                                print(f"  TTHA Losses: Total={ttha_loss_components['total_loss']:.4f}, "
                                      f"Power={ttha_loss_components['power_loss']:.4f}, "
                                      f"Temp={ttha_loss_components['temp_loss']:.4f}, "
                                      f"Latency={ttha_loss_components['latency_penalty']:.4f}")

                        # Step the profiler schedule
                        prof.step()
                        # Exit profiler early to avoid huge trace files
                        if prof.step_num >= (1 + 1 + 2) * 1 : # One repetition of wait, warmup, active
                            if batch_idx > 0 : # Only break if at least one actual batch was processed
                                prof.stop() # Stop profiling early
                                # print(f"Profiler stopped early for {profile_name}")


    print(f"--- Experiment Phase Complete for Expert Type: {expert_type}, Strategy: {strategy_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MoE Routing Experiment")
    parser.add_argument("--d_model", type=int, default=64, help="Dimension of model embeddings.")
    parser.add_argument("--num_experts", type=int, default=8, help="Total number of experts.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to route to.")
    parser.add_argument("--batch_size", type=int, default=32, help="Base batch size for inference.")
    parser.add_argument("--num_samples_per_workload", type=int, default=512, help="Number of samples per workload dataset.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to run for each strategy/workload.")
    parser.add_argument("--profile_dir", type=str, default="experiment_logs/tb_logs", help="Directory for TensorBoard profiles.")
    parser.add_argument("--log_file", type=str, default="experiment_logs/metrics.csv", help="CSV file to log metrics.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to monitor.")
    
    # Experiment control arguments
    parser.add_argument("--expert_type", type=str, default="quantized", choices=["simple", "quantized"],
                        help="Type of expert to use: 'simple' or 'quantized'.")
    parser.add_argument("--quantization_bits", type=int, default=4, choices=[2, 4],
                        help="Number of bits for quantization if expert_type is 'quantized'.")
    parser.add_argument("--kernel_cost_model_json", type=str, default="kernel_cost_models/kernel_cost_model_d64.json",
                        help="Path to the JSON file containing kernel cost model data.")

    # TTHA specific arguments
    parser.add_argument("--ttha_target_power", type=float, default=150.0,
                        help="Target GPU power in Watts for TTHA loss.")
    parser.add_argument("--ttha_target_temp", type=float, default=70.0,
                        help="Target GPU temperature in Celsius for TTHA loss.")
    parser.add_argument("--ttha_latency_penalty_weight", type=float, default=0.1,
                        help="Weight for latency penalty in TTHA loss.")
    
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize KernelCostModel first, so it can be passed to others
    kernel_cost_model = KernelCostModel(data_path=args.kernel_cost_model_json)

    # Initialize GpuSystemMonitor
    gpu_monitor = GpuSystemMonitor(device_id=args.device_id)

    # Initialize Data Manager and Metrics Logger
    dataloader_manager = DataLoaderManager(d_model=args.d_model)
    metrics_logger = MetricsLogger(args.log_file)

    # --- Phase 2: Baseline Characterization ---
    print("\n--- Starting Phase 2: Baseline Characterization ---")
    
    # Initialize the model for baseline runs
    model_baseline = MoETransformerBlock(
        args.d_model, args.num_experts, args.top_k,
        kernel_cost_model, gpu_monitor,
        routing_strategy=RoutingStrategy.BASELINE.value, # Explicitly set baseline
        expert_type=args.expert_type,
        quantization_bits=args.quantization_bits
    ).to(device)

    # Run Baseline and Static Optimal
    run_experiment_phase(args, model_baseline, dataloader_manager, gpu_monitor, metrics_logger,
                         [RoutingStrategy.BASELINE, RoutingStrategy.STATIC_OPTIMAL],
                         args.expert_type, kernel_cost_model, epoch_offset=0)

    # --- Phase 3: Real-time Kernel-Aware TTHA ---
    print("\n--- Starting Phase 3: Real-time Kernel-Aware TTHA ---")

    # Initialize a new model instance for TTHA to ensure clean state and TTHA adapter is initialized
    model_ttha = MoETransformerBlock(
        args.d_model, args.num_experts, args.top_k,
        kernel_cost_model, gpu_monitor,
        routing_strategy=RoutingStrategy.KERNEL_AWARE_TTHA.value, # Explicitly set TTHA
        expert_type=args.expert_type,
        quantization_bits=args.quantization_bits
    ).to(device)
    
    # Set the base latency for penalty, typically from baseline performance
    # This should be estimated from the collected baseline data or a dedicated run
    # For now, let's use a dummy or a value from prior runs:
    # A more robust way: load baseline stats from CSV, compute avg latency, and set it.
    model_ttha.moe_layer.router.base_latency_for_penalty = 50.0 # Example default value in ms
    print(f"Set TTHA router's base_latency_for_penalty to {model_ttha.moe_layer.router.base_latency_for_penalty:.2f} ms")


    run_experiment_phase(args, model_ttha, dataloader_manager, gpu_monitor, metrics_logger,
                         [RoutingStrategy.KERNEL_AWARE_TTHA], # Only run TTHA strategy here
                         args.expert_type, kernel_cost_model, epoch_offset=args.epochs) # Offset epochs for logging

    print("\n--- All Experiment Phases Complete. Data collected in metrics.csv ---")
    print(f"TensorBoard logs for detailed profiling are in: {args.profile_dir}")

    # Cleanup monitor threads
    gpu_monitor.stop()