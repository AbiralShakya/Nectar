import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
import time
from typing import Dict, Tuple, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.moe_models import MoEConfig, MoETransformerBlock, OptimizedMoELayer, SwiGLUExpert, OptimizedQuantizedExpert
from routers import RoutingStrategy, GpuSystemMonitor, AdaptiveRouter
from src.kernelcostmodel import KernelCostModel
from data_utils import DataLoaderManager
from metrics_logger import MetricsLogger

# Assuming compute_energy_loss is now part of OptimizedMoELayer or MoETransformerBlock's forward,
# or a utility function taking new args. If it's a standalone func, it needs access to KCM.
# For simplicity, let's keep it defined here for the energy_loss calculation in run_experiment_phase
# but note that OptimizedMoELayer also computes and logs this internally now.
def _compute_energy_loss_for_logging(expert_indices: torch.Tensor, expert_probs: torch.Tensor, 
                                     kernel_cost_model: KernelCostModel, moe_config: MoEConfig,
                                     num_tokens_in_batch: int) -> torch.Tensor:
    """
    Computes a weighted energy loss based on activated experts and their routing probabilities,
    using the KernelCostModel for predicted costs. This is for external logging.
    """
    if kernel_cost_model is None:
        return torch.tensor(0.0, device=expert_indices.device)

    total_predicted_energy = torch.tensor(0.0, device=expert_indices.device)
    
    # Operations that make up an expert's forward pass
    expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
    if moe_config.expert_type == "quantized":
        expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 

    for expert_id in range(moe_config.num_experts):
        # Calculate approximate number of tokens routed to this expert for KCM lookup
        # This is a rough estimation for profiling cost aggregation
        avg_tokens_per_selected_expert = num_tokens_in_batch / moe_config.top_k 
        
        # Get energy costs for operations in this expert for the estimated batch size
        expert_energy_for_logging = 0.0
        for op_name in expert_ops_for_cost:
            cost = kernel_cost_model.get_cost(op_name, int(avg_tokens_per_selected_expert))
            expert_energy_for_logging += cost.get("energy_joules", 0.0)
        
        # Sum over experts weighted by their overall selection probability (router_probs if available, or just uniform)
        # Here we'll just sum the potential energy if activated, for a simpler aggregated loss
        total_predicted_energy += expert_energy_for_logging 

    return total_predicted_energy * 0.001 # A small scaling factor for the loss weight

def run_experiment_phase(args, moe_config: MoEConfig, model: MoETransformerBlock, dataloader_manager: DataLoaderManager,
                         gpu_monitor: GpuSystemMonitor, metrics_logger: MetricsLogger,
                         current_phase_strategies: List[RoutingStrategy],
                         kernel_cost_model: KernelCostModel, epoch_offset: int = 0):
    """
    Runs a single phase of the experiment for baseline and static_optimal strategies.
    """
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() # Dummy task loss

    workload_types = ["standard", "high_complexity", "small_batch", "large_batch"]

    print(f"\n--- Starting Experiment Phase for Expert Type: {moe_config.expert_type} ---")
    
    # Estimate base latency for TTHA penalties (if needed for later phases)
    # This part of the logic is more critical for dynamic_adaptation script, but good to keep structure
    if moe_config.d_model == 64: # Fallback for old default config in comments
        model.moe_layer.adaptive_router.base_latency_for_penalty = 50.0 
    else: # Estimate for larger models (rough estimation)
        model.moe_layer.adaptive_router.base_latency_for_penalty = 1.0 # 1ms base per token for large models
        # A more robust way: Run a short baseline to estimate
        print("Estimating base latency for TTHA penalty...")
        temp_model = MoETransformerBlock(
            moe_config, kernel_cost_model, gpu_monitor
        ).to(device)
        temp_model.moe_layer.adaptive_router.strategy = RoutingStrategy.BASELINE # Set to baseline for estimation
        temp_dataloader = dataloader_manager.get_workload("standard", args.batch_size, args.num_samples_per_workload // 10)
        avg_latency = 0.0
        num_batches = 0
        for x, _ in temp_dataloader:
            x = x.to(device)
            start_time = time.perf_counter()
            with torch.no_grad(): # No need to track gradients for estimation
                _, _ = temp_model(x, use_adaptive_routing=False) # Ensure standard routing for baseline estimate
            avg_latency += (time.perf_counter() - start_time) * 1000.0
            num_batches += 1
        model.moe_layer.adaptive_router.base_latency_for_penalty = avg_latency / num_batches if num_batches > 0 else 50.0
        print(f"Estimated base_latency_for_penalty: {model.moe_layer.adaptive_router.base_latency_for_penalty:.2f} ms")


    for strategy_enum in current_phase_strategies:
        strategy_name = strategy_enum.value
        print(f"\nRunning with strategy: {strategy_name}")
        model.moe_layer.adaptive_router.strategy = strategy_enum # Set the router's strategy

        for workload_type in workload_types:
            print(f"  Running workload: {workload_type}")
            dataloader = dataloader_manager.get_workload(workload_type, args.batch_size, args.num_samples_per_workload)
            
            # Reset expert cumulative timings for this strategy/workload run
            model.moe_layer.expert_timings = {} 

            # Torch profiler (for detailed trace analysis of a subset of batches)
            profile_name = f"{moe_config.expert_type}_{workload_type}_{strategy_name}"
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.profile_dir, profile_name)),
                record_shapes=True,
                with_stack=True
            ) as prof:
                for epoch in range(epoch_offset, epoch_offset + args.epochs):
                    for batch_idx, (x, y) in enumerate(dataloader):
                        x, y = x.to(device), y.to(device)
                        optimizer.zero_grad()

                        batch_start_time = time.perf_counter()
                        # Pass num_tokens_in_batch for KCM lookup, and control adaptive routing
                        output, metrics = model(x, use_adaptive_routing=(strategy_enum != RoutingStrategy.BASELINE))
                        batch_end_time = time.perf_counter()

                        inference_latency_ms = (batch_end_time - batch_start_time) * 1000.0
                        throughput_qps = x.size(0) / (batch_end_time - batch_start_time)
                        throughput_tokens_per_sec = x.size(0) * moe_config.d_model / (batch_end_time - batch_start_time) 

                        task_loss = criterion(output, y)
                        
                        # Energy loss is now calculated within OptimizedMoELayer and returned in metrics
                        # We need to extract it and other auxiliary losses
                        aux_losses_dict = metrics.get("aux_losses", {})
                        energy_loss = aux_losses_dict.get("energy_loss", torch.tensor(0.0, device=device))
                        load_balance_loss = aux_losses_dict.get("load_balance_loss", torch.tensor(0.0, device=device))
                        router_z_loss = aux_losses_dict.get("router_z_loss", torch.tensor(0.0, device=device))

                        # Combine all losses for optimization
                        total_loss = task_loss + \
                                     moe_config.load_balance_weight * load_balance_loss + \
                                     moe_config.router_z_loss_weight * router_z_loss + \
                                     0.001 * energy_loss # Weight for energy_loss (tune this!)
                        
                        total_loss.backward()
                        optimizer.step()

                        # TTHA Update Step is only for KERNEL_AWARE_TTHA, not baselines
                        ttha_loss_components = {} # Baselines don't have TTHA updates

                        # Collect all data for logging
                        gpu_stats = gpu_monitor.get_current_stats()
                        log_data = {
                            "timestamp": datetime.now().isoformat(),
                            "epoch": epoch,
                            "batch": batch_idx,
                            "workload_type": workload_type,
                            "strategy": strategy_name,
                            "expert_type": moe_config.expert_type,
                            "loss": total_loss.item(),
                            "task_loss": task_loss.item(),
                            "aux_loss": load_balance_loss.item(), # Using load_balance_loss as primary aux for logging
                            "router_z_loss": router_z_loss.item(),
                            "energy_loss": energy_loss.item(),
                            "inference_latency_ms": inference_latency_ms,
                            "throughput_qps": throughput_qps,
                            "throughput_tokens_per_sec": throughput_tokens_per_sec, # Added
                            "gpu_temperature_c": gpu_stats['temperature'],
                            "gpu_power_watt": gpu_stats['power_watt'],
                            "gpu_thermal_state": gpu_stats['thermal_state'],
                            "gpu_utilization_percent": gpu_stats['gpu_utilization_percent'],
                            "memory_utilization_percent": gpu_stats['memory_utilization_percent'],
                            "expert_usage_counts": metrics["expert_usage"].tolist(), # Corrected key
                            "expert_batch_timings_ms": metrics["expert_timings"], # Corrected key
                            "routing_entropy": metrics["routing_entropy"].item(), # Added
                            "routing_metadata_biases_base": metrics["routing_metadata"].get("base_cost_biases", []).tolist(), # Convert np array to list
                            "routing_metadata_biases_final": metrics["routing_metadata"].get("final_biases", []).tolist(),
                        }
                        metrics_logger.log(log_data)

                        if batch_idx % 10 == 0:
                            print(f"[{moe_config.expert_type}/{strategy_name}/{workload_type}] Epoch {epoch+1} Batch {batch_idx}: "
                                  f"Loss: {total_loss.item():.4f}, Task: {task_loss.item():.4f}, Energy: {energy_loss.item():.4f}, "
                                  f"Temp: {gpu_stats['temperature']:.1f}°C, Power: {gpu_stats['power_watt']:.1f}W, "
                                  f"Latency: {inference_latency_ms:.2f}ms, Thruput: {throughput_tokens_per_sec:.2f}tok/s")
                            if ttha_loss_components:
                                print(f"  TTHA Losses: Total={ttha_loss_components.get('total_loss',0):.4f}, "
                                      f"Power={ttha_loss_components.get('power_loss',0):.4f}, "
                                      f"Temp={ttha_loss_components.get('temp_loss',0):.4f}, "
                                      f"Latency={ttha_loss_components.get('latency_penalty',0):.4f}")

                        prof.step()
                        if prof.step_num >= (1 + 1 + 2) * 1 : # Stop after one repetition of profile schedule
                            if batch_idx > 0 : 
                                prof.stop() 


    print(f"--- Experiment Phase Complete for Expert Type: {moe_config.expert_type}, Strategy: {strategy_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NECTAR MoE Routing Experiment - Baselines")
    parser.add_argument("--d_model", type=int, default=4096, help="Dimension of model embeddings.")
    parser.add_argument("--num_experts", type=int, default=8, help="Total number of experts.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to route to.")
    parser.add_argument("--batch_size", type=int, default=32, help="Base batch size for inference (actual tokens per batch).")
    parser.add_argument("--num_samples_per_workload", type=int, default=4096, help="Number of samples per workload dataset.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run for each strategy/workload.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for model optimizer.")
    parser.add_argument("--profile_dir", type=str, default="experiment_logs/tb_logs_baselines", help="Directory for TensorBoard profiles.")
    parser.add_argument("--log_file", type=str, default="experiment_logs/metrics_baselines.csv", help="CSV file to log metrics.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to monitor.")
    
    parser.add_argument("--expert_type", type=str, default="swiglu_ffn", choices=["swiglu_ffn", "quantized"],
                        help="Type of expert to use: 'swiglu_ffn' or 'quantized'.")
    parser.add_argument("--quantization_bits", type=int, default=8, choices=[8],
                        help="Number of bits for quantization if expert_type is 'quantized'.")
    parser.add_argument("--kernel_cost_model_json", type=str, default="kernel_cost_models/kernel_cost_model_d4096.json",
                        help="Path to the JSON file containing kernel cost model data.")
    parser.add_argument("--gpu_type", type=str, default="A100", help="Type of GPU for KCM (e.g., A100, H100).")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize MoEConfig
    moe_config = MoEConfig(
        d_model=args.d_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dropout=args.dropout, # Need to add dropout arg to parser if using
        expert_type=args.expert_type,
        use_grouped_gemm=True,
        load_balance_weight=0.01, # Default. Can add to parser if needed
        router_z_loss_weight=0.001, # Default. Can add to parser if needed
        # Pass quantization_bits to config if using quantized experts
        # This will need to be handled if MoEConfig doesn't directly take it,
        # or if OptimizedQuantizedExpert expects it directly in __init__.
        # For now, OptimizedQuantizedExpert takes it in its __init__.
    )
    # Correctly pass quantization_bits if expert_type is quantized
    if moe_config.expert_type == "quantized":
        moe_config.quantization_bits = args.quantization_bits


    # Initialize KernelCostModel
    kernel_cost_model = KernelCostModel(data_path=args.kernel_cost_model_json, gpu_type=args.gpu_type)

    # Initialize GpuSystemMonitor
    gpu_monitor = GpuSystemMonitor(device_id=args.device_id)

    # Initialize Data Manager and Metrics Logger
    dataloader_manager = DataLoaderManager(d_model=args.d_model) # d_model used for dummy data generation
    metrics_logger = MetricsLogger(args.log_file)

    # --- Phase 1: Baseline and Static Optimal ---
    print("\n--- Starting Phase 1: Baseline and Static Optimal Characterization ---")
    
    # Create MoE model for this phase
    model_baselines = MoETransformerBlock(
        moe_config, kernel_cost_model, gpu_monitor
    ).to(device)

    run_experiment_phase(args, moe_config, model_baselines, dataloader_manager, gpu_monitor, metrics_logger,
                         [RoutingStrategy.BASELINE, RoutingStrategy.STATIC_OPTIMAL],
                         kernel_cost_model, epoch_offset=0)

    print("\n--- Phase 1 Complete. Data collected in metrics_baselines.csv ---")
    print(f"TensorBoard logs for detailed profiling are in: {args.profile_dir}")

    gpu_monitor.stop() # Ensure monitor thread is stopped