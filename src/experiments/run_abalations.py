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

def _compute_energy_loss_for_logging(expert_indices: torch.Tensor, expert_probs: torch.Tensor, 
                                     kernel_cost_model: KernelCostModel, moe_config: MoEConfig,
                                     num_tokens_in_batch: int) -> torch.Tensor:
    """
    Computes a weighted energy loss for external logging.
    """
    if kernel_cost_model is None:
        return torch.tensor(0.0, device=expert_indices.device)

    total_predicted_energy = torch.tensor(0.0, device=expert_indices.device)
    expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
    if moe_config.expert_type == "quantized":
        expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 

    avg_tokens_per_selected_expert = num_tokens_in_batch / moe_config.top_k 
    
    for expert_id in range(moe_config.num_experts):
        expert_energy_for_logging = 0.0
        for op_name in expert_ops_for_cost:
            cost = kernel_cost_model.get_cost(op_name, int(avg_tokens_per_selected_expert))
            expert_energy_for_logging += cost.get("energy_joules", 0.0)
        total_predicted_energy += expert_energy_for_logging 

    return total_predicted_energy * 0.001

def run_experiment_phase(args, moe_config: MoEConfig, model: MoETransformerBlock, dataloader_manager: DataLoaderManager,
                         gpu_monitor: GpuSystemMonitor, metrics_logger: MetricsLogger,
                         current_phase_strategies: List[RoutingStrategy],
                         kernel_cost_model: KernelCostModel, epoch_offset: int = 0):
    """
    Runs a single phase of the experiment for ablations and advanced strategies.
    """
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss() 

    # For ablations, we might define specific workloads
    workload_types = ["standard", "high_complexity", "small_batch", "large_batch"]
    
    print(f"\n--- Starting Experiment Phase for Expert Type: {moe_config.expert_type} ---")
    
    # Estimate base latency for TTHA penalties (essential)
    if getattr(model.moe_layer.adaptive_router, 'base_latency_for_penalty', 0.0) == 0.0:
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
            with torch.no_grad():
                _, _ = temp_model(x, use_adaptive_routing=False) 
            avg_latency += (time.perf_counter() - start_time) * 1000.0
            num_batches += 1
        model.moe_layer.adaptive_router.base_latency_for_penalty = avg_latency / num_batches if num_batches > 0 else 50.0
        print(f"Estimated base_latency_for_penalty: {model.moe_layer.adaptive_router.base_latency_for_penalty:.2f} ms")


    for strategy_enum in current_phase_strategies:
        strategy_name = strategy_enum.value
        print(f"\nRunning with strategy: {strategy_name}")
        model.moe_layer.adaptive_router.strategy = strategy_enum # Set the router's strategy
        
        # Set TTHA targets for the router (if strategy uses them)
        model.moe_layer.adaptive_router.update_ttha( # Call once to set targets for logger
            observed_metrics=gpu_monitor.get_current_stats(), # Dummy for init
            target_power=args.ttha_target_power,
            target_temp=args.ttha_target_temp,
            target_latency=model.moe_layer.adaptive_router.base_latency_for_penalty,
            target_memory_util=args.ttha_target_memory_util,
            # Pass weights to update_ttha if this is an ablation, 
            # otherwise it uses router's internal objective_weights
            latency_penalty_weight=args.ttha_latency_penalty_weight, # This will be unused by router but kept for update_ttha signature
            memory_penalty_weight=args.ttha_memory_penalty_weight # New arg
        )
        # If specific objective_weights are passed via args, set them (for ablation studies)
        if args.objective_weights:
            weights_dict = dict(item.split('=') for item in args.objective_weights.split(','))
            weights_dict = {k: float(v) for k, v in weights_dict.items()}
            model.moe_layer.adaptive_router.set_objective_weights(**weights_dict)
            print(f"  Router objective weights set to: {model.moe_layer.adaptive_router.objective_weights}")


        for workload_type in workload_types:
            print(f"  Running workload: {workload_type}")
            dataloader = dataloader_manager.get_workload(workload_type, args.batch_size, args.num_samples_per_workload)
            
            model.moe_layer.expert_timings = {} 

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
                        output, metrics = model(x, use_adaptive_routing=True) # Always use adaptive routing for these strategies
                        batch_end_time = time.perf_counter()

                        inference_latency_ms = (batch_end_time - batch_start_time) * 1000.0
                        throughput_qps = x.size(0) / (batch_end_time - batch_start_time)
                        throughput_tokens_per_sec = x.size(0) * moe_config.d_model / (batch_end_time - batch_start_time) 

                        task_loss = criterion(output, y)
                        
                        aux_losses_dict = metrics.get("aux_losses", {})
                        energy_loss = aux_losses_dict.get("energy_loss", torch.tensor(0.0, device=device))
                        load_balance_loss = aux_losses_dict.get("load_balance_loss", torch.tensor(0.0, device=device))
                        router_z_loss = aux_losses_dict.get("router_z_loss", torch.tensor(0.0, device=device))

                        total_loss = task_loss + \
                                     moe_config.load_balance_weight * load_balance_loss + \
                                     moe_config.router_z_loss_weight * router_z_loss + \
                                     0.001 * energy_loss # Energy loss weight
                        
                        total_loss.backward()
                        optimizer.step()

                        # --- TTHA Update Step (crucial for adaptive strategies) ---
                        observed_metrics_for_ttha = gpu_monitor.get_current_stats()
                        observed_metrics_for_ttha["inference_latency_ms"] = inference_latency_ms
                        observed_metrics_for_ttha["throughput_tokens_per_sec"] = throughput_tokens_per_sec
                        
                        ttha_loss_components = model.moe_layer.adaptive_router.update_ttha(
                            observed_metrics_for_ttha,
                            target_power=args.ttha_target_power,
                            target_temp=args.ttha_target_temp,
                            target_latency=model.moe_layer.adaptive_router.base_latency_for_penalty, # Use the estimated base latency
                            target_memory_util=args.ttha_target_memory_util,
                            latency_penalty_weight=args.ttha_latency_penalty_weight, # This will be ignored by router's internal update, but kept for signature
                            memory_penalty_weight=args.ttha_memory_penalty_weight # New arg
                        )

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
                            "aux_loss": load_balance_loss.item(), 
                            "router_z_loss": router_z_loss.item(),
                            "energy_loss": energy_loss.item(),
                            "inference_latency_ms": inference_latency_ms,
                            "throughput_qps": throughput_qps,
                            "throughput_tokens_per_sec": throughput_tokens_per_sec,
                            "gpu_temperature_c": gpu_stats['temperature'],
                            "gpu_power_watt": gpu_stats['power_watt'],
                            "gpu_thermal_state": gpu_stats['thermal_state'],
                            "gpu_utilization_percent": gpu_stats['gpu_utilization_percent'],
                            "memory_utilization_percent": gpu_stats['memory_utilization_percent'],
                            "expert_usage_counts": metrics["expert_usage"].tolist(),
                            "expert_batch_timings_ms": metrics["expert_timings"],
                            "routing_entropy": metrics["routing_entropy"].item(),
                            "ttha_total_loss": ttha_loss_components.get("total_loss",0), 
                            "ttha_power_loss": ttha_loss_components.get("power_loss",0),
                            "ttha_temp_loss": ttha_loss_components.get("temp_loss",0),
                            "ttha_latency_penalty": ttha_loss_components.get("latency_penalty",0),
                            "ttha_memory_penalty": ttha_loss_components.get("memory_penalty",0),
                            "ttha_throughput_bonus": ttha_loss_components.get("throughput_bonus",0),
                            "routing_metadata_biases_base": metrics["routing_metadata"].get("base_cost_biases", []).tolist(),
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
                                      f"Latency={ttha_loss_components.get('latency_penalty',0):.4f}, "
                                      f"Mem={ttha_loss_components.get('memory_penalty',0):.4f}")

                        prof.step()
                        if prof.step_num >= (1 + 1 + 2) * 1 : 
                            if batch_idx > 0 : 
                                prof.stop() 

    print(f"--- Experiment Phase Complete for Expert Type: {moe_config.expert_type}, Strategy: {strategy_name} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NECTAR MoE Routing Experiment - Ablations & Advanced Strategies")
    parser.add_argument("--d_model", type=int, default=4096, help="Dimension of model embeddings.")
    parser.add_argument("--num_experts", type=int, default=8, help="Total number of experts.")
    parser.add_argument("--top_k", type=int, default=2, help="Number of experts to route to.")
    parser.add_argument("--batch_size", type=int, default=32, help="Base batch size for inference (actual tokens per batch).")
    parser.add_argument("--num_samples_per_workload", type=int, default=4096, help="Number of samples per workload dataset.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to run for each strategy/workload.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for model optimizer.")
    parser.add_argument("--profile_dir", type=str, default="experiment_logs/tb_logs_ablations", help="Directory for TensorBoard profiles.")
    parser.add_argument("--log_file", type=str, default="experiment_logs/metrics_ablations.csv", help="CSV file to log metrics.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID to monitor.")
    
    parser.add_argument("--expert_type", type=str, default="swiglu_ffn", choices=["swiglu_ffn", "quantized"],
                        help="Type of expert to use: 'swiglu_ffn' or 'quantized'.")
    parser.add_argument("--quantization_bits", type=int, default=8, choices=[8],
                        help="Number of bits for quantization if expert_type is 'quantized'.")
    parser.add_argument("--kernel_cost_model_json", type=str, default="kernel_cost_models/kernel_cost_model_d4096.json",
                        help="Path to the JSON file containing kernel cost model data.")
    parser.add_argument("--gpu_type", type=str, default="A100", help="Type of GPU for KCM (e.g., A100, H100).")

    # TTHA specific arguments (adjust default values for your targets)
    parser.add_argument("--ttha_target_power", type=float, default=150.0,
                        help="Target GPU power in Watts for TTHA loss.")
    parser.add_argument("--ttha_target_temp", type=float, default=60.0,
                        help="Target GPU temperature in Celsius for TTHA loss.")
    parser.add_argument("--ttha_target_memory_util", type=float, default=0.7,
                        help="Target GPU memory utilization (0-1.0) for TTHA loss.")
    parser.add_argument("--ttha_latency_penalty_weight", type=float, default=0.1, # This will be unused by router's internal update, but kept for signature
                        help="Weight for latency penalty in TTHA loss (for objective_weights).")
    parser.add_argument("--ttha_memory_penalty_weight", type=float, default=0.05, # New arg
                        help="Weight for memory penalty in TTHA loss (for objective_weights).")
    
    # Ablation-specific argument
    parser.add_argument("--objective_weights", type=str, default="",
                        help="Override default objective weights (e.g., 'performance=0.5,energy=0.5').")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    moe_config = MoEConfig(
        d_model=args.d_model,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dropout=0.1, use_bias=False, activation="swiglu", expert_dropout=0.0, use_grouped_gemm=True,
        load_balance_weight=0.01, router_z_loss_weight=0.001, capacity_factor=1.25
    )
    if moe_config.expert_type == "quantized":
        moe_config.quantization_bits = args.quantization_bits

    kernel_cost_model = KernelCostModel(data_path=args.kernel_cost_model_json, gpu_type=args.gpu_type)
    gpu_monitor = GpuSystemMonitor(device_id=args.device_id)
    dataloader_manager = DataLoaderManager(d_model=args.d_model)
    metrics_logger = MetricsLogger(args.log_file)

    print("\n--- Starting Phase 3: Ablations and Advanced Strategies ---")

    model_ablations = MoETransformerBlock(
        moe_config, kernel_cost_model, gpu_monitor
    ).to(device)

    # Strategies for this phase
    # Example: Run KERNEL_AWARE_TTHA, then PREDICTIVE_THERMAL, etc.
    # You can add MULTI_GPU_AWARE or HIERARCHICAL_ADAPTIVE if implemented
    strategies_to_run = [
        RoutingStrategy.KERNEL_AWARE_TTHA,
        # RoutingStrategy.PREDICTIVE_THERMAL, # Uncomment if you want to test this
        # RoutingStrategy.HIERARCHICAL_ADAPTIVE, # Uncomment if you want to test this
    ]

    run_experiment_phase(args, moe_config, model_ablations, dataloader_manager, gpu_monitor, metrics_logger,
                         strategies_to_run, kernel_cost_model, epoch_offset=0)

    print("\n--- Phase 3 Complete. Data collected in metrics_ablations.csv ---")
    print(f"TensorBoard logs for detailed profiling are in: {args.profile_dir}")

    gpu_monitor.stop()