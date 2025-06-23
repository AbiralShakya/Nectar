import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import math

# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from moe_models import MoEConfig, CapacityBasedRouter
from routers import AdaptiveRouter, RoutingStrategy, HardwareMetrics
from kernelcostmodel import KernelCostModel
from monitor import GpuSystemMonitor

def run_test_router():
    print("--- Testing Router Modules ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Common MoEConfig for testing routers
    config = MoEConfig(d_model=128, num_experts=4, top_k=2, capacity_factor=0.5,
                       expert_type="swiglu_ffn") # Use swiglu_ffn for KCM lookups
    dummy_input_tokens = torch.randn(64, config.d_model, device=device) # 64 tokens

    # --- Test CapacityBasedRouter ---
    print("\n--- Testing CapacityBasedRouter ---")
    cap_router = CapacityBasedRouter(config).to(device)
    expert_indices, expert_probs, aux_losses = cap_router(dummy_input_tokens)
    
    print(f"  Expert Indices shape: {expert_indices.shape}, values (first 5): {expert_indices[:5].tolist()}")
    print(f"  Expert Probs shape: {expert_probs.shape}, values (first 5): {expert_probs[:5].tolist()}")
    print(f"  Load Balance Loss: {aux_losses['load_balance_loss'].item():.4f}")
    print(f"  Router Z Loss: {aux_losses['router_z_loss'].item():.4f}")
    print(f"  Expert Usage (top-1 counts): {aux_losses['expert_usage'].tolist()}")
    
    # Test capacity constraint (expecting drops with tight capacity)
    config_tight_cap = MoEConfig(d_model=128, num_experts=4, top_k=2, capacity_factor=0.1).to(device) 
    cap_router_tight = CapacityBasedRouter(config_tight_cap).to(device)
    indices_tight, probs_tight, _ = cap_router_tight(dummy_input_tokens)
    dropped_tokens_count = (indices_tight == -1).sum().item()
    print(f"  Tokens dropped due to tight capacity: {dropped_tokens_count} (expect > 0)")
    assert dropped_tokens_count > 0, "CapacityBasedRouter did not drop tokens with tight capacity."
    print("CapacityBasedRouter passed basic functionality and capacity test.")


    # --- Test AdaptiveRouter ---
    print("\n--- Testing AdaptiveRouter ---")
    kcm = KernelCostModel(data_path="kernel_cost_models/kernel_cost_model_d4096.json", gpu_type="A100") # Ensure exists
    monitor = GpuSystemMonitor(device_id=0) # Will simulate if no pynvml
    
    # Create AdaptiveRouter with KERNEL_AWARE_TTHA strategy
    adaptive_router = AdaptiveRouter(config, kcm, monitor, strategy=RoutingStrategy.KERNEL_AWARE_TTHA).to(device)
    
    # Initialize base_latency_for_penalty (important for TTHA loss)
    adaptive_router.base_latency_for_penalty = 10.0 # Example value
    
    # Simulate a forward pass through AdaptiveRouter
    # Initial gate_logits from CapacityBasedRouter's gate
    initial_gate_logits = cap_router.gate(dummy_input_tokens) 
    
    final_indices, final_probs, routing_info = adaptive_router(initial_gate_logits, dummy_input_tokens.size(0))
    print(f"  AdaptiveRouter final indices shape: {final_indices.shape}")
    print(f"  AdaptiveRouter final probs shape: {final_probs.shape}")
    print(f"  Routing Latency: {routing_info['routing_latency']:.4f}s")
    print(f"  Strategy Used: {routing_info['strategy']}")
    print(f"  System Health: {routing_info['system_health']:.2f}")
    print(f"  Base Cost Biases (first 3): {routing_info['base_cost_biases'][:3].tolist()}")
    print(f"  Final Biases (first 3): {routing_info['final_biases'][:3].tolist()}")
    print(f"  Cache Hit Rate: {routing_info['cache_hit_rate']:.2f}")
    assert not np.array_equal(routing_info['base_cost_biases'], routing_info['final_biases']), \
           "Final biases should be different from base for KERNEL_AWARE_TTHA strategy."
    
    # Test TTHA update (simulated observed metrics)
    print("\n--- Testing TTHA Update ---")
    observed_metrics = monitor.get_current_stats()
    observed_metrics['inference_latency_ms'] = 15.0 # Example higher latency
    observed_metrics['throughput_tokens_per_sec'] = 1000.0
    
    ttha_loss_comps = adaptive_router.update_ttha(
        observed_metrics,
        target_power=100.0,
        target_temp=50.0,
        target_latency=adaptive_router.base_latency_for_penalty,
        target_memory_util=0.5
    )
    print(f"  TTHA Update Loss Components: {ttha_loss_comps}")
    assert 'total_loss' in ttha_loss_comps and not math.isnan(ttha_loss_comps['total_loss']), "TTHA update did not return valid loss."
    
    # Check if TTHA history is populated
    print(f"  TTHA History total_loss count: {len(adaptive_router.ttha_history.get('total_loss', []))}")
    assert len(adaptive_router.ttha_history.get('total_loss', [])) > 0, "TTHA history not populated."

    # Test AdaptiveRouter stats
    router_stats = adaptive_router.get_routing_statistics()
    print(f"  AdaptiveRouter Statistics: {router_stats}")
    
    monitor.stop() # Stop monitor thread
    print("AdaptiveRouter passed basic functionality and TTHA update test.")
    
    print("--- All Router Modules Tests Complete ---")

if __name__ == "__main__":
    run_test_router()