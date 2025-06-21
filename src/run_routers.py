from routers import KernelCostModel, GpuSystemMonitor, create_router_factory
import torch

if __name__ == "__main__":
    # Example configuration for production deployment
    config = {
        'strategy': 'kernel_aware_ttha',
        'objective_weights': {
            'performance': 0.35,
            'energy': 0.35,
            'thermal': 0.25,
            'load_balance': 0.05
        },
        'device_topology': {
            'num_gpus': 4,
            'gpu_memory_gb': 24,
            'interconnect': 'nvlink'
        }
    }
    
    # Initialize components
    kernel_model = KernelCostModel()
    gpu_monitor = GpuSystemMonitor(num_gpus=4)
    
    # Create router
    router_factory = create_router_factory(config)
    router = router_factory(
        num_experts=64,
        top_k=4,
        kernel_cost_model=kernel_model,
        gpu_system_monitor=gpu_monitor
    )
    
    # Example forward pass
    batch_size = 32
    seq_length = 512
    num_tokens = batch_size * seq_length
    num_experts = 64
    
    gate_logits = torch.randn(num_tokens, num_experts)
    
    context = {
        'sequence_length': seq_length,
        'task_type': 'code_generation',
        'urgency': 0.3
    }
    
    # Route tokens
    expert_indices, routing_weights, routing_info = router(
        gate_logits, batch_size, context
    )
    
    print(f"Routing completed in {routing_info['routing_latency']:.4f}s")
    print(f"System health score: {routing_info['system_health']:.3f}")
    print(f"Strategy: {routing_info['strategy']}")
    
    # Simulate hardware feedback and update TTHA
    observed_metrics = {
        'gpu_power_watt': 180.0,
        'gpu_temperature_c': 75.0,
        'inference_latency_ms': 12.5,
        'throughput_tokens_per_sec': 8500.0
    }
    
    loss_components = router.update_ttha(observed_metrics)
    print(f"TTHA update losses: {loss_components}")
    
    # Get statistics
    stats = router.get_routing_statistics()
    print(f"Routing statistics: {stats}")
    
    # Cleanup
    router.cleanup()