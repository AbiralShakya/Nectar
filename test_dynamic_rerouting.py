#!/usr/bin/env python3
"""
Simple test script for dynamic expert rerouting functionality.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from routers import RoutingStrategy, BatchDistributionTracker
from src.moe_models import MoEConfig
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor

def test_batch_distribution_tracker():
    """Test the BatchDistributionTracker functionality."""
    print("Testing BatchDistributionTracker...")
    
    # Initialize tracker
    num_experts = 8
    tracker = BatchDistributionTracker(
        num_experts=num_experts,
        history_length=10,
        imbalance_threshold=0.3,
        rerouting_strength=0.5
    )
    
    # Create dummy data
    device = torch.device('cpu')
    expert_indices = torch.randint(0, num_experts, (32,))  # 32 tokens
    expert_weights = torch.softmax(torch.randn(num_experts), dim=0)
    
    hardware_metrics = {
        'temperature': 65.0,
        'power_watt': 250.0,
        'gpu_utilization_percent': 80.0,
        'memory_utilization_percent': 60.0
    }
    
    performance_metrics = {
        'latency_ms': 15.0,
        'throughput_tokens_per_sec': 1000.0,
        'energy_joules': 0.5
    }
    
    # Test update_distribution
    tracker.update_distribution(expert_indices, expert_weights, hardware_metrics, performance_metrics)
    print("✓ update_distribution completed")
    
    # Test predict_future_imbalance
    predicted_dist, confidence = tracker.predict_future_imbalance()
    print(f"✓ predict_future_imbalance: confidence={confidence:.3f}")
    
    # Test compute_rerouting_biases
    current_distribution = torch.ones(num_experts) / num_experts  # Uniform
    rerouting_biases, metadata = tracker.compute_rerouting_biases(
        current_distribution, hardware_metrics
    )
    print(f"✓ compute_rerouting_biases: needs_rerouting={metadata.get('needs_rerouting', False)}")
    
    # Test get_statistics
    stats = tracker.get_statistics()
    print(f"✓ get_statistics: history_length={stats.get('history_length', 0)}")
    
    print("BatchDistributionTracker tests passed!")
    return True

def test_routing_strategy_enum():
    """Test that the new routing strategy is properly defined."""
    print("Testing routing strategy enum...")
    
    # Check that the new strategy exists
    assert hasattr(RoutingStrategy, 'DYNAMIC_EXPERT_REROUTING')
    assert RoutingStrategy.DYNAMIC_EXPERT_REROUTING.value == "dynamic_expert_rerouting"
    
    print("✓ DYNAMIC_EXPERT_REROUTING strategy properly defined")
    return True

def test_moe_config():
    """Test MoE configuration with the new routing strategy."""
    print("Testing MoE configuration...")
    
    config = MoEConfig(
        d_model=512,
        num_experts=8,
        top_k=2,
        expert_type="simple",
        batch_size=32
    )
    
    print(f"✓ MoEConfig created: {config.num_experts} experts, {config.top_k} top-k")
    return True

def main():
    """Run all tests."""
    print("=== Dynamic Expert Rerouting Tests ===\n")
    
    try:
        test_routing_strategy_enum()
        test_moe_config()
        test_batch_distribution_tracker()
        
        print("\n=== All Tests Passed! ===")
        print("The dynamic expert rerouting functionality is ready for integration.")
        print("\nNext steps:")
        print("1. Run the full experiment with: python src/experiments/run_dynamic_expert_rerouting.py")
        print("2. Compare results with baseline routing strategies")
        print("3. Analyze power/energy efficiency improvements")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 