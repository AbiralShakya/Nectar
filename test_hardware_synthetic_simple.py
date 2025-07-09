#!/usr/bin/env python3
"""
Simple test script for hardware metrics + synthetic stress demonstration.
This script shows how to:
1. Query real hardware energy/temperature (if available)
2. Apply controlled synthetic stress to test routing logic
3. Demonstrate that the router responds to energy/temperature imbalances
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Dict, Any, List
from dataclasses import dataclass

# Try to import pynvml for real hardware metrics
try:
    import pynvml
    pynvml.nvmlInit()
    HARDWARE_AVAILABLE = True
    print("✓ pynvml available - using real hardware metrics")
except ImportError:
    HARDWARE_AVAILABLE = False
    print("⚠ pynvml not available - using simulated hardware metrics")

from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class HardwareMetrics:
    """Hardware metrics (real or simulated)."""
    energy_joules: float
    temperature_celsius: float
    power_watts: float
    timestamp: float

@dataclass
class StressScenario:
    """Synthetic stress scenario for testing routing logic."""
    name: str
    description: str
    expert_energy_multipliers: List[float]  # Per-expert energy multipliers
    expert_temperature_offsets: List[float]  # Per-expert temperature offsets
    expected_behavior: str

class SimpleHardwareMonitor:
    """Simple monitor for hardware metrics (real or simulated)."""
    
    def __init__(self):
        self.device_handle = None
        self.last_timestamp = None
        
        if HARDWARE_AVAILABLE:
            try:
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.last_timestamp = time.time()
                print(f"✓ Connected to GPU: {pynvml.nvmlDeviceGetName(self.device_handle).decode()}")
            except Exception as e:
                print(f"⚠ Failed to initialize hardware monitor: {e}")
                # Note: Can't modify global in this context, will handle in get_metrics
    
    def get_metrics(self) -> HardwareMetrics:
        """Get current hardware metrics."""
        if not HARDWARE_AVAILABLE or self.device_handle is None:
            # Simulated metrics
            return HardwareMetrics(
                energy_joules=0.001,
                temperature_celsius=60.0,
                power_watts=200.0,
                timestamp=time.time()
            )
        
        try:
            current_time = time.time()
            
            # Calculate energy delta from power (simplified)
            power = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0
            time_delta = current_time - (self.last_timestamp or current_time)
            energy_delta = power * time_delta if time_delta > 0 else 0.001
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.device_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Update tracking
            self.last_timestamp = current_time
            
            return HardwareMetrics(
                energy_joules=float(energy_delta),
                temperature_celsius=float(temp),
                power_watts=float(power),
                timestamp=current_time
            )
            
        except Exception as e:
            print(f"⚠ Error reading hardware metrics: {e}")
            return HardwareMetrics(
                energy_joules=0.001,
                temperature_celsius=60.0,
                power_watts=200.0,
                timestamp=time.time()
            )
    
    def cleanup(self):
        """Cleanup hardware monitor."""
        if HARDWARE_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

class SimpleHardwareSyntheticTest:
    """Simple test of hardware metrics + synthetic stress."""
    
    def __init__(self, num_experts=16, d_model=768):
        self.num_experts = num_experts
        self.d_model = d_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize hardware monitor
        self.hardware_monitor = SimpleHardwareMonitor()
        
        # Initialize TTT router
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=2,
            lambda_energy=0.05
        ).to(self.device)
        
        # Define stress scenarios
        self.stress_scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> List[StressScenario]:
        """Define synthetic stress scenarios."""
        scenarios = []
        
        # Scenario 1: Energy imbalance - experts 0-3 are "hot"
        scenarios.append(StressScenario(
            name="energy_imbalance",
            description="Experts 0-3 have 3x higher energy consumption",
            expert_energy_multipliers=[3.0, 3.0, 3.0, 3.0] + [1.0] * (self.num_experts - 4),
            expert_temperature_offsets=[0.0] * self.num_experts,
            expected_behavior="Router should avoid experts 0-3"
        ))
        
        # Scenario 2: Thermal imbalance - experts 4-7 are "hot"
        scenarios.append(StressScenario(
            name="thermal_imbalance",
            description="Experts 4-7 have +20°C temperature offset",
            expert_energy_multipliers=[1.0] * self.num_experts,
            expert_temperature_offsets=[0.0] * 4 + [20.0] * 4 + [0.0] * (self.num_experts - 8),
            expected_behavior="Router should avoid experts 4-7"
        ))
        
        # Scenario 3: Combined stress
        scenarios.append(StressScenario(
            name="combined_stress",
            description="Experts 8-11 have both high energy (2x) and temperature (+15°C)",
            expert_energy_multipliers=[1.0] * 8 + [2.0] * 4 + [1.0] * (self.num_experts - 12),
            expert_temperature_offsets=[0.0] * 8 + [15.0] * 4 + [0.0] * (self.num_experts - 12),
            expected_behavior="Router should strongly avoid experts 8-11"
        ))
        
        return scenarios
    
    def _apply_stress(self, base_metrics: HardwareMetrics, 
                     scenario: StressScenario) -> HardwareMetrics:
        """Apply synthetic stress to base metrics."""
        # Apply energy multipliers
        avg_energy_multiplier = float(np.mean(scenario.expert_energy_multipliers))
        stressed_energy = base_metrics.energy_joules * avg_energy_multiplier
        
        # Apply temperature offsets
        max_temp_offset = float(max(scenario.expert_temperature_offsets))
        stressed_temperature = base_metrics.temperature_celsius + max_temp_offset
        
        return HardwareMetrics(
            energy_joules=stressed_energy,
            temperature_celsius=stressed_temperature,
            power_watts=base_metrics.power_watts,
            timestamp=base_metrics.timestamp
        )
    
    def _calculate_adaptation_score(self, scenario: StressScenario, 
                                  expert_usage_history: List[List[float]]) -> float:
        """Calculate how well the router adapted to stress."""
        if len(expert_usage_history) < 5:
            return 0.0
        
        # Identify stressed experts based on scenario
        stressed_experts = []
        if scenario.name == "energy_imbalance":
            stressed_experts = [0, 1, 2, 3]
        elif scenario.name == "thermal_imbalance":
            stressed_experts = [4, 5, 6, 7]
        elif scenario.name == "combined_stress":
            stressed_experts = [8, 9, 10, 11]
        
        if not stressed_experts:
            return 0.0
        
        # Calculate average usage of stressed vs non-stressed experts
        recent_usage = expert_usage_history[-5:]  # Last 5 batches
        
        stressed_usage = []
        non_stressed_usage = []
        
        for usage in recent_usage:
            stressed_avg = np.mean([usage[i] for i in stressed_experts])
            non_stressed_avg = np.mean([usage[i] for i in range(len(usage)) if i not in stressed_experts])
            stressed_usage.append(stressed_avg)
            non_stressed_usage.append(non_stressed_avg)
        
        # Adaptation score: lower stressed usage = better adaptation
        avg_stressed = np.mean(stressed_usage)
        avg_non_stressed = np.mean(non_stressed_usage)
        
        if avg_non_stressed == 0:
            return 0.0
        
        adaptation_score = 1.0 - (avg_stressed / avg_non_stressed)
        return max(0.0, min(1.0, float(adaptation_score)))
    
    def run_test(self, num_batches=30):
        """Run the test."""
        print(f"\n=== Simple Hardware + Synthetic Stress Test ===")
        print(f"Hardware metrics available: {HARDWARE_AVAILABLE}")
        print(f"Number of experts: {self.num_experts}")
        print(f"Number of batches: {num_batches}")
        print(f"Device: {self.device}")
        
        # Track results
        expert_usage_history = []
        current_scenario = None
        scenario_batch_count = 0
        
        for batch_idx in range(num_batches):
            # Get real hardware metrics
            real_metrics = self.hardware_monitor.get_metrics()
            
            # Determine current scenario
            if current_scenario is None or scenario_batch_count >= 10:
                scenario_idx = (batch_idx // 10) % len(self.stress_scenarios)
                current_scenario = self.stress_scenarios[scenario_idx]
                scenario_batch_count = 0
                print(f"\n--- Starting Scenario: {current_scenario.name} ---")
                print(f"Description: {current_scenario.description}")
                print(f"Expected behavior: {current_scenario.expected_behavior}")
            
            # Apply synthetic stress
            stressed_metrics = self._apply_stress(real_metrics, current_scenario)
            
            # Create input tensor
            input_tensor = torch.randn(8, 64, self.d_model).to(self.device)
            
            # Run TTT routing
            expert_indices, expert_weights, router_metadata = self.ttt_router(
                input_tensor, ttt_context={'batch_size': input_tensor.size(0)}
            )
            
            # Calculate expert usage
            expert_usage = torch.zeros(self.num_experts, device=self.device)
            for i in range(self.num_experts):
                expert_usage[i] = (expert_indices == i).sum().item()
            
            expert_usage_history.append(expert_usage.cpu().tolist())
            
            # Update TTT router with stressed metrics
            feedback = {
                'estimated_energy': stressed_metrics.energy_joules,
                'expert_usage': expert_usage,
                'token_count': input_tensor.numel(),
                'batch_size': input_tensor.size(0),
                'seq_length': input_tensor.size(1)
            }
            self.ttt_router.ttt_update(feedback)
            
            scenario_batch_count += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: Energy={stressed_metrics.energy_joules:.6f}J, "
                      f"Temp={stressed_metrics.temperature_celsius:.1f}°C, "
                      f"Scenario={current_scenario.name}")
        
        # Calculate adaptation scores for each scenario
        print(f"\n=== Adaptation Analysis ===")
        adaptation_scores = {}
        for scenario in self.stress_scenarios:
            adaptation_score = self._calculate_adaptation_score(scenario, expert_usage_history)
            adaptation_scores[scenario.name] = adaptation_score
            print(f"{scenario.name}: {adaptation_score:.3f}")
            
            # Grade the adaptation
            if adaptation_score > 0.7:
                grade = "A"
            elif adaptation_score > 0.5:
                grade = "B"
            elif adaptation_score > 0.3:
                grade = "C"
            else:
                grade = "D"
            
            print(f"  Grade: {grade}")
            if adaptation_score > 0.5:
                print(f"  ✓ Router successfully adapted to {scenario.name}")
            else:
                print(f"  ⚠ Router did not show strong adaptation to {scenario.name}")
        
        # Overall assessment
        avg_adaptation = np.mean(list(adaptation_scores.values()))
        
        print(f"\n=== Overall Assessment ===")
        print(f"Average Adaptation Score: {avg_adaptation:.3f}")
        print(f"TTT Updates: {self.ttt_router.ttt_update_count}")
        print(f"Hardware Metrics Used: {HARDWARE_AVAILABLE}")
        
        if avg_adaptation > 0.5:
            print("✓ Router successfully demonstrated adaptation to synthetic stress")
        else:
            print("⚠ Router needs tuning - consider adjusting lambda_energy or penalty scaling")
        
        return {
            'avg_adaptation_score': float(avg_adaptation),
            'ttt_updates': self.ttt_router.ttt_update_count,
            'hardware_used': HARDWARE_AVAILABLE,
            'adaptation_scores': adaptation_scores,
            'expert_usage_history': expert_usage_history
        }
    
    def cleanup(self):
        """Cleanup resources."""
        self.hardware_monitor.cleanup()

def main():
    """Run the simple hardware + synthetic stress test."""
    print("Simple Hardware + Synthetic Stress TTT Test")
    print("This test shows how the router responds to controlled synthetic stress scenarios")
    print("while using real hardware metrics when available.")
    
    # Run test
    test = SimpleHardwareSyntheticTest(num_experts=16, d_model=768)
    
    try:
        results = test.run_test(num_batches=30)
        
        # Save results
        with open('simple_hardware_synthetic_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, torch.Tensor) else x)
        
        print(f"\nResults saved to simple_hardware_synthetic_results.json")
        
    finally:
        test.cleanup()

if __name__ == "__main__":
    main() 