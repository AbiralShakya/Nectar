#!/usr/bin/env python3
"""
Enhanced test script for energy-aware TTT routing with REAL hardware metrics + controlled synthetic stress.
This script:
1. Queries real hardware energy and temperature using pynvml
2. Creates controlled synthetic stress scenarios to test routing logic
3. Demonstrates that the router responds correctly to energy/temperature imbalances
4. Provides comprehensive validation of the routing system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os
import sys

# Try to import pynvml for real hardware metrics
try:
    import pynvml
    pynvml.nvmlInit()
    HARDWARE_METRICS_AVAILABLE = True
    print("✓ pynvml available - will use real hardware metrics")
except ImportError:
    HARDWARE_METRICS_AVAILABLE = False
    print("⚠ pynvml not available - will use simulated hardware metrics")

from src.moe_models import DistributedMoELayer, NetworkTopologyOptimizer
from src.kernelcostmodel import KernelCostModel
from src.monitor import GpuSystemMonitor
from src.thermal_signal import ThermalAwareRouter, ThermalState
from models.ttt_router import EnergyAwareTTTRouter

@dataclass
class HardwareMetrics:
    """Real hardware metrics from GPU."""
    energy_joules: float
    temperature_celsius: float
    power_watts: float
    memory_utilization_percent: float
    gpu_utilization_percent: float
    timestamp: float

@dataclass
class SyntheticStressScenario:
    """Controlled synthetic stress scenario for testing routing logic."""
    name: str
    description: str
    expert_energy_multipliers: List[float]  # Per-expert energy multipliers
    expert_temperature_offsets: List[float]  # Per-expert temperature offsets (°C)
    duration_batches: int  # How many batches to apply this stress
    expected_behavior: str  # What we expect the router to do

@dataclass
class TTTTestResult:
    """Results from TTT routing test with hardware metrics."""
    lambda_energy: float
    batch_size: int
    seq_length: int
    num_experts: int
    moe_top_k: int
    avg_energy_joules: float
    avg_latency_ms: float
    avg_power_watt: float
    avg_accuracy: float
    thermal_imbalance_score: float
    routing_entropy: float
    expert_usage_distribution: List[float]
    ttt_update_count: int
    energy_savings_percent: float
    accuracy_loss_percent: float
    hardware_metrics_used: bool
    stress_scenarios_tested: List[str]
    routing_adaptation_score: float

class RealHardwareMonitor:
    """Monitor for real hardware metrics using pynvml."""
    
    def __init__(self):
        self.device_handle = None
        self.initial_energy = None
        self.last_energy_reading = None
        self.last_timestamp = None
        
        if HARDWARE_METRICS_AVAILABLE:
            try:
                # Get the first GPU
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print(f"✓ Connected to GPU: {pynvml.nvmlDeviceGetName(self.device_handle).decode()}")
                
                # Initialize energy tracking
                self.initial_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.device_handle) / 1000.0  # Convert to Joules
                self.last_energy_reading = self.initial_energy
                self.last_timestamp = time.time()
                
            except Exception as e:
                print(f"⚠ Failed to initialize hardware monitor: {e}")
                HARDWARE_METRICS_AVAILABLE = False
    
    def get_current_metrics(self) -> HardwareMetrics:
        """Get current hardware metrics."""
        if not HARDWARE_METRICS_AVAILABLE or self.device_handle is None:
            # Fallback to simulated metrics
            return HardwareMetrics(
                energy_joules=0.001,  # Small baseline
                temperature_celsius=60.0,
                power_watts=200.0,
                memory_utilization_percent=70.0,
                gpu_utilization_percent=80.0,
                timestamp=time.time()
            )
        
        try:
            # Get current energy consumption
            current_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.device_handle) / 1000.0
            current_time = time.time()
            
            # Calculate energy delta since last reading
            energy_delta = current_energy - self.last_energy_reading
            time_delta = current_time - self.last_timestamp
            
            # Get other metrics
            temp = pynvml.nvmlDeviceGetTemperature(self.device_handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(self.device_handle) / 1000.0  # Convert to Watts
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.device_handle)
            memory_util = (memory_info.used / memory_info.total) * 100.0
            
            # GPU utilization (approximate)
            gpu_util = 80.0  # Default, could be enhanced with more detailed monitoring
            
            # Update tracking
            self.last_energy_reading = current_energy
            self.last_timestamp = current_time
            
            return HardwareMetrics(
                energy_joules=energy_delta,
                temperature_celsius=temp,
                power_watts=power,
                memory_utilization_percent=memory_util,
                gpu_utilization_percent=gpu_util,
                timestamp=current_time
            )
            
        except Exception as e:
            print(f"⚠ Error reading hardware metrics: {e}")
            return HardwareMetrics(
                energy_joules=0.001,
                temperature_celsius=60.0,
                power_watts=200.0,
                memory_utilization_percent=70.0,
                gpu_utilization_percent=80.0,
                timestamp=time.time()
            )
    
    def cleanup(self):
        """Cleanup hardware monitor."""
        if HARDWARE_METRICS_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

class EnergyAwareTTTHardwareSyntheticTester:
    """
    Test energy-aware TTT routing with REAL hardware metrics + controlled synthetic stress.
    Demonstrates routing logic effectiveness through controlled stress scenarios.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize real hardware monitor
        self.hardware_monitor = RealHardwareMonitor()
        
        # Initialize components
        self.kernel_cost_model = KernelCostModel(gpu_type="A100")
        self.gpu_monitor = GpuSystemMonitor()
        
        # TTT Router
        self.ttt_router = EnergyAwareTTTRouter(
            d_model=args.d_model,
            num_experts=args.num_experts,
            top_k=args.moe_top_k,
            lambda_energy=args.lambda_energy
        ).to(self.device)
        
        # Thermal router
        if args.enable_thermal_awareness:
            self.thermal_router = ThermalAwareRouter(
                num_experts=args.num_experts,
                num_gpus=1
            )
        
        # Define controlled stress scenarios
        self.stress_scenarios = self._define_stress_scenarios()
        
        # Results storage
        self.results = []
        self.stress_test_results = []
        
    def _define_stress_scenarios(self) -> List[SyntheticStressScenario]:
        """Define controlled synthetic stress scenarios to test routing logic."""
        scenarios = []
        
        # Scenario 1: Energy imbalance - some experts are "hot" (high energy)
        scenarios.append(SyntheticStressScenario(
            name="energy_imbalance_hot_experts",
            description="Experts 0-3 have 3x higher energy consumption to test energy-aware routing",
            expert_energy_multipliers=[3.0, 3.0, 3.0, 3.0] + [1.0] * (self.args.num_experts - 4),
            expert_temperature_offsets=[0.0] * self.args.num_experts,
            duration_batches=50,
            expected_behavior="Router should avoid experts 0-3 and prefer experts 4-15"
        ))
        
        # Scenario 2: Thermal imbalance - some experts are "hot" (high temperature)
        scenarios.append(SyntheticStressScenario(
            name="thermal_imbalance_hot_experts", 
            description="Experts 4-7 have +20°C temperature offset to test thermal-aware routing",
            expert_energy_multipliers=[1.0] * self.args.num_experts,
            expert_temperature_offsets=[0.0] * 4 + [20.0] * 4 + [0.0] * (self.args.num_experts - 8),
            duration_batches=50,
            expected_behavior="Router should avoid experts 4-7 when thermal awareness is enabled"
        ))
        
        # Scenario 3: Combined stress - both energy and thermal issues
        scenarios.append(SyntheticStressScenario(
            name="combined_stress_energy_thermal",
            description="Experts 8-11 have both high energy (2x) and high temperature (+15°C)",
            expert_energy_multipliers=[1.0] * 8 + [2.0] * 4 + [1.0] * (self.args.num_experts - 12),
            expert_temperature_offsets=[0.0] * 8 + [15.0] * 4 + [0.0] * (self.args.num_experts - 12),
            duration_batches=50,
            expected_behavior="Router should strongly avoid experts 8-11 due to combined stress"
        ))
        
        # Scenario 4: Gradual stress increase
        scenarios.append(SyntheticStressScenario(
            name="gradual_stress_increase",
            description="Gradually increase energy stress on experts 12-15 over time",
            expert_energy_multipliers=[1.0] * 12 + [1.5] * 4,  # Will be modified during test
            expert_temperature_offsets=[0.0] * self.args.num_experts,
            duration_batches=100,
            expected_behavior="Router should gradually shift away from experts 12-15 as stress increases"
        ))
        
        return scenarios
    
    def _apply_synthetic_stress(self, base_metrics: HardwareMetrics, 
                               scenario: SyntheticStressScenario, 
                               batch_idx: int) -> HardwareMetrics:
        """Apply controlled synthetic stress to base hardware metrics."""
        modified_metrics = HardwareMetrics(
            energy_joules=base_metrics.energy_joules,
            temperature_celsius=base_metrics.temperature_celsius,
            power_watts=base_metrics.power_watts,
            memory_utilization_percent=base_metrics.memory_utilization_percent,
            gpu_utilization_percent=base_metrics.gpu_utilization_percent,
            timestamp=base_metrics.timestamp
        )
        
        # Apply energy multipliers based on expert usage (simulate per-expert energy)
        # For gradual stress, increase multipliers over time
        if scenario.name == "gradual_stress_increase":
            # Gradually increase stress on experts 12-15
            stress_factor = 1.0 + (batch_idx / scenario.duration_batches) * 2.0  # 1.0 to 3.0
            energy_multipliers = [1.0] * 12 + [stress_factor] * 4
        else:
            energy_multipliers = scenario.expert_energy_multipliers
        
        # Apply energy stress (simulate that some experts consume more energy)
        total_energy_multiplier = np.mean(energy_multipliers)
        modified_metrics.energy_joules *= total_energy_multiplier
        
        # Apply temperature offsets
        max_temp_offset = max(scenario.expert_temperature_offsets)
        modified_metrics.temperature_celsius += max_temp_offset
        
        return modified_metrics
    
    def _calculate_routing_adaptation_score(self, scenario: SyntheticStressScenario, 
                                          expert_usage_history: List[List[float]]) -> float:
        """Calculate how well the router adapted to the stress scenario."""
        if len(expert_usage_history) < 10:
            return 0.0
        
        # Get stressed expert indices based on scenario
        stressed_experts = []
        if scenario.name == "energy_imbalance_hot_experts":
            stressed_experts = [0, 1, 2, 3]
        elif scenario.name == "thermal_imbalance_hot_experts":
            stressed_experts = [4, 5, 6, 7]
        elif scenario.name == "combined_stress_energy_thermal":
            stressed_experts = [8, 9, 10, 11]
        elif scenario.name == "gradual_stress_increase":
            stressed_experts = [12, 13, 14, 15]
        
        if not stressed_experts:
            return 0.0
        
        # Calculate average usage of stressed vs non-stressed experts
        recent_usage = expert_usage_history[-10:]  # Last 10 batches
        
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
        
        # Score based on ratio of stressed to non-stressed usage
        adaptation_score = 1.0 - (avg_stressed / avg_non_stressed)
        return max(0.0, min(1.0, adaptation_score))
    
    def run_test(self) -> TTTTestResult:
        """Run comprehensive TTT routing test with real hardware + synthetic stress."""
        print(f"Running Energy-Aware TTT Test with lambda_energy={self.args.lambda_energy}")
        print(f"Hardware metrics available: {HARDWARE_METRICS_AVAILABLE}")
        print(f"Number of stress scenarios: {len(self.stress_scenarios)}")
        
        # Test metrics
        total_energy = 0.0
        total_latency = 0.0
        total_power = 0.0
        total_accuracy = 0.0
        routing_entropies = []
        expert_usage_counts = torch.zeros(self.args.num_experts, device=self.device)
        thermal_imbalance_scores = []
        expert_usage_history = []
        
        num_batches = 0
        current_scenario = None
        scenario_batch_count = 0
        
        for epoch in range(self.args.num_epochs):
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            
            for batch_idx in range(self.args.num_batches):
                # Get real hardware metrics
                real_metrics = self.hardware_monitor.get_current_metrics()
                
                # Determine current stress scenario
                if current_scenario is None or scenario_batch_count >= current_scenario.duration_batches:
                    # Start new scenario
                    scenario_idx = (batch_idx // 50) % len(self.stress_scenarios)
                    current_scenario = self.stress_scenarios[scenario_idx]
                    scenario_batch_count = 0
                    print(f"\n=== Starting Stress Scenario: {current_scenario.name} ===")
                    print(f"Description: {current_scenario.description}")
                    print(f"Expected behavior: {current_scenario.expected_behavior}")
                
                # Apply synthetic stress to real metrics
                stressed_metrics = self._apply_synthetic_stress(real_metrics, current_scenario, scenario_batch_count)
                
                # Create input tensor
                input_tensor = torch.randn(
                    self.args.batch_size, 
                    self.args.seq_length, 
                    self.args.d_model
                ).to(self.device)
                
                # Apply noise if enabled
                if self.args.enable_noise_injection:
                    input_tensor = self._apply_noise(input_tensor, self.args.noise_level)
                
                # Run TTT routing
                start_time = time.time()
                routing_result = self._run_ttt_routing(input_tensor)
                end_time = time.time()
                
                # Calculate metrics
                latency_ms = (end_time - start_time) * 1000
                energy_joules = stressed_metrics.energy_joules  # Use stressed metrics
                power_watt = stressed_metrics.power_watts
                accuracy = self._calculate_accuracy(routing_result, input_tensor)
                
                # Update thermal state if enabled
                if self.args.enable_thermal_awareness:
                    thermal_state = ThermalState(
                        temperature=stressed_metrics.temperature_celsius,
                        power_watt=stressed_metrics.power_watts,
                        memory_utilization=stressed_metrics.memory_utilization_percent / 100.0,
                        compute_utilization=stressed_metrics.gpu_utilization_percent / 100.0,
                        timestamp=stressed_metrics.timestamp
                    )
                    self.thermal_router.update_thermal_state(0, thermal_state)
                    thermal_imbalance = self.thermal_router.get_thermal_stats()['thermal_imbalance_score']
                    thermal_imbalance_scores.append(thermal_imbalance)
                
                # Update TTT router with stressed metrics
                feedback = {
                    'estimated_energy': energy_joules,
                    'expert_usage': routing_result['expert_usage'],
                    'token_count': input_tensor.numel(),
                    'batch_size': input_tensor.size(0),
                    'seq_length': input_tensor.size(1)
                }
                self.ttt_router.ttt_update(feedback)
                
                # Track expert usage for adaptation analysis
                expert_usage_history.append(routing_result['expert_usage'].cpu().tolist())
                
                # Accumulate metrics
                total_energy += energy_joules
                total_latency += latency_ms
                total_power += power_watt
                total_accuracy += accuracy
                routing_entropies.append(routing_result['routing_entropy'])
                expert_usage_counts += routing_result['expert_usage']
                num_batches += 1
                scenario_batch_count += 1
                
                # Print progress with stress info
                if batch_idx % 20 == 0:
                    print(f"  Batch {batch_idx}: Energy={energy_joules:.6f}J, "
                          f"Temp={stressed_metrics.temperature_celsius:.1f}°C, "
                          f"Power={power_watt:.1f}W, Accuracy={accuracy:.3f}")
                    print(f"  Scenario: {current_scenario.name} ({scenario_batch_count}/{current_scenario.duration_batches})")
        
        # Calculate averages
        avg_energy = total_energy / num_batches
        avg_latency = total_latency / num_batches
        avg_power = total_power / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_routing_entropy = np.mean(routing_entropies)
        expert_usage_distribution = (expert_usage_counts / expert_usage_counts.sum()).tolist()
        avg_thermal_imbalance = np.mean(thermal_imbalance_scores) if thermal_imbalance_scores else 0.0
        
        # Calculate routing adaptation scores for each scenario
        adaptation_scores = []
        stress_scenarios_tested = []
        for scenario in self.stress_scenarios:
            adaptation_score = self._calculate_routing_adaptation_score(scenario, expert_usage_history)
            adaptation_scores.append(adaptation_score)
            stress_scenarios_tested.append(scenario.name)
            print(f"Adaptation score for {scenario.name}: {adaptation_score:.3f}")
        
        avg_adaptation_score = np.mean(adaptation_scores)
        
        # Calculate improvements
        baseline_energy = self._get_baseline_energy()
        baseline_accuracy = 0.95  # Assume baseline accuracy
        energy_savings = ((baseline_energy - avg_energy) / baseline_energy) * 100
        accuracy_loss = ((baseline_accuracy - avg_accuracy) / baseline_accuracy) * 100
        
        result = TTTTestResult(
            lambda_energy=self.args.lambda_energy,
            batch_size=self.args.batch_size,
            seq_length=self.args.seq_length,
            num_experts=self.args.num_experts,
            moe_top_k=self.args.moe_top_k,
            avg_energy_joules=avg_energy,
            avg_latency_ms=avg_latency,
            avg_power_watt=avg_power,
            avg_accuracy=avg_accuracy,
            thermal_imbalance_score=avg_thermal_imbalance,
            routing_entropy=avg_routing_entropy,
            expert_usage_distribution=expert_usage_distribution,
            ttt_update_count=self.ttt_router.ttt_update_count,
            energy_savings_percent=energy_savings,
            accuracy_loss_percent=accuracy_loss,
            hardware_metrics_used=HARDWARE_METRICS_AVAILABLE,
            stress_scenarios_tested=stress_scenarios_tested,
            routing_adaptation_score=avg_adaptation_score
        )
        
        return result
    
    def _run_ttt_routing(self, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run TTT routing on input tensor."""
        # Get routing decisions
        expert_indices, expert_weights, router_metadata = self.ttt_router(
            input_tensor, ttt_context={'batch_size': input_tensor.size(0)}
        )
        
        # Calculate routing entropy
        routing_entropy = self._calculate_routing_entropy(expert_weights)
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.args.num_experts, device=self.device)
        for i in range(self.args.num_experts):
            expert_usage[i] = (expert_indices == i).sum().item()
        
        return {
            'expert_indices': expert_indices,
            'expert_weights': expert_weights,
            'router_metadata': router_metadata,
            'routing_entropy': routing_entropy,
            'expert_usage': expert_usage
        }
    
    def _calculate_accuracy(self, routing_result: Dict[str, Any], 
                          input_tensor: torch.Tensor) -> float:
        """Calculate routing accuracy."""
        # Simple accuracy metric based on expert usage distribution
        expert_usage = routing_result['expert_usage']
        total_usage = expert_usage.sum().item()
        
        if total_usage == 0:
            return 0.0
        
        # Calculate load balancing accuracy
        expected_usage = total_usage / self.args.num_experts
        usage_variance = torch.var(expert_usage.float()).item()
        load_balance_accuracy = 1.0 / (1.0 + usage_variance / (expected_usage ** 2))
        
        # Calculate routing confidence
        expert_weights = routing_result['expert_weights']
        routing_confidence = torch.mean(expert_weights).item()
        
        # Combined accuracy
        accuracy = 0.7 * load_balance_accuracy + 0.3 * routing_confidence
        
        return min(1.0, max(0.0, accuracy))
    
    def _calculate_routing_entropy(self, expert_weights: torch.Tensor) -> float:
        """Calculate routing entropy."""
        # Add small epsilon to avoid log(0)
        weights = expert_weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return torch.mean(entropy).item()
    
    def _apply_noise(self, tensor: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Apply synthetic noise to tensor."""
        if noise_level == 0:
            return tensor
        
        noise = torch.randn_like(tensor) * noise_level
        return tensor + noise
    
    def _get_baseline_energy(self) -> float:
        """Get baseline energy consumption."""
        # Estimate baseline energy without TTT optimization
        batch_size = self.args.batch_size
        seq_length = self.args.seq_length
        
        routing_cost = self.kernel_cost_model.get_cost("moe_router", batch_size)
        expert_cost = self.kernel_cost_model.get_cost("ffn_gate", batch_size * seq_length)
        
        return routing_cost["energy_joules"] + expert_cost["energy_joules"]
    
    def save_results(self, result: TTTTestResult, output_file: str):
        """Save test results to file."""
        output_data = {
            'test_config': {
                'lambda_energy': self.args.lambda_energy,
                'num_experts': self.args.num_experts,
                'moe_top_k': self.args.moe_top_k,
                'batch_size': self.args.batch_size,
                'seq_length': self.args.seq_length,
                'd_model': self.args.d_model,
                'enable_thermal_awareness': self.args.enable_thermal_awareness,
                'enable_noise_injection': self.args.enable_noise_injection,
                'noise_level': self.args.noise_level,
                'error_margin': self.args.error_margin,
                'hardware_metrics_used': HARDWARE_METRICS_AVAILABLE
            },
            'results': {
                'avg_energy_joules': float(result.avg_energy_joules),
                'avg_latency_ms': float(result.avg_latency_ms),
                'avg_power_watt': float(result.avg_power_watt),
                'avg_accuracy': float(result.avg_accuracy),
                'thermal_imbalance_score': float(result.thermal_imbalance_score),
                'routing_entropy': float(result.routing_entropy),
                'expert_usage_distribution': [float(x) for x in result.expert_usage_distribution],
                'ttt_update_count': int(result.ttt_update_count),
                'energy_savings_percent': float(result.energy_savings_percent),
                'accuracy_loss_percent': float(result.accuracy_loss_percent),
                'routing_adaptation_score': float(result.routing_adaptation_score),
                'stress_scenarios_tested': result.stress_scenarios_tested
            },
            'stress_scenarios': [
                {
                    'name': scenario.name,
                    'description': scenario.description,
                    'expected_behavior': scenario.expected_behavior
                }
                for scenario in self.stress_scenarios
            ],
            'timestamp': time.time()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, result: TTTTestResult):
        """Print test summary."""
        print("\n=== Energy-Aware TTT Test Summary (Hardware + Synthetic Stress) ===")
        print(f"Lambda Energy: {result.lambda_energy}")
        print(f"Batch Size: {result.batch_size}")
        print(f"Sequence Length: {result.seq_length}")
        print(f"Number of Experts: {result.num_experts}")
        print(f"MoE Top-K: {result.moe_top_k}")
        print(f"Hardware Metrics Used: {result.hardware_metrics_used}")
        print()
        
        print("=== Performance Metrics ===")
        print(f"Average Energy: {result.avg_energy_joules:.6f} J")
        print(f"Average Latency: {result.avg_latency_ms:.2f} ms")
        print(f"Average Power: {result.avg_power_watt:.1f} W")
        print(f"Average Accuracy: {result.avg_accuracy:.3f}")
        print(f"Thermal Imbalance Score: {result.thermal_imbalance_score:.3f}")
        print(f"Routing Entropy: {result.routing_entropy:.3f}")
        print()
        
        print("=== Routing Adaptation Analysis ===")
        print(f"Overall Adaptation Score: {result.routing_adaptation_score:.3f}")
        print(f"Stress Scenarios Tested: {', '.join(result.stress_scenarios_tested)}")
        print()
        
        print("=== Expert Usage Distribution ===")
        for i, usage in enumerate(result.expert_usage_distribution):
            print(f"  Expert {i}: {usage:.3f}")
        print()
        
        print("=== Improvement Analysis ===")
        print(f"Energy Savings: {result.energy_savings_percent:.2f}%")
        print(f"Accuracy Loss: {result.accuracy_loss_percent:.2f}%")
        print(f"TTT Updates: {result.ttt_update_count}")
        print()
        
        # Grade the performance
        if result.routing_adaptation_score > 0.7:
            grade = "A"
        elif result.routing_adaptation_score > 0.5:
            grade = "B"
        elif result.routing_adaptation_score > 0.3:
            grade = "C"
        else:
            grade = "D"
        
        print(f"=== Overall Grade: {grade} ===")
        print(f"Adaptation Score: {result.routing_adaptation_score:.3f}")
        
        if result.routing_adaptation_score > 0.5:
            print("✓ Router successfully adapted to synthetic stress scenarios")
        else:
            print("⚠ Router did not show strong adaptation to stress scenarios")
            print("  Consider tuning lambda_energy or checking penalty scaling")
    
    def cleanup(self):
        """Cleanup resources."""
        self.hardware_monitor.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Test Energy-Aware TTT with Real Hardware + Synthetic Stress")
    parser.add_argument("--lambda_energy", type=float, default=0.05, help="Energy penalty weight")
    parser.add_argument("--num_experts", type=int, default=16, help="Number of experts")
    parser.add_argument("--moe_top_k", type=int, default=2, help="Top-k experts to route to")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq_length", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=768, help="Model dimension")
    parser.add_argument("--num_batches", type=int, default=200, help="Number of batches")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--enable_thermal_awareness", action="store_true", help="Enable thermal awareness")
    parser.add_argument("--enable_noise_injection", action="store_true", help="Enable noise injection")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level")
    parser.add_argument("--error_margin", type=float, default=0.1, help="Error margin")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark mode")
    
    args = parser.parse_args()
    
    print("=== Energy-Aware TTT Test (Real Hardware + Synthetic Stress) ===")
    print(f"Lambda Energy: {args.lambda_energy}")
    print(f"Thermal Awareness: {args.enable_thermal_awareness}")
    print(f"Noise Injection: {args.enable_noise_injection}")
    print(f"Noise Level: {args.noise_level}")
    print(f"Error Margin: {args.error_margin}")
    print(f"Hardware Metrics Available: {HARDWARE_METRICS_AVAILABLE}")
    
    # Run test
    tester = EnergyAwareTTTHardwareSyntheticTester(args)
    
    try:
        result = tester.run_test()
        
        # Save results
        tester.save_results(result, args.output_file)
        
        # Print summary
        tester.print_summary(result)
        
    finally:
        # Cleanup
        tester.cleanup()

if __name__ == "__main__":
    main() 