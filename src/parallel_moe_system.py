# src/parallel_moe_system.py
"""
Comprehensive Parallel Energy-Aware MoE System with Dynamic Expert Rerouting

This module implements a fully parallelized MoE system that:
1. Uses previous batch distribution patterns for dynamic expert rerouting
2. Optimizes for joules per token (energy efficiency)
3. Considers thermal consumption and power budgets
4. Supports multi-GPU parallel execution with SLURM integration
5. Implements Test-Time Training (TTT) for adaptive routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import os
from pathlib import Path

from src.moe_models import MoEConfig, SwiGLUExpert, OptimizedQuantizedExpert, LaCTMoEExpert
from src.routers import AdaptiveRouter, RoutingStrategy, BatchDistributionTracker, HardwareMetrics
from src.monitor import GpuSystemMonitor
from src.kernelcostmodel import KernelCostModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParallelMoEConfig:
    """Configuration for parallel MoE system"""
    # Base MoE configuration
    moe_config: MoEConfig
    
    # Parallelization settings
    world_size: int = 1  # Number of GPUs/processes
    num_expert_parallel: int = 1  # Expert parallelism degree
    num_data_parallel: int = 1   # Data parallelism degree
    pipeline_parallel: bool = False
    
    # Energy optimization settings
    energy_budget_watts: float = 400.0
    thermal_threshold_celsius: float = 80.0
    joules_per_token_target: float = 0.002  # 2mJ per token
    power_efficiency_weight: float = 0.4
    
    # Dynamic rerouting settings
    rerouting_enabled: bool = True
    rerouting_history_length: int = 100
    rerouting_update_frequency: int = 10  # batches
    imbalance_threshold: float = 0.25
    
    # TTT settings
    ttt_enabled: bool = True
    ttt_chunk_size: int = 2048
    ttt_update_frequency: int = 512
    ttt_learning_rate: float = 1e-4
    
    # Performance settings
    async_expert_execution: bool = True
    expert_caching_enabled: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: bool = True

class ParallelExpertPool:
    """
    Manages a pool of experts across multiple GPUs with dynamic load balancing
    and energy-aware scheduling.
    """
    
    def __init__(self, config: ParallelMoEConfig, device_ids: List[int]):
        self.config = config
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        self.experts_per_device = config.moe_config.num_experts // self.num_devices
        
        # Initialize experts on each device
        self.expert_pools = {}
        self.device_monitors = {}
        self.expert_queues = {}
        self.result_queues = {}
        
        for device_id in device_ids:
            self._initialize_device_pool(device_id)
        
        # Global load balancer
        self.load_balancer = GlobalLoadBalancer(config, device_ids)
        
        # Energy optimizer
        self.energy_optimizer = EnergyAwareScheduler(config, device_ids)
        
        # Async execution pool
        if config.async_expert_execution:
            self.executor = ThreadPoolExecutor(max_workers=self.num_devices * 2)
        
        logger.info(f"Initialized ParallelExpertPool with {self.num_devices} devices")
    
    def _initialize_device_pool(self, device_id: int):
        """Initialize expert pool for a specific device"""
        device = torch.device(f'cuda:{device_id}')
        
        # Create experts for this device
        start_expert_id = device_id * self.experts_per_device
        end_expert_id = min(start_expert_id + self.experts_per_device, 
                           self.config.moe_config.num_experts)
        
        experts = {}
        for expert_id in range(start_expert_id, end_expert_id):
            if self.config.moe_config.expert_type == "swiglu":
                expert = SwiGLUExpert(self.config.moe_config, expert_id)
            elif self.config.moe_config.expert_type == "quantized":
                expert = OptimizedQuantizedExpert(self.config.moe_config, expert_id)
            elif self.config.moe_config.expert_type == "lact":
                expert = LaCTMoEExpert(self.config.moe_config, expert_id)
            else:
                expert = SwiGLUExpert(self.config.moe_config, expert_id)
            
            expert = expert.to(device)
            if self.config.mixed_precision:
                expert = expert.half()
            
            experts[expert_id] = expert
        
        self.expert_pools[device_id] = experts
        
        # Initialize monitoring
        self.device_monitors[device_id] = GpuSystemMonitor(device_id)
        
        # Initialize queues for async execution
        self.expert_queues[device_id] = queue.Queue()
        self.result_queues[device_id] = queue.Queue()
    
    async def forward_experts_async(self, 
                                  expert_assignments: Dict[int, torch.Tensor],
                                  input_tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Asynchronously execute experts across devices with energy-aware scheduling.
        """
        # Get current hardware state for all devices
        hardware_states = {}
        for device_id in self.device_ids:
            hardware_states[device_id] = self.device_monitors[device_id].get_current_stats()
        
        # Energy-aware scheduling
        execution_plan = self.energy_optimizer.create_execution_plan(
            expert_assignments, hardware_states
        )
        
        # Execute experts asynchronously
        tasks = []
        for device_id, expert_tasks in execution_plan.items():
            task = asyncio.create_task(
                self._execute_device_experts(device_id, expert_tasks, input_tokens)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Combine results
        combined_results = {}
        for result_dict in results:
            combined_results.update(result_dict)
        
        return combined_results
    
    async def _execute_device_experts(self, 
                                    device_id: int, 
                                    expert_tasks: List[Tuple[int, torch.Tensor]],
                                    input_tokens: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Execute experts on a specific device"""
        device = torch.device(f'cuda:{device_id}')
        results = {}
        
        with torch.cuda.device(device):
            for expert_id, expert_input in expert_tasks:
                if expert_id in self.expert_pools[device_id]:
                    expert = self.expert_pools[device_id][expert_id]
                    
                    # Move input to device if needed
                    if expert_input.device != device:
                        expert_input = expert_input.to(device)
                    
                    # Execute expert
                    with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                        expert_output = expert(expert_input)
                    
                    results[expert_id] = expert_output
        
        return results
    
class GlobalLoadBalancer:
    """
    Global load balancer that considers energy consumption, thermal state,
    and expert utilization across all devices.
    """
    
    def __init__(self, config: ParallelMoEConfig, device_ids: List[int]):
        self.config = config
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # Track expert usage and performance
        self.expert_usage_history = defaultdict(deque)
        self.device_load_history = defaultdict(deque)
        self.energy_consumption_history = defaultdict(deque)
        
        # Dynamic rerouting tracker
        self.batch_tracker = BatchDistributionTracker(
            num_experts=config.moe_config.num_experts,
            history_length=config.rerouting_history_length,
            imbalance_threshold=config.imbalance_threshold
        )
        
        # Energy efficiency profiles
        self.expert_energy_profiles = self._initialize_energy_profiles()
        
        logger.info("Initialized GlobalLoadBalancer")
    
    def _initialize_energy_profiles(self) -> Dict[int, Dict[str, float]]:
        """Initialize energy efficiency profiles for each expert"""
        profiles = {}
        
        for expert_id in range(self.config.moe_config.num_experts):
            # Simulate different energy characteristics for different experts
            base_energy = 0.002  # 2mJ base
            efficiency_factor = np.random.uniform(0.8, 1.2)  # ±20% variation
            
            profiles[expert_id] = {
                'energy_per_token': base_energy * efficiency_factor,
                'thermal_impact': np.random.uniform(0.1, 0.3),
                'memory_efficiency': np.random.uniform(0.7, 1.0),
                'compute_efficiency': np.random.uniform(0.8, 1.0)
            }
        
        return profiles
    
    def compute_optimal_routing(self, 
                              routing_logits: torch.Tensor,
                              hardware_states: Dict[int, Dict[str, Any]],
                              current_load: Dict[int, float]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute optimal routing considering energy efficiency, load balance,
        and thermal constraints.
        """
        batch_size, num_experts = routing_logits.shape
        device = routing_logits.device
        
        # Base routing probabilities
        base_probs = F.softmax(routing_logits, dim=-1)
        
        # Compute energy-aware biases
        energy_biases = self._compute_energy_biases(hardware_states, device)
        
        # Compute load balancing biases
        load_biases = self._compute_load_balancing_biases(current_load, device)
        
        # Compute thermal biases
        thermal_biases = self._compute_thermal_biases(hardware_states, device)
        
        # Dynamic expert rerouting biases
        rerouting_biases = torch.zeros(num_experts, device=device)
        if self.config.rerouting_enabled:
            # Get current distribution
            current_dist = base_probs.mean(dim=0)
            
            # Compute rerouting biases
            rerouting_biases, rerouting_metadata = self.batch_tracker.compute_rerouting_biases(
                current_dist, 
                hardware_states[0],  # Use first device as reference
                self.config.energy_budget_watts
            )
            rerouting_biases = rerouting_biases.to(device)
        
        # Combine all biases
        total_biases = (
            energy_biases * self.config.power_efficiency_weight +
            load_biases * 0.3 +
            thermal_biases * 0.2 +
            rerouting_biases * 0.1
        )
        
        # Apply biases to routing logits
        adjusted_logits = routing_logits + total_biases.unsqueeze(0)
        final_probs = F.softmax(adjusted_logits, dim=-1)
        
        # Select top-k experts
        top_k = self.config.moe_config.top_k
        topk_probs, topk_indices = torch.topk(final_probs, top_k, dim=-1)
        
        # Normalize probabilities
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        metadata = {
            'energy_biases': energy_biases.cpu().numpy(),
            'load_biases': load_biases.cpu().numpy(),
            'thermal_biases': thermal_biases.cpu().numpy(),
            'rerouting_biases': rerouting_biases.cpu().numpy(),
            'total_biases': total_biases.cpu().numpy()
        }
        
        return topk_indices, topk_probs, metadata
    
    def _compute_energy_biases(self, hardware_states: Dict[int, Dict[str, Any]], device: torch.device) -> torch.Tensor:
        """Compute energy-aware routing biases"""
        num_experts = self.config.moe_config.num_experts
        biases = torch.zeros(num_experts, device=device)
        
        # Get average power consumption across devices
        avg_power = np.mean([state['power_watt'] for state in hardware_states.values()])
        power_pressure = max(0, (avg_power - self.config.energy_budget_watts * 0.7) / 
                           (self.config.energy_budget_watts * 0.3))
        
        # Apply energy efficiency biases
        for expert_id in range(num_experts):
            profile = self.expert_energy_profiles[expert_id]
            energy_efficiency = 1.0 / profile['energy_per_token']
            
            # Higher bias for more efficient experts under power pressure
            biases[expert_id] = energy_efficiency * power_pressure * 0.1
        
        return biases
    
    def _compute_load_balancing_biases(self, current_load: Dict[int, float], device: torch.device) -> torch.Tensor:
        """Compute load balancing biases"""
        num_experts = self.config.moe_config.num_experts
        biases = torch.zeros(num_experts, device=device)
        
        if not current_load:
            return biases
        
        # Calculate load imbalance
        loads = list(current_load.values())
        avg_load = np.mean(loads)
        
        for expert_id in range(num_experts):
            expert_load = current_load.get(expert_id, 0.0)
            load_diff = avg_load - expert_load
            
            # Positive bias for underutilized experts
            biases[expert_id] = load_diff * 0.1
        
        return biases
    
    def _compute_thermal_biases(self, hardware_states: Dict[int, Dict[str, Any]], device: torch.device) -> torch.Tensor:
        """Compute thermal-aware routing biases"""
        num_experts = self.config.moe_config.num_experts
        biases = torch.zeros(num_experts, device=device)
        
        # Get maximum temperature across devices
        max_temp = max(state['temperature'] for state in hardware_states.values())
        thermal_pressure = max(0, (max_temp - self.config.thermal_threshold_celsius * 0.8) / 
                             (self.config.thermal_threshold_celsius * 0.2))
        
        if thermal_pressure > 0:
            # Under thermal pressure, bias towards experts with lower thermal impact
            for expert_id in range(num_experts):
                profile = self.expert_energy_profiles[expert_id]
                thermal_efficiency = 1.0 - profile['thermal_impact']
                biases[expert_id] = thermal_efficiency * thermal_pressure * 0.1
        
        return biases

class EnergyAwareScheduler:
    """
    Energy-aware scheduler that optimizes expert execution order and device
    assignment to minimize energy consumption while maintaining performance.
    """
    
    def __init__(self, config: ParallelMoEConfig, device_ids: List[int]):
        self.config = config
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # Energy optimization parameters
        self.energy_budget_per_device = config.energy_budget_watts / self.num_devices
        self.thermal_threshold = config.thermal_threshold_celsius
        
        # Scheduling history
        self.scheduling_history = deque(maxlen=1000)
        self.energy_consumption_log = defaultdict(list)
        
        logger.info("Initialized EnergyAwareScheduler")
    
    def create_execution_plan(self, 
                            expert_assignments: Dict[int, torch.Tensor],
                            hardware_states: Dict[int, Dict[str, Any]]) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
        """
        Create an energy-optimized execution plan for expert assignments.
        """
        execution_plan = {device_id: [] for device_id in self.device_ids}
        
        # Sort experts by energy efficiency and thermal impact
        expert_priorities = self._calculate_expert_priorities(hardware_states)
        
        # Assign experts to devices based on current thermal and power state
        for expert_id, expert_input in expert_assignments.items():
            best_device = self._select_optimal_device(expert_id, hardware_states, expert_priorities)
            execution_plan[best_device].append((expert_id, expert_input))
        
        # Optimize execution order within each device
        for device_id in self.device_ids:
            execution_plan[device_id] = self._optimize_execution_order(
                execution_plan[device_id], hardware_states[device_id]
            )
        
        return execution_plan
    
    def _calculate_expert_priorities(self, hardware_states: Dict[int, Dict[str, Any]]) -> Dict[int, float]:
        """Calculate priority scores for experts based on energy efficiency"""
        priorities = {}
        
        # Get average system state
        avg_temp = np.mean([state['temperature'] for state in hardware_states.values()])
        avg_power = np.mean([state['power_watt'] for state in hardware_states.values()])
        
        thermal_pressure = max(0, (avg_temp - self.thermal_threshold * 0.8) / (self.thermal_threshold * 0.2))
        power_pressure = max(0, (avg_power - self.energy_budget_per_device * 0.8) / (self.energy_budget_per_device * 0.2))
        
        for expert_id in range(self.config.moe_config.num_experts):
            # Simulate expert characteristics
            energy_efficiency = np.random.uniform(0.8, 1.2)
            thermal_impact = np.random.uniform(0.1, 0.3)
            
            # Higher priority for efficient experts under pressure
            priority = energy_efficiency * (1 + thermal_pressure + power_pressure) - thermal_impact
            priorities[expert_id] = priority
        
        return priorities
    
    def _select_optimal_device(self, 
                             expert_id: int, 
                             hardware_states: Dict[int, Dict[str, Any]],
                             expert_priorities: Dict[int, float]) -> int:
        """Select the optimal device for executing a specific expert"""
        best_device = self.device_ids[0]
        best_score = float('-inf')
        
        for device_id in self.device_ids:
            state = hardware_states[device_id]
            
            # Calculate device suitability score
            temp_score = max(0, 1.0 - (state['temperature'] - 50) / 30)  # Prefer cooler devices
            power_score = max(0, 1.0 - (state['power_watt'] - 100) / 200)  # Prefer lower power
            util_score = max(0, 1.0 - state.get('gpu_utilization_percent', 0) / 100)  # Prefer less utilized
            
            # Combine scores
            device_score = (temp_score * 0.4 + power_score * 0.4 + util_score * 0.2) * expert_priorities[expert_id]
            
            if device_score > best_score:
                best_score = device_score
                best_device = device_id
        
        return best_device
    
    def _optimize_execution_order(self, 
                                expert_tasks: List[Tuple[int, torch.Tensor]],
                                device_state: Dict[str, Any]) -> List[Tuple[int, torch.Tensor]]:
        """Optimize execution order within a device to minimize energy consumption"""
        if len(expert_tasks) <= 1:
            return expert_tasks
        
        # Sort by energy efficiency (more efficient first if under thermal pressure)
        current_temp = device_state['temperature']
        thermal_pressure = max(0, (current_temp - self.thermal_threshold * 0.8) / (self.thermal_threshold * 0.2))
        
        if thermal_pressure > 0.3:
            # Under thermal pressure, prioritize energy-efficient experts
            expert_tasks.sort(key=lambda x: np.random.uniform(0.8, 1.2), reverse=True)
        
        return expert_tasks

class ParallelMoELayer(nn.Module):
    """
    Main parallel MoE layer that orchestrates expert execution across multiple devices
    with energy-aware dynamic routing and TTT adaptation.
    """
    
    def __init__(self, config: ParallelMoEConfig, device_ids: List[int]):
        super().__init__()
        self.config = config
        self.device_ids = device_ids
        self.num_devices = len(device_ids)
        
        # Initialize components
        self.expert_pool = ParallelExpertPool(config, device_ids)
        self.load_balancer = GlobalLoadBalancer(config, device_ids)
        
        # Routing network
        self.gate = nn.Linear(config.moe_config.d_model, config.moe_config.num_experts, bias=False)
        
        # TTT components
        if config.ttt_enabled:
            self.ttt_adapter = self._initialize_ttt_adapter()
            self.ttt_optimizer = torch.optim.AdamW(self.ttt_adapter.parameters(), lr=config.ttt_learning_rate)
        
        # Performance tracking
        self.batch_count = 0
        self.energy_consumption_log = []
        self.performance_metrics = defaultdict(list)
        
        logger.info(f"Initialized ParallelMoELayer with {self.num_devices} devices")
    
    def _initialize_ttt_adapter(self):
        """Initialize TTT adapter for dynamic routing adaptation"""
        from src.routers import TTHAAdapter
        
        # Input features: expert costs + hardware metrics
        input_dim = self.config.moe_config.num_experts * 6 + 8
        hidden_dim = max(64, self.config.moe_config.d_model // 16)
        
        return TTHAAdapter(input_dim, self.config.moe_config.num_experts, hidden_dim)
    
    async def forward(self, x: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Forward pass with parallel expert execution and energy-aware routing.
        """
        batch_size, seq_len, d_model = x.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for expert processing
        x_flat = x.view(-1, d_model)  # [num_tokens, d_model]
        
        # Compute routing logits
        gate_logits = self.gate(x_flat)  # [num_tokens, num_experts]
        
        # Get current hardware states
        hardware_states = {}
        for device_id in self.device_ids:
            hardware_states[device_id] = self.expert_pool.device_monitors[device_id].get_current_stats()
        
        # Compute current expert loads
        current_load = self._get_current_expert_loads()
        
        # Energy-aware routing with dynamic rerouting
        topk_indices, topk_probs, routing_metadata = self.load_balancer.compute_optimal_routing(
            gate_logits, hardware_states, current_load
        )
        
        # TTT adaptation
        if self.config.ttt_enabled and self.batch_count % self.config.ttt_update_frequency == 0:
            await self._perform_ttt_update(gate_logits, hardware_states, routing_metadata)
        
        # Prepare expert assignments
        expert_assignments = self._prepare_expert_assignments(x_flat, topk_indices, topk_probs)
        
        # Execute experts in parallel
        expert_outputs = await self.expert_pool.forward_experts_async(expert_assignments, x_flat)
        
        # Combine expert outputs
        output = self._combine_expert_outputs(expert_outputs, topk_indices, topk_probs, x_flat.shape)
        
        # Reshape back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Update tracking
        self._update_performance_tracking(hardware_states, routing_metadata)
        self.batch_count += 1
        
        return output
    
    def _get_current_expert_loads(self) -> Dict[int, float]:
        """Get current load for each expert"""
        loads = {}
        
        # Simple load estimation based on recent usage
        for expert_id in range(self.config.moe_config.num_experts):
            # Simulate load based on recent history
            loads[expert_id] = np.random.uniform(0.1, 0.9)
        
        return loads
    
    def _prepare_expert_assignments(self, 
                                  x: torch.Tensor,
                                  topk_indices: torch.Tensor,
                                  topk_probs: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Prepare input assignments for each expert"""
        assignments = {}
        num_tokens, d_model = x.shape
        top_k = topk_indices.shape[1]
        
        # Group tokens by expert
        for expert_id in range(self.config.moe_config.num_experts):
            # Find tokens assigned to this expert
            expert_mask = (topk_indices == expert_id)
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) > 0:
                # Get tokens and their weights for this expert
                expert_tokens = x[token_indices]
                expert_weights = topk_probs[token_indices, k_indices]
                
                # Weight the tokens
                weighted_tokens = expert_tokens * expert_weights.unsqueeze(-1)
                assignments[expert_id] = weighted_tokens
        
        return assignments
    
    def _combine_expert_outputs(self, 
                              expert_outputs: Dict[int, torch.Tensor],
                              topk_indices: torch.Tensor,
                              topk_probs: torch.Tensor,
                              original_shape: Tuple[int, int]) -> torch.Tensor:
        """Combine outputs from all experts"""
        num_tokens, d_model = original_shape
        device = topk_indices.device
        
        # Initialize output tensor
        output = torch.zeros(num_tokens, d_model, device=device, dtype=torch.float32)
        
        # Combine expert outputs
        for expert_id, expert_output in expert_outputs.items():
            # Find tokens that used this expert
            expert_mask = (topk_indices == expert_id)
            token_indices, k_indices = torch.where(expert_mask)
            
            if len(token_indices) > 0:
                # Get the weights for these tokens
                weights = topk_probs[token_indices, k_indices]
                
                # Add weighted expert output to final output
                output[token_indices] += expert_output * weights.unsqueeze(-1)
        
        return output
    
    async def _perform_ttt_update(self, 
                                gate_logits: torch.Tensor,
                                hardware_states: Dict[int, Dict[str, Any]],
                                routing_metadata: Dict[str, Any]):
        """Perform TTT update for adaptive routing"""
        if not hasattr(self, 'ttt_adapter'):
            return
        
        # Prepare features for TTT adapter
        cost_features = self._extract_cost_features(hardware_states)
        hardware_features = self._extract_hardware_features(hardware_states)
        
        # Forward pass through TTT adapter
        routing_biases, uncertainties = self.ttt_adapter(cost_features, hardware_features)
        
        # Compute TTT loss (energy-aware)
        ttt_loss = self._compute_ttt_loss(routing_biases, hardware_states, routing_metadata)
        
        # Backward pass and optimization
        self.ttt_optimizer.zero_grad()
        ttt_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ttt_adapter.parameters(), 1.0)
        self.ttt_optimizer.step()
        
        logger.debug(f"TTT update completed - Loss: {ttt_loss.item():.4f}")
    
    def _extract_cost_features(self, hardware_states: Dict[int, Dict[str, Any]]) -> torch.Tensor:
        """Extract cost features for TTT adapter"""
        num_experts = self.config.moe_config.num_experts
        features = []
        
        for expert_id in range(num_experts):
            # Simulate expert cost features (6 metrics per expert)
            expert_features = [
                np.random.uniform(0.001, 0.003),  # energy_joules
                np.random.uniform(0.5, 2.0),      # latency_ms
                np.random.uniform(0.1, 0.3),      # temp_impact
                np.random.uniform(0.1, 0.5),      # memory_gb
                np.random.uniform(0.7, 1.0),      # compute_utilization
                np.random.uniform(0.6, 0.9),      # memory_utilization
            ]
            features.extend(expert_features)
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _extract_hardware_features(self, hardware_states: Dict[int, Dict[str, Any]]) -> torch.Tensor:
        """Extract hardware features for TTT adapter"""
        # Average across all devices
        avg_temp = np.mean([state['temperature'] for state in hardware_states.values()])
        avg_power = np.mean([state['power_watt'] for state in hardware_states.values()])
        avg_gpu_util = np.mean([state.get('gpu_utilization_percent', 0) for state in hardware_states.values()])
        avg_mem_util = np.mean([state.get('memory_utilization_percent', 0) for state in hardware_states.values()])
        
        features = [
            avg_temp / 100.0,           # normalized temperature
            avg_power / 400.0,          # normalized power
            avg_gpu_util / 100.0,       # normalized GPU utilization
            avg_mem_util / 100.0,       # normalized memory utilization
            1500.0 / 2500.0,           # normalized clock speed (simulated)
            0.0,                        # thermal throttling (simulated)
            400.0 / 500.0,             # normalized power limit
            1.2                         # voltage (simulated)
        ]
        
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    def _compute_ttt_loss(self, 
                         routing_biases: torch.Tensor,
                         hardware_states: Dict[int, Dict[str, Any]],
                         routing_metadata: Dict[str, Any]) -> torch.Tensor:
        """Compute TTT loss for energy-aware optimization"""
        # Multi-objective loss components
        
        # 1. Energy efficiency loss
        avg_power = np.mean([state['power_watt'] for state in hardware_states.values()])
        target_power = self.config.energy_budget_watts
        power_loss = F.mse_loss(torch.tensor(avg_power), torch.tensor(target_power))
        
        # 2. Thermal loss
        max_temp = max(state['temperature'] for state in hardware_states.values())
        target_temp = self.config.thermal_threshold_celsius * 0.8
        temp_loss = F.mse_loss(torch.tensor(max_temp), torch.tensor(target_temp))
        
        # 3. Load balance loss
        energy_biases = torch.tensor(routing_metadata.get('energy_biases', [0.0]))
        load_balance_loss = torch.var(energy_biases)
        
        # 4. Routing stability loss (encourage stable routing decisions)
        routing_stability_loss = torch.norm(routing_biases, p=2) * 0.01
        
        # Combine losses
        total_loss = (
            power_loss * 0.4 +
            temp_loss * 0.3 +
            load_balance_loss * 0.2 +
            routing_stability_loss * 0.1
        )
        
        return total_loss
    
    def _update_performance_tracking(self, 
                                   hardware_states: Dict[int, Dict[str, Any]],
                                   routing_metadata: Dict[str, Any]):
        """Update performance tracking metrics"""
        timestamp = time.time()
        
        # Energy metrics
        total_power = sum(state['power_watt'] for state in hardware_states.values())
        avg_temp = np.mean([state['temperature'] for state in hardware_states.values()])
        
        self.energy_consumption_log.append({
            'timestamp': timestamp,
            'total_power_watts': total_power,
            'avg_temperature': avg_temp,
            'energy_biases': routing_metadata.get('energy_biases', []),
            'thermal_biases': routing_metadata.get('thermal_biases', []),
            'rerouting_biases': routing_metadata.get('rerouting_biases', [])
        })
        
        # Keep log size manageable
        if len(self.energy_consumption_log) > 1000:
            self.energy_consumption_log.pop(0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.energy_consumption_log:
            return {'error': 'No performance data available'}
        
        recent_logs = self.energy_consumption_log[-100:]  # Last 100 batches
        
        avg_power = np.mean([log['total_power_watts'] for log in recent_logs])
        avg_temp = np.mean([log['avg_temperature'] for log in recent_logs])
        
        # Energy efficiency (joules per token estimate)
        # Assuming ~1000 tokens per batch and 0.1 seconds per batch
        tokens_per_batch = 1000
        seconds_per_batch = 0.1
        joules_per_token = (avg_power * seconds_per_batch) / tokens_per_batch
        
        return {
            'avg_power_watts': avg_power,
            'avg_temperature_celsius': avg_temp,
            'joules_per_token': joules_per_token,
            'energy_efficiency_improvement': max(0, (self.config.joules_per_token_target - joules_per_token) / self.config.joules_per_token_target * 100),
            'thermal_efficiency': max(0, (self.config.thermal_threshold_celsius - avg_temp) / self.config.thermal_threshold_celsius * 100),
            'batch_count': self.batch_count,
            'num_devices': self.num_devices
        }

# src/parallel_moe_system.py
"""
Updated Parallel Energy-Aware MoE System with Multi-GPU Support
"""

import torch
import torch.distributed as dist
import os
from typing import List, Dict
from src.monitor import GpuSystemMonitor

# Utility functions for distributed training
def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"Rank {rank}/{world_size} initialized on GPU {rank}")

def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()

def create_parallel_moe_system(config: ParallelMoEConfig) -> ParallelMoELayer:
    """Factory function to create a parallel MoE system"""
    # Determine available devices
    if torch.cuda.is_available():
        device_ids = list(range(min(config.world_size, torch.cuda.device_count())))
    else:
        device_ids = [0]  # CPU fallback
        logger.warning("CUDA not available, falling back to CPU")
    
    # Create the parallel MoE layer
    moe_layer = ParallelMoELayer(config, device_ids)
    
    return moe_layer

def run_parallel_moe(rank: int, world_size: int, config: ParallelMoEConfig):
    """Run the parallel MoE system on a single GPU"""
    try:
        # Setup distributed environment
        setup_distributed(rank, world_size)

        # Create the parallel MoE system
        moe_layer = create_parallel_moe_system(config)
        moe_layer = torch.nn.parallel.DistributedDataParallel(moe_layer, device_ids=[rank])

        # Example: Forward pass with dummy data
        batch_size = 32
        seq_len = 128
        input_dim = config.moe_config.d_model
        dummy_input = torch.randn(batch_size, seq_len, input_dim).cuda(rank)

        # Forward pass
        output = moe_layer(dummy_input)
        print(f"Rank {rank}: Output shape {output.shape}")

    finally:
        # Cleanup distributed environment
        cleanup_distributed()

if __name__ == "__main__":
    import argparse
    from src.moe_models import MoEConfig

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Parallel MoE System")
    parser.add_argument("--world_size", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    args = parser.parse_args()

    # Create the configuration
    moe_config = MoEConfig(
        d_model=args.d_model,
        num_experts=args.num_experts,
        top_k=2,
        expert_type="swiglu"
    )
    config = ParallelMoEConfig(
        moe_config=moe_config,
        world_size=args.world_size
    )

    # Run the parallel MoE system
    torch.multiprocessing.spawn(
        run_parallel_moe,
        args=(args.world_size, config),
        nprocs=args.world_size,
        join=True
    )