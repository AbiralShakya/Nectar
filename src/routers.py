# src/routers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import time
import math
from abc import ABC, abstractmethod
import logging
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    BASELINE = "baseline"
    STATIC_OPTIMAL = "static_optimal"
    KERNEL_AWARE_TTHA = "kernel_aware_ttha"
    HIERARCHICAL_ADAPTIVE = "hierarchical_adaptive"
    PREDICTIVE_THERMAL = "predictive_thermal"
    MULTI_GPU_AWARE = "multi_gpu_aware"

@dataclass
class HardwareMetrics:
    """Comprehensive hardware metrics structure"""
    timestamp: float
    gpu_id: int
    temperature: float
    power_watts: float
    utilization: float
    memory_used: float
    memory_total: float
    clock_speed: int
    fan_speed: int
    thermal_throttling: bool
    power_limit: float
    voltage: float
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert metrics to tensor for ML processing"""
        return torch.tensor([
            self.temperature, self.power_watts, self.utilization,
            self.memory_used / self.memory_total, self.clock_speed / 2000.0,
            float(self.thermal_throttling), self.power_limit, self.voltage
        ], device=device, dtype=torch.float32)

@dataclass
class ExpertProfile:
    """Detailed expert computational profile"""
    expert_id: int
    operations: Dict[str, Dict[str, float]]  # op_name -> {energy, latency, flops, memory}
    specialization_score: float  # How specialized this expert is
    load_balancing_weight: float  # For load balancing
    thermal_sensitivity: float  # How much this expert affects temperature
    memory_footprint: int  # Peak memory usage
    cache_efficiency: float  # L1/L2 cache hit rates
    
class KernelCostModel:
    """Advanced kernel cost modeling with multi-dimensional profiling"""
    
    def __init__(self, profile_data_path: str = None):
        self.cost_database = {}
        self.interpolation_cache = {}
        self.thermal_coefficients = {}
        self.memory_access_patterns = {}
        self._load_profiles(profile_data_path)
    
    def _load_profiles(self, path: str):
        """Load comprehensive kernel profiles from disk"""
        # In production, this would load from extensive profiling data
        # For now, we'll use sophisticated synthetic profiles
        operations = ['linear_fc1', 'linear_fc2', 'relu', 'gelu', 'layernorm', 
                     'attention_qkv', 'attention_out', 'softmax']
        
        for op in operations:
            self.cost_database[op] = {}
            for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
                # Sophisticated cost modeling with non-linear scaling
                base_energy = self._calculate_base_energy(op, batch_size)
                base_latency = self._calculate_base_latency(op, batch_size)
                
                self.cost_database[op][batch_size] = {
                    'energy_joules': base_energy,
                    'latency_ms': base_latency,
                    'temp_impact': base_energy * 0.1,  # Thermal impact
                    'memory_bandwidth': self._calculate_memory_bandwidth(op, batch_size),
                    'cache_misses': self._calculate_cache_misses(op, batch_size),
                    'register_pressure': self._calculate_register_pressure(op, batch_size)
                }
    
    def _calculate_base_energy(self, op: str, batch_size: int) -> float:
        """Calculate base energy consumption with realistic scaling"""
        base_costs = {
            'linear_fc1': 0.5, 'linear_fc2': 0.3, 'relu': 0.05, 'gelu': 0.08,
            'layernorm': 0.15, 'attention_qkv': 0.8, 'attention_out': 0.6, 'softmax': 0.2
        }
        
        base = base_costs.get(op, 0.1)
        # Non-linear scaling with memory hierarchy effects
        scaling = batch_size ** 0.9 if batch_size <= 64 else 64 ** 0.9 * (batch_size / 64) ** 1.2
        return base * scaling
    
    def _calculate_base_latency(self, op: str, batch_size: int) -> float:
        """Calculate base latency with memory access patterns"""
        base_latencies = {
            'linear_fc1': 0.1, 'linear_fc2': 0.08, 'relu': 0.01, 'gelu': 0.02,
            'layernorm': 0.03, 'attention_qkv': 0.15, 'attention_out': 0.12, 'softmax': 0.05
        }
        
        base = base_latencies.get(op, 0.02)
        # Memory bandwidth limited scaling
        scaling = min(batch_size, 128) ** 0.7 + max(0, batch_size - 128) ** 1.1
        return base * scaling
    
    def _calculate_memory_bandwidth(self, op: str, batch_size: int) -> float:
        """Calculate memory bandwidth requirements"""
        if 'linear' in op:
            return batch_size * 4096 * 4  # 4KB per token, 4 bytes per float
        elif 'attention' in op:
            return batch_size * batch_size * 768 * 4  # Attention matrix
        else:
            return batch_size * 768 * 4  # Standard activation
    
    def _calculate_cache_misses(self, op: str, batch_size: int) -> float:
        """Estimate cache miss rate"""
        if batch_size <= 32:
            return 0.05  # Good cache locality
        elif batch_size <= 128:
            return 0.15  # Moderate cache pressure
        else:
            return 0.35  # High cache pressure
    
    def _calculate_register_pressure(self, op: str, batch_size: int) -> float:
        """Estimate register pressure"""
        if 'attention' in op:
            return min(1.0, batch_size / 64.0)
        else:
            return min(0.8, batch_size / 128.0)
    
    def get_cost(self, op_type: str, batch_size: int, 
                 current_temp: float = 70.0, current_power: float = 150.0) -> Dict[str, float]:
        """Get cost with thermal and power state adjustments"""
        if op_type not in self.cost_database:
            return {'energy_joules': 0.0, 'latency_ms': 0.0, 'temp_impact': 0.0}
        
        # Interpolate between profiled batch sizes
        costs = self._interpolate_costs(op_type, batch_size)
        
        # Adjust for current thermal state
        temp_factor = 1.0 + (current_temp - 70.0) * 0.02  # 2% increase per degree
        power_factor = 1.0 + (current_power - 150.0) * 0.001  # 0.1% increase per watt
        
        adjusted_costs = {}
        for key, value in costs.items():
            if key in ['energy_joules', 'latency_ms']:
                adjusted_costs[key] = value * temp_factor * power_factor
            else:
                adjusted_costs[key] = value
        
        return adjusted_costs
    
    def _interpolate_costs(self, op_type: str, batch_size: int) -> Dict[str, float]:
        """Interpolate costs for non-profiled batch sizes"""
        cache_key = (op_type, batch_size)
        if cache_key in self.interpolation_cache:
            return self.interpolation_cache[cache_key]
        
        available_sizes = sorted(self.cost_database[op_type].keys())
        
        if batch_size in available_sizes:
            result = self.cost_database[op_type][batch_size]
        else:
            # Find bounding sizes for interpolation
            lower = max([s for s in available_sizes if s <= batch_size], default=available_sizes[0])
            upper = min([s for s in available_sizes if s >= batch_size], default=available_sizes[-1])
            
            if lower == upper:
                result = self.cost_database[op_type][lower]
            else:
                # Linear interpolation
                alpha = (batch_size - lower) / (upper - lower)
                lower_costs = self.cost_database[op_type][lower]
                upper_costs = self.cost_database[op_type][upper]
                
                result = {}
                for key in lower_costs:
                    result[key] = lower_costs[key] * (1 - alpha) + upper_costs[key] * alpha
        
        self.interpolation_cache[cache_key] = result
        return result

class GpuSystemMonitor:
    """Advanced GPU system monitoring with predictive capabilities"""
    
    def __init__(self, num_gpus: int = 1, history_size: int = 1000):
        self.num_gpus = num_gpus
        self.history_size = history_size
        self.metrics_history = {i: deque(maxlen=history_size) for i in range(num_gpus)}
        self.prediction_models = {}
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self._initialize_monitoring()
    
    def _initialize_monitoring(self):
        """Initialize background monitoring thread"""
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                for gpu_id in range(self.num_gpus):
                    metrics = self._collect_gpu_metrics(gpu_id)
                    self.metrics_history[gpu_id].append(metrics)
                
                time.sleep(0.1)  # 10Hz monitoring
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_gpu_metrics(self, gpu_id: int) -> HardwareMetrics:
        """Collect comprehensive GPU metrics"""
        # In production, this would use nvidia-ml-py or similar
        # For now, we'll simulate realistic metrics
        current_time = time.time()
        base_temp = 65 + np.sin(current_time * 0.1) * 10  # Oscillating temperature
        base_power = 150 + np.random.normal(0, 10)  # Power with noise
        
        return HardwareMetrics(
            timestamp=current_time,
            gpu_id=gpu_id,
            temperature=max(30, base_temp + np.random.normal(0, 2)),
            power_watts=max(50, base_power),
            utilization=np.random.uniform(0.7, 0.95),
            memory_used=np.random.uniform(8000, 12000),
            memory_total=24000,
            clock_speed=int(1800 + np.random.normal(0, 50)),
            fan_speed=int(2000 + (base_temp - 65) * 20),
            thermal_throttling=base_temp > 83,
            power_limit=300.0,
            voltage=1.05 + np.random.normal(0, 0.02)
        )
    
    def get_current_stats(self, gpu_id: int = 0) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if gpu_id not in self.metrics_history or not self.metrics_history[gpu_id]:
            return self._get_default_stats()
        
        latest = self.metrics_history[gpu_id][-1]
        return {
            'temperature': latest.temperature,
            'power_watt': latest.power_watts,
            'utilization': latest.utilization,
            'memory_usage': latest.memory_used / latest.memory_total,
            'thermal_throttling': latest.thermal_throttling,
            'timestamp': latest.timestamp
        }
    
    def _get_default_stats(self) -> Dict[str, Any]:
        """Default stats when monitoring not available"""
        return {
            'temperature': 70.0,
            'power_watt': 150.0,
            'utilization': 0.8,
            'memory_usage': 0.5,
            'thermal_throttling': False,
            'timestamp': time.time()
        }
    
    def predict_thermal_trajectory(self, gpu_id: int, horizon_seconds: int = 30) -> List[float]:
        """Predict future temperature trajectory"""
        if gpu_id not in self.metrics_history or len(self.metrics_history[gpu_id]) < 10:
            current_temp = self.get_current_stats(gpu_id)['temperature']
            return [current_temp] * (horizon_seconds // 5)  # Flat prediction
        
        # Simple linear extrapolation (in production, use more sophisticated models)
        recent_temps = [m.temperature for m in list(self.metrics_history[gpu_id])[-10:]]
        trend = (recent_temps[-1] - recent_temps[0]) / len(recent_temps)
        
        predictions = []
        current_temp = recent_temps[-1]
        for i in range(horizon_seconds // 5):
            current_temp += trend
            predictions.append(max(30, min(95, current_temp)))  # Clamp to reasonable range
        
        return predictions
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score"""
        total_score = 0.0
        for gpu_id in range(self.num_gpus):
            stats = self.get_current_stats(gpu_id)
            
            # Temperature score (0-1, lower is better)
            temp_score = max(0, min(1, (stats['temperature'] - 60) / 25))
            
            # Power score (0-1, lower is better)
            power_score = max(0, min(1, (stats['power_watt'] - 100) / 200))
            
            # Thermal throttling penalty
            throttling_penalty = 0.5 if stats['thermal_throttling'] else 0.0
            
            gpu_score = 1.0 - (temp_score * 0.4 + power_score * 0.4 + throttling_penalty * 0.2)
            total_score += gpu_score
        
        return total_score / self.num_gpus
    
    def shutdown(self):
        """Shutdown monitoring thread"""
        self.shutdown_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

class TTHAAdapter(nn.Module):
    """Advanced Test-Time Hardware-Efficiency Adaptation module"""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        # Multi-head attention for processing heterogeneous inputs
        self.input_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Separate processing branches for different input types
        self.cost_processor = nn.Sequential(
            nn.Linear(num_experts * 4, hidden_dim),  # 4 cost metrics per expert
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.hardware_processor = nn.Sequential(
            nn.Linear(8, hidden_dim),  # 8 hardware metrics
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.temporal_processor = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim//2, 
            num_layers=2, batch_first=True, dropout=0.1
        )
        
        # Fusion and output layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Tanh()  # Bounded output
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, num_experts),
            nn.Softplus()  # Positive uncertainty values
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, cost_features: torch.Tensor, hardware_features: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-modal input processing
        
        Args:
            cost_features: Expert cost features [batch_size, num_experts * 4]
            hardware_features: Hardware metrics [batch_size, 8]
            temporal_features: Temporal context [batch_size, seq_len, hidden_dim]
        
        Returns:
            routing_biases: Bias adjustments for each expert
            uncertainties: Uncertainty estimates for each bias
        """
        batch_size = cost_features.size(0)
        
        # Process different input modalities
        cost_embed = self.cost_processor(cost_features)
        hardware_embed = self.hardware_processor(hardware_features)
        
        # Temporal processing
        if temporal_features is not None:
            temporal_embed, _ = self.temporal_processor(temporal_features)
            temporal_embed = temporal_embed[:, -1, :]  # Use last timestep
        else:
            temporal_embed = torch.zeros_like(cost_embed)
        
        # Attention-based fusion
        combined_features = torch.stack([cost_embed, hardware_embed, temporal_embed], dim=1)
        attended_features, _ = self.input_attention(
            combined_features, combined_features, combined_features
        )
        
        # Flatten and fuse
        fused_features = self.fusion_layer(attended_features.view(batch_size, -1))
        
        # Generate outputs
        routing_biases = self.output_layer(fused_features)
        uncertainties = self.uncertainty_head(fused_features)
        
        return routing_biases, uncertainties

class AdaptiveRouter(nn.Module):
    """Production-grade adaptive router with advanced features"""
    
    def __init__(self, 
                 num_experts: int,
                 top_k: int,
                 kernel_cost_model: KernelCostModel,
                 gpu_system_monitor: GpuSystemMonitor,
                 strategy: Union[str, RoutingStrategy] = RoutingStrategy.BASELINE,
                 device_topology: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        self.strategy = RoutingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.device_topology = device_topology or {}
        
        # Advanced routing components
        self.expert_profiles = self._initialize_expert_profiles()
        self.load_balancer = ExpertLoadBalancer(num_experts)
        self.thermal_predictor = ThermalPredictor()
        
        # TTHA components
        self.ttha_adapter = None
        self.ttha_optimizer = None
        self.ttha_scheduler = None
        self.ttha_history = defaultdict(list)
        
        # Multi-objective optimization
        self.objective_weights = {
            'performance': 0.4,
            'energy': 0.3,
            'thermal': 0.2,
            'load_balance': 0.1
        }
        
        # Caching and optimization
        self.bias_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Performance monitoring
        self.routing_latencies = deque(maxlen=1000)
        self.routing_decisions = deque(maxlen=10000)
        
        if self.strategy in [RoutingStrategy.KERNEL_AWARE_TTHA, RoutingStrategy.HIERARCHICAL_ADAPTIVE]:
            self._initialize_ttha()
        
        logger.info(f"Initialized AdaptiveRouter with strategy: {self.strategy}")
    
    def _initialize_expert_profiles(self) -> List[ExpertProfile]:
        """
        Initialize detailed expert profiles by querying the KernelCostModel
        for the costs of each expert's constituent operations.
        Assumes experts are structurally homogeneous for initial profiling.
        """
        profiles = []
        # Define the set of operations that make up a single expert's forward pass
        # These MUST match the op_types you profiled in expert_kernel_profiler.py
        expert_op_types = ["dequant_unpack_op", "linear_fc1", "relu", "linear_fc2"] 
        
        # We need a reference batch size for these static profiles.
        # For simplicity, use an "effective_kernel_lookup_batch_size" of 1,
        # assuming costs scale from there, or a representative average.
        # For router's expert_profiles, these are generic intrinsic costs.
        reference_batch_size_for_profile = 1 

        # Get a dummy/current hardware state for cost adjustment in KCM.
        # This makes the base profile influenced by "typical" conditions.
        current_gpu_stats = self.gpu_system_monitor.get_current_stats()
        current_temp = current_gpu_stats['temperature']
        current_power = current_gpu_stats['power_watt']

        for expert_id in range(self.num_experts):
            expert_operations_costs = {}
            total_expert_energy = 0.0
            total_expert_latency = 0.0
            
            for op_name in expert_op_types:
                # Query KernelCostModel for the cost of each operation
                # Pass current_temp/power for context-aware cost lookup if KCM supports it.
                op_costs = self.kernel_cost_model.get_cost(
                    op_name, reference_batch_size_for_profile,
                    current_temp=current_temp, current_power=current_power
                )
                
                expert_operations_costs[op_name] = {
                    'energy': op_costs.get('energy_joules', 0.0),
                    'latency': op_costs.get('latency_ms', 0.0),
                    'temp_impact': op_costs.get('temp_impact', 0.0),
                    # Add other metrics if available from KCM and relevant for ExpertProfile
                    'memory': op_costs.get('memory_bandwidth', 0.0), # Reusing 'memory' key for bandwidth
                    'flops': op_costs.get('flops', 0.0), # If you profiled FLOPs
                    'cache_misses': op_costs.get('cache_misses', 0.0) # If you profiled
                }
                total_expert_energy += op_costs.get('energy_joules', 0.0)
                total_expert_latency += op_costs.get('latency_ms', 0.0)

            # Assign properties based on aggregated costs or simulated values
            # These can be refined later based on actual expert behavior analysis.
            # For now, derive from total energy/latency
            specialization = np.random.uniform(0.6, 0.9) # Still dummy for now
            thermal_sensitivity = total_expert_energy * 0.1 # More energy, more sensitive
            cache_efficiency = 1.0 - (total_expert_energy / (total_expert_latency + 1e-5)) * 0.01 # Dummy inverse relation
            memory_footprint = int(total_expert_energy * 1000) # Dummy relation

            profile = ExpertProfile(
                expert_id=expert_id,
                operations=expert_operations_costs, # Use the actual profiled costs
                specialization_score=specialization,
                load_balancing_weight=1.0, # Default, adjusted by LoadBalancer
                thermal_sensitivity=thermal_sensitivity,
                memory_footprint=memory_footprint,
                cache_efficiency=cache_efficiency
            )
            profiles.append(profile)
        
        return profiles
    
    def _initialize_ttha(self):
        """Initialize TTHA adaptation components"""
        input_dim = self.num_experts * 4 + 8  # 4 metrics per expert + 8 hardware metrics
        self.ttha_adapter = TTHAAdapter(input_dim, self.num_experts)
        
        self.ttha_optimizer = torch.optim.AdamW(
            self.ttha_adapter.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        self.ttha_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.ttha_optimizer, T_0=100, T_mult=2
        )
        
        # Exponential moving averages for stable training
        self.ema_power_loss = 0.0
        self.ema_temp_loss = 0.0
        self.ema_latency_penalty = 0.0
        self.ema_decay = 0.99
    
    def forward(self, 
                gate_logits: torch.Tensor,
                current_batch_size: int,
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Advanced forward pass with comprehensive routing logic
        
        Args:
            gate_logits: Gating network outputs [num_tokens, num_experts]
            current_batch_size: Actual batch size for kernel cost lookup
            context: Additional context information
        
        Returns:
            topk_indices: Selected expert indices
            routing_weights: Routing weights
            routing_info: Additional routing information
        """
        start_time = time.time()
        device = gate_logits.device
        num_tokens = gate_logits.size(0)
        
        # Get base cost biases
        base_cost_biases = self._compute_base_cost_biases(device, current_batch_size)
        
        # Apply strategy-specific routing
        if self.strategy == RoutingStrategy.BASELINE:
            final_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)
        
        elif self.strategy == RoutingStrategy.STATIC_OPTIMAL:
            final_biases = base_cost_biases
        
        elif self.strategy == RoutingStrategy.KERNEL_AWARE_TTHA:
            final_biases = self._compute_ttha_biases(base_cost_biases, device)
        
        elif self.strategy == RoutingStrategy.HIERARCHICAL_ADAPTIVE:
            final_biases = self._compute_hierarchical_biases(base_cost_biases, device, context)
        
        elif self.strategy == RoutingStrategy.PREDICTIVE_THERMAL:
            final_biases = self._compute_predictive_thermal_biases(base_cost_biases, device)
        
        elif self.strategy == RoutingStrategy.MULTI_GPU_AWARE:
            final_biases = self._compute_multi_gpu_biases(base_cost_biases, device)
        
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")
        
        # Apply load balancing
        load_balance_biases = self.load_balancer.get_balancing_biases(device)
        final_biases += load_balance_biases
        
        # Compute final routing
        biased_logits = gate_logits + final_biases.unsqueeze(0)  # Broadcast to all tokens
        topk_vals, topk_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_vals, dim=-1)
        
        # Update load balancer
        self.load_balancer.update_loads(topk_indices, routing_weights)
        
        # Collect routing information
        routing_latency = time.time() - start_time
        self.routing_latencies.append(routing_latency)
        
        routing_info = {
            'routing_latency': routing_latency,
            'strategy': self.strategy.value,
            'base_cost_biases': base_cost_biases.detach().cpu(),
            'final_biases': final_biases.detach().cpu(),
            'load_balance_biases': load_balance_biases.detach().cpu(),
            'system_health': self.gpu_system_monitor.get_system_health_score()
        }
        
        return topk_indices, routing_weights, routing_info
    
    def _compute_base_cost_biases(self, device: torch.device, batch_size: int) -> torch.Tensor:
        """Compute base cost biases with caching"""
        cache_key = (device.type, batch_size)
        if cache_key in self.bias_cache:
            self.cache_hit_count += 1
            return self.bias_cache[cache_key]
        
        self.cache_miss_count += 1
        
        # Get current hardware state for cost adjustment
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        current_temp = gpu_stats['temperature']
        current_power = gpu_stats['power_watt']
        
        biases = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        
        for expert_id in range(self.num_experts):
            total_energy = 0.0
            total_temp_impact = 0.0
            
            profile = self.expert_profiles[expert_id]
            
            for op_name, op_metrics in profile.operations.items():
                cost_data = self.kernel_cost_model.get_cost(
                    op_name, batch_size, current_temp, current_power
                )
                
                total_energy += cost_data.get('energy_joules', 0.0)
                total_temp_impact += cost_data.get('temp_impact', 0.0)
            
            # Multi-objective bias calculation
            energy_bias = -total_energy * self.objective_weights['energy'] * 100.0
            thermal_bias = -total_temp_impact * self.objective_weights['thermal'] * 50.0
            
            # Adjust for expert-specific characteristics
            thermal_bias *= profile.thermal_sensitivity
            
            biases[expert_id] = energy_bias + thermal_bias
        
        # Cache the result
        self.bias_cache[cache_key] = biases
        
        # Limit cache size
        if len(self.bias_cache) > 100:
            oldest_key = next(iter(self.bias_cache))
            del self.bias_cache[oldest_key]
        
        return biases
    
    def _compute_ttha_biases(self, base_biases: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute TTHA-based dynamic biases"""
        if self.ttha_adapter is None:
            return base_biases
        
        # Prepare cost features
        cost_features = torch.zeros(1, self.num_experts * 4, device=device)
        for i, profile in enumerate(self.expert_profiles):
            base_idx = i * 4
            cost_features[0, base_idx:base_idx+4] = torch.tensor([
                sum(op['energy'] for op in profile.operations.values()),
                sum(op['latency'] for op in profile.operations.values()),
                profile.thermal_sensitivity,
                profile.cache_efficiency
            ], device=device)
        
        # Prepare hardware features
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        hardware_metrics = HardwareMetrics(
            timestamp=gpu_stats['timestamp'],
            gpu_id=0,
            temperature=gpu_stats['temperature'],
            power_watts=gpu_stats['power_watt'],
            utilization=gpu_stats['utilization'],
            memory_used=gpu_stats.get('memory_usage', 0.5) * 24000,
            memory_total=24000,
            clock_speed=1800,
            fan_speed=2000,
            thermal_throttling=gpu_stats.get('thermal_throttling', False),
            power_limit=300.0,
            voltage=1.05
        )
        
        hardware_features = hardware_metrics.to_tensor(device).unsqueeze(0)
        
        # Get TTHA predictions
        with torch.no_grad():
            dynamic_biases, uncertainties = self.ttha_adapter(
                cost_features, hardware_features
            )
        
        # Combine base biases with dynamic adjustments
        combined_biases = base_biases + dynamic_biases.squeeze(0) * 0.5  # Scale dynamic component
        
        return combined_biases
    
    def _compute_hierarchical_biases(self, base_biases: torch.Tensor, 
                                   device: torch.device, 
                                   context: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Compute hierarchical adaptive biases"""
        # Start with TTHA biases
        ttha_biases = self._compute_ttha_biases(base_biases, device)
        
        # Add hierarchical adjustments based on context
        if context:
            seq_length = context.get('sequence_length', 512)
            task_type = context.get('task_type', 'general')
            urgency = context.get('urgency', 0.5)
            
            # Sequence length adjustments
            if seq_length > 1024:
                # Favor more efficient experts for long sequences
                efficiency_scores = torch.tensor([
                    profile.cache_efficiency for profile in self.expert_profiles
                ], device=device)
                ttha_biases += efficiency_scores * 0.2
            
            # Task-specific adjustments
            if task_type == 'code_generation':
                # Favor experts with better specialization for code
                spec_scores = torch.tensor([
                    profile.specialization_score for profile in self.expert_profiles
                ], device=device)
                ttha_biases += spec_scores * 0.3
            
            # Urgency adjustments
            if urgency > 0.8:
                # Prioritize low-latency experts for urgent requests
                latency_scores = torch.tensor([
                    -sum(op['latency'] for op in profile.operations.values())
                    for profile in self.expert_profiles
                ], device=device)
                ttha_biases += latency_scores * urgency * 0.4
        
        return ttha_biases
    
    def _compute_predictive_thermal_biases(self, base_biases: torch.Tensor, 
                                         device: torch.device) -> torch.Tensor:
        """Compute biases based on predicted thermal trajectory"""
        # Predict future temperatures
        thermal_predictions = self.gpu_system_monitor.predict_thermal_trajectory(0, 30)
        
        if thermal_predictions:
            max_predicted_temp = max(thermal_predictions)
            
            if max_predicted_temp > 80.0:  # Thermal concern threshold
                # Heavily bias against thermally sensitive experts
                thermal_penalties = torch.tensor([
                    -profile.thermal_sensitivity * (max_predicted_temp - 70.0) * 0.1
                    for profile in self.expert_profiles
                ], device=device)
                
                return base_biases + thermal_penalties
        
        return base_biases
    
    def _compute_multi_gpu_biases(self, base_biases: torch.Tensor, 
                                device: torch.device) -> torch.Tensor:
        """Compute biases for multi-GPU scenarios"""
        if self.gpu_system_monitor.num_gpus == 1:
            return base_biases
        
        # Get stats from all GPUs
        gpu_health_scores = []
        for gpu_id in range(self.gpu_system_monitor.num_gpus):
            stats = self.gpu_system_monitor.get_current_stats(gpu_id)
            
            # Calculate health score for this GPU
            temp_score = max(0, 1 - (stats['temperature'] - 60) / 25)
            power_score = max(0, 1 - (stats['power_watt'] - 100) / 200)
            util_score = 1 - stats['utilization']  # Lower utilization is better
            
            health_score = (temp_score + power_score + util_score) / 3
            gpu_health_scores.append(health_score)
        
        # Bias towards experts on healthier GPUs
        # This is a simplified example - in practice, you'd need expert-to-GPU mapping
        avg_health = sum(gpu_health_scores) / len(gpu_health_scores)
        health_bonus = torch.full_like(base_biases, avg_health * 0.2)
        
        return base_biases + health_bonus
    
    def update_ttha(self, 
                    observed_metrics: Dict[str, float],
                    target_power: float = 150.0,
                    target_temp: float = 70.0,
                    latency_penalty_weight: float = 0.1) -> Dict[str, float]:
        """
        Advanced TTHA update with multi-objective optimization
        
        Args:
            observed_metrics: Dictionary of observed hardware metrics
            target_power: Target power consumption in watts
            target_temp: Target temperature in Celsius
            latency_penalty_weight: Weight for latency penalty
        
        Returns:
            Dictionary of loss components
        """
        if self.strategy not in [RoutingStrategy.KERNEL_AWARE_TTHA, RoutingStrategy.HIERARCHICAL_ADAPTIVE]:
            return {}
        
        if self.ttha_adapter is None or self.ttha_optimizer is None:
            return {}
        
        device = next(self.ttha_adapter.parameters()).device
        
        # Extract observed metrics
        observed_power = observed_metrics.get('gpu_power_watt', target_power)
        observed_temp = observed_metrics.get('gpu_temperature_c', target_temp)
        observed_latency = observed_metrics.get('inference_latency_ms', 0.0)
        observed_throughput = observed_metrics.get('throughput_tokens_per_sec', 1000.0)
        
        # Calculate loss components
        power_loss = max(0, observed_power - target_power) ** 2 / (target_power ** 2)
        temp_loss = max(0, observed_temp - target_temp) ** 2 / (target_temp ** 2)
        
        # Latency penalty (encourage staying close to baseline)
        if self.base_latency_for_penalty > 0:
            latency_penalty = max(0, observed_latency - self.base_latency_for_penalty) ** 2
            latency_penalty /= (self.base_latency_for_penalty ** 2)
        else:
            latency_penalty = 0.0
        
        # Throughput bonus (reward higher throughput)
        throughput_bonus = min(0.1, observed_throughput / 10000.0)  # Cap at 0.1
        
        # Update exponential moving averages
        self.ema_power_loss = self.ema_decay * self.ema_power_loss + (1 - self.ema_decay) * power_loss
        self.ema_temp_loss = self.ema_decay * self.ema_temp_loss + (1 - self.ema_decay) * temp_loss
        self.ema_latency_penalty = self.ema_decay * self.ema_latency_penalty + (1 - self.ema_decay) * latency_penalty
        
        # Combined loss with adaptive weighting
        total_loss = (
            self.ema_power_loss * self.objective_weights['energy'] +
            self.ema_temp_loss * self.objective_weights['thermal'] +
            self.ema_latency_penalty * latency_penalty_weight -
            throughput_bonus * 0.1
        )
        
        # Perform gradient update
        self.ttha_optimizer.zero_grad()
        
        # Create a differentiable loss tensor
        loss_tensor = torch.tensor(total_loss, device=device, dtype=torch.float32, requires_grad=True)
        
        # Simulate gradient computation (in practice, this would come from actual forward pass)
        # This is a simplified version - real implementation would backpropagate through routing decisions
        dummy_input = torch.randn(1, self.num_experts * 4, device=device, requires_grad=True)
        dummy_hardware = torch.randn(1, 8, device=device, requires_grad=True)
        
        biases, uncertainties = self.ttha_adapter(dummy_input, dummy_hardware)
        
        # Regularization losses
        l2_reg = sum(p.pow(2).sum() for p in self.ttha_adapter.parameters()) * 1e-6
        uncertainty_reg = uncertainties.mean() * 0.01  # Encourage confident predictions
        
        total_model_loss = loss_tensor + l2_reg + uncertainty_reg
        total_model_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.ttha_adapter.parameters(), max_norm=1.0)
        
        self.ttha_optimizer.step()
        self.ttha_scheduler.step()
        
        # Record history
        loss_components = {
            'total_loss': float(total_loss),
            'power_loss': float(power_loss),
            'temp_loss': float(temp_loss),
            'latency_penalty': float(latency_penalty),
            'throughput_bonus': float(throughput_bonus),
            'l2_reg': float(l2_reg),
            'uncertainty_reg': float(uncertainty_reg.item()),
            'learning_rate': self.ttha_scheduler.get_last_lr()[0]
        }
        
        for key, value in loss_components.items():
            self.ttha_history[key].append(value)
            
        # Limit history size
        max_history = 10000
        for key in self.ttha_history:
            if len(self.ttha_history[key]) > max_history:
                self.ttha_history[key] = self.ttha_history[key][-max_history:]
        
        return loss_components
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        stats = {
            'strategy': self.strategy.value,
            'cache_hit_rate': self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count),
            'avg_routing_latency_ms': np.mean(self.routing_latencies) * 1000 if self.routing_latencies else 0,
            'system_health_score': self.gpu_system_monitor.get_system_health_score(),
            'expert_utilization': self.load_balancer.get_utilization_stats(),
            'num_gpus': self.gpu_system_monitor.num_gpus
        }
        
        if self.ttha_history:
            stats['ttha_stats'] = {
                'avg_power_loss': np.mean(self.ttha_history['power_loss'][-100:]) if self.ttha_history['power_loss'] else 0,
                'avg_temp_loss': np.mean(self.ttha_history['temp_loss'][-100:]) if self.ttha_history['temp_loss'] else 0,
                'avg_latency_penalty': np.mean(self.ttha_history['latency_penalty'][-100:]) if self.ttha_history['latency_penalty'] else 0,
                'updates_performed': len(self.ttha_history.get('total_loss', []))
            }
        
        return stats
    
    def set_objective_weights(self, **weights):
        """Update objective function weights"""
        for key, value in weights.items():
            if key in self.objective_weights:
                self.objective_weights[key] = value
        
        # Normalize weights
        total_weight = sum(self.objective_weights.values())
        if total_weight > 0:
            for key in self.objective_weights:
                self.objective_weights[key] /= total_weight
    
    def save_state(self, filepath: str):
        """Save router state for persistence"""
        state = {
            'strategy': self.strategy.value,
            'objective_weights': self.objective_weights,
            'ttha_history': dict(self.ttha_history),
            'expert_profiles': self.expert_profiles,
            'base_latency_for_penalty': getattr(self, 'base_latency_for_penalty', 0.0)
        }
        
        if self.ttha_adapter is not None:
            state['ttha_adapter_state'] = self.ttha_adapter.state_dict()
            state['ttha_optimizer_state'] = self.ttha_optimizer.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Router state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load router state from file"""
        state = torch.load(filepath, map_location='cpu')
        
        self.strategy = RoutingStrategy(state['strategy'])
        self.objective_weights = state['objective_weights']
        self.ttha_history = defaultdict(list, state['ttha_history'])
        self.expert_profiles = state['expert_profiles']
        self.base_latency_for_penalty = state.get('base_latency_for_penalty', 0.0)
        
        if 'ttha_adapter_state' in state and self.ttha_adapter is not None:
            self.ttha_adapter.load_state_dict(state['ttha_adapter_state'])
            
        if 'ttha_optimizer_state' in state and self.ttha_optimizer is not None:
            self.ttha_optimizer.load_state_dict(state['ttha_optimizer_state'])
            
        logger.info(f"Router state loaded from {filepath}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.gpu_system_monitor, 'shutdown'):
            self.gpu_system_monitor.shutdown()


class ExpertLoadBalancer:
    """Advanced load balancing for MoE experts"""
    
    def __init__(self, num_experts: int, smoothing_factor: float = 0.9):
        self.num_experts = num_experts
        self.smoothing_factor = smoothing_factor
        self.expert_loads = np.ones(num_experts) / num_experts  # Initialize uniform
        self.expert_latencies = np.ones(num_experts) * 10.0  # Average latency per expert
        self.load_history = deque(maxlen=1000)
        
    def update_loads(self, expert_indices: torch.Tensor, routing_weights: torch.Tensor):
        """Update expert load statistics"""
        current_loads = np.zeros(self.num_experts)
        
        # Count token assignments
        for token_idx in range(expert_indices.size(0)):
            for k in range(expert_indices.size(1)):
                expert_id = expert_indices[token_idx, k].item()
                weight = routing_weights[token_idx, k].item()
                current_loads[expert_id] += weight
        
        # Normalize by number of tokens
        if expert_indices.size(0) > 0:
            current_loads /= expert_indices.size(0)
        
        # Exponential moving average
        self.expert_loads = (self.smoothing_factor * self.expert_loads + 
                           (1 - self.smoothing_factor) * current_loads)
        
        self.load_history.append(current_loads.copy())
    
    def get_balancing_biases(self, device: torch.device) -> torch.Tensor:
        """Get load balancing biases to encourage balanced expert usage"""
        target_load = 1.0 / self.num_experts
        load_deviations = self.expert_loads - target_load
        
        # Penalize overloaded experts, reward underloaded ones
        balancing_biases = -load_deviations * 2.0  # Scale factor
        
        return torch.tensor(balancing_biases, device=device, dtype=torch.float32)
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get expert utilization statistics"""
        if len(self.load_history) == 0:
            return {'balance_score': 1.0, 'std_dev': 0.0}
        
        recent_loads = np.array(list(self.load_history)[-100:])  # Last 100 batches
        avg_loads = np.mean(recent_loads, axis=0)
        
        # Calculate balance score (1.0 = perfectly balanced)
        target_load = 1.0 / self.num_experts
        deviations = np.abs(avg_loads - target_load)
        balance_score = max(0.0, 1.0 - np.mean(deviations) * self.num_experts)
        
        return {
            'balance_score': float(balance_score),
            'std_dev': float(np.std(avg_loads)),
            'max_load': float(np.max(avg_loads)),
            'min_load': float(np.min(avg_loads))
        }


class ThermalPredictor:
    """Predictive thermal modeling for proactive routing"""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.temperature_history = deque(maxlen=history_length)
        self.power_history = deque(maxlen=history_length)
        
    def update(self, temperature: float, power: float):
        """Update thermal history"""
        self.temperature_history.append(temperature)
        self.power_history.append(power)
    
    def predict_thermal_impact(self, expert_power_consumption: float, 
                             horizon_seconds: int = 30) -> float:
        """Predict thermal impact of routing decision"""
        if len(self.temperature_history) < 10:
            return 0.0  # Not enough data
        
        # Simple thermal model: temp_change ≈ k * power_change
        recent_temps = list(self.temperature_history)[-10:]
        recent_powers = list(self.power_history)[-10:]
        
        # Estimate thermal coefficient
        if len(set(recent_powers)) > 1:  # Avoid division by zero
            temp_range = max(recent_temps) - min(recent_temps)
            power_range = max(recent_powers) - min(recent_powers)
            thermal_coeff = temp_range / max(power_range, 1.0)
        else:
            thermal_coeff = 0.1  # Default assumption
        
        # Predict temperature increase
        predicted_temp_increase = thermal_coeff * expert_power_consumption
        
        # Time decay factor
        time_factor = min(1.0, horizon_seconds / 60.0)  # Linear up to 1 minute
        
        return predicted_temp_increase * time_factor


# Utility functions for advanced routing
def compute_pareto_efficiency(costs: np.ndarray) -> np.ndarray:
    """Compute Pareto-efficient solutions for multi-objective optimization"""
    n_points = costs.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    
    for i in range(n_points):
        if is_efficient[i]:
            # Remove dominated points
            dominated = np.all(costs >= costs[i], axis=1) & np.any(costs > costs[i], axis=1)
            is_efficient[dominated] = False
    
    return is_efficient


def create_router_factory(config: Dict[str, Any]) -> callable:
    """Factory function for creating configured routers"""
    def create_router(num_experts: int, top_k: int, 
                     kernel_cost_model: KernelCostModel,
                     gpu_system_monitor: GpuSystemMonitor) -> AdaptiveRouter:
        
        strategy = RoutingStrategy(config.get('strategy', 'baseline'))
        device_topology = config.get('device_topology', {})
        
        router = AdaptiveRouter(
            num_experts=num_experts,
            top_k=top_k,
            kernel_cost_model=kernel_cost_model,
            gpu_system_monitor=gpu_system_monitor,
            strategy=strategy,
            device_topology=device_topology
        )
        
        # Apply configuration
        if 'objective_weights' in config:
            router.set_objective_weights(**config['objective_weights'])
            
        return router
    
    return create_router


# Example usage and testing
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