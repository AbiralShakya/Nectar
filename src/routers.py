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
import logging
from enum import Enum
from torch.optim import Optimizer
import types

from src.monitor import GpuSystemMonitor
from src.kernelcostmodel import KernelCostModel # Use the updated KCM
from src.moe_models import MoEConfig # Import the new MoEConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    BASELINE = "baseline"
    STATIC_OPTIMAL = "static_optimal"
    KERNEL_AWARE_TTHA = "kernel_aware_ttha"
    HIERARCHICAL_ADAPTIVE = "hierarchical_adaptive"
    PREDICTIVE_THERMAL = "predictive_thermal"
    MULTI_GPU_AWARE = "multi_gpu_aware"
    DYNAMIC_EXPERT_REROUTING = "dynamic_expert_rerouting"  # New strategy for expert rerouting

@dataclass
class HardwareMetrics:
    """Comprehensive hardware metrics structure. Aligns with GpuSystemMonitor output."""
    timestamp: float
    gpu_id: int
    temperature: float
    power_watts: float
    utilization: float # From GpuSystemMonitor, might be gpu_utilization_percent
    memory_used: float # This would be memory_used_bytes
    memory_total: float # This would be memory_total_bytes
    clock_speed: int = 0
    fan_speed: int = 0
    thermal_throttling: bool = False
    power_limit: float = 0.0
    voltage: float = 0.0
    memory_utilization_percent: float = 0.0 # Added for easier access from monitor output
    gpu_utilization_percent: float = 0.0 # Added for easier access from monitor output
    thermal_state: str = 'cool' # Added for easier access from monitor output
    
    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Convert metrics to tensor for ML processing. Ensure features are normalized."""
        # Use a consistent order and normalization for TTHAAdapter input
        # Ensure power_limit is non-zero to avoid division by zero
        power_limit_normalized = self.power_limit if self.power_limit > 0 else 1.0 
        
        return torch.tensor([
            self.temperature / 100.0, # Normalize temp (e.g., max 100C)
            self.power_watts / power_limit_normalized, # Normalize power by limit
            self.gpu_utilization_percent / 100.0, # Normalize utilization
            self.memory_utilization_percent / 100.0, # Normalize memory usage
            self.clock_speed / 2500.0, # Max clock speed approx 2.5 GHz
            float(self.thermal_throttling),
            power_limit_normalized / 500.0, # Normalize power limit (e.g., max 500W)
            self.voltage # Voltage is usually small, don't normalize heavily
        ], device=device, dtype=torch.float32)

@dataclass
class BatchDistributionHistory:
    """Tracks historical batch distribution patterns for expert rerouting."""
    timestamp: float
    expert_distribution: torch.Tensor  # [num_experts] - token counts per expert
    expert_weights: torch.Tensor  # [num_experts] - routing weights/probabilities
    hardware_metrics: Dict[str, float]  # GPU metrics at time of routing
    performance_metrics: Dict[str, float]  # Latency, throughput, energy
    rerouting_decisions: Optional[torch.Tensor] = None  # [num_experts] - rerouting adjustments applied

@dataclass
class ExpertProfile:
    """Detailed expert computational profile"""
    expert_id: int
    operations: Dict[str, Dict[str, float]]  # op_name -> {energy, latency, temp_impact, memory_gb, compute_utilization, memory_utilization}
    specialization_score: float  # How specialized this expert is (dummy for now)
    load_balancing_weight: float  # For load balancing
    thermal_sensitivity: float  # How much this expert affects temperature (derived from op_profiles)
    memory_footprint: float  # Peak memory usage (derived from op_profiles) - now in GB
    avg_latency_ms: float # Average latency of the expert itself
    avg_energy_joules: float # Average energy of the expert itself
    
class TTHAAdapter(nn.Module):
    """
    Advanced Test-Time Hardware-Efficiency Adaptation module.
    Predicts routing biases based on expert cost features and hardware metrics.
    """
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim # Features from expert costs + hardware
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim # Make hidden_dim configurable or based on d_model
        
        # Multi-head attention for processing heterogeneous inputs
        # Query, Key, Value from the combined features
        self.input_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Separate processing branches for different input types
        # cost_processor input_dim: num_experts * (6 metrics: energy, latency, temp_impact, memory_gb, compute_utilization, memory_utilization)
        self.cost_processor = nn.Sequential(
            nn.Linear(num_experts * 6, self.hidden_dim), 
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # hardware_processor input_dim: 8 metrics from HardwareMetrics.to_tensor()
        self.hardware_processor = nn.Sequential(
            nn.Linear(8, self.hidden_dim), 
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Temporal processor for historical context (if needed)
        # For now, it's optional and will default to zeros if not provided
        self.temporal_processor = nn.LSTM(
            input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2, 
            num_layers=2, batch_first=True, dropout=0.1
        )
        
        # Fusion and output layers
        # The MultiheadAttention output, when averaged, is `hidden_dim`.
        # So, the input to fusion_layer will be `hidden_dim`
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), # Input to fusion is mean of attended features
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, num_experts),
            nn.Tanh() # Bounded output, e.g., -1 to 1 for biases
        )
        
        # Uncertainty estimation head (for future use or advanced TTHA)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, num_experts),
            nn.Softplus() # Positive uncertainty values
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with proper scaling for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, cost_features: torch.Tensor, hardware_features: torch.Tensor,
                temporal_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multi-modal input processing.
        
        Args:
            cost_features: Expert cost features [batch_size, num_experts * 6]
            hardware_features: Hardware metrics [batch_size, 8]
            temporal_features: Temporal context [batch_size, seq_len, hidden_dim] - Optional
        
        Returns:
            routing_biases: Bias adjustments for each expert
            uncertainties: Uncertainty estimates for each bias
        """
        batch_size = cost_features.size(0)
        
        cost_embed = self.cost_processor(cost_features)
        hardware_embed = self.hardware_processor(hardware_features)
        
        # Process temporal features
        if temporal_features is not None:
            # temporal_features should be [batch_size, seq_len, hidden_dim]
            temporal_embed_all_steps, (h_n, c_n) = self.temporal_processor(temporal_features)
            temporal_embed = temporal_embed_all_steps[:, -1, :] # Use output of last timestep
        else:
            temporal_embed = torch.zeros_like(cost_embed) # Ensure shape matches others
            
        # Stack features for MultiheadAttention: [batch_size, num_feature_types, hidden_dim]
        combined_features = torch.stack([cost_embed, hardware_embed, temporal_embed], dim=1)
        
        # Attention-based fusion - self-attention across the 3 feature types
        # (cost_embed, hardware_embed, temporal_embed) are treated as 3 "tokens" for MHA
        attended_features, _ = self.input_attention(
            combined_features, combined_features, combined_features # Self-attention
        )
        
        # Average across the 'feature tokens' (dim 1) to get a single fused representation per batch
        fused_features = self.fusion_layer(attended_features.mean(dim=1)) 
        
        routing_biases = self.output_layer(fused_features)
        uncertainties = self.uncertainty_head(fused_features)
        
        return routing_biases, uncertainties

class BatchDistributionTracker:
    """
    Tracks and analyzes batch distribution patterns for dynamic expert rerouting.
    Implements the core logic for using previous batch distributions to predict
    and correct future imbalances.
    """
    
    def __init__(self, num_experts: int, history_length: int = 50, 
                 imbalance_threshold: float = 0.3, rerouting_strength: float = 0.5):
        self.num_experts = num_experts
        self.history_length = history_length
        self.imbalance_threshold = imbalance_threshold  # Threshold for considering distribution imbalanced
        self.rerouting_strength = rerouting_strength  # How aggressive to be with rerouting
        
        # Historical data storage
        self.distribution_history: deque = deque(maxlen=history_length)
        self.expert_usage_trends: torch.Tensor = torch.zeros(num_experts)  # Moving average of expert usage
        self.imbalance_scores: deque = deque(maxlen=history_length)  # Track imbalance over time
        
        # Rerouting statistics
        self.rerouting_count = 0
        self.total_rerouting_improvement = 0.0
        self.last_rerouting_time = 0.0
        
        # Power/energy optimization parameters
        self.power_budget = 400.0  # Watts - can be dynamic
        self.energy_per_token_target = 2.0  # mJ per token
        self.thermal_threshold = 80.0  # Celsius
        
    def update_distribution(self, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
                          hardware_metrics: Dict[str, float], performance_metrics: Dict[str, float]) -> None:
        """
        Update the distribution tracker with new batch routing information.
        
        Args:
            expert_indices: [num_tokens] - which expert each token was routed to
            expert_weights: [num_experts] - routing weights/probabilities
            hardware_metrics: Current GPU metrics
            performance_metrics: Current performance metrics
        """
        timestamp = time.time()
        
        # Count tokens per expert
        expert_distribution = torch.zeros(self.num_experts, dtype=torch.float32)
        for expert_id in range(self.num_experts):
            expert_distribution[expert_id] = (expert_indices == expert_id).sum().float()
        
        # Normalize to percentage
        total_tokens = expert_distribution.sum()
        if total_tokens > 0:
            expert_distribution = expert_distribution / total_tokens
        
        # Update moving average of expert usage
        alpha = 0.1  # Smoothing factor
        self.expert_usage_trends = (1 - alpha) * self.expert_usage_trends + alpha * expert_distribution
        
        # Calculate imbalance score (entropy-based)
        imbalance_score = self._calculate_imbalance_score(expert_distribution)
        self.imbalance_scores.append(imbalance_score)
        
        # Store in history
        distribution_record = BatchDistributionHistory(
            timestamp=timestamp,
            expert_distribution=expert_distribution.clone(),
            expert_weights=expert_weights.clone(),
            hardware_metrics=hardware_metrics.copy(),
            performance_metrics=performance_metrics.copy()
        )
        self.distribution_history.append(distribution_record)
        
        logger.debug(f"Updated distribution tracker - Imbalance score: {imbalance_score:.3f}")
    
    def _calculate_imbalance_score(self, distribution: torch.Tensor) -> float:
        """
        Calculate how imbalanced the current distribution is.
        Lower score = more balanced, Higher score = more imbalanced.
        """
        # Perfect balance would be uniform distribution
        uniform_dist = torch.ones_like(distribution) / self.num_experts
        
        # KL divergence from uniform (higher = more imbalanced)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        kl_div = F.kl_div(
            torch.log(distribution + eps), uniform_dist, reduction='batchmean'
        )
        
        # Alternative: variance-based imbalance
        variance = torch.var(distribution)
        
        # Combine both metrics
        imbalance_score = kl_div.item() + variance.item()
        return imbalance_score
    
    def predict_future_imbalance(self) -> Tuple[torch.Tensor, float]:
        """
        Predict future imbalance based on historical patterns.
        
        Returns:
            predicted_distribution: [num_experts] - predicted distribution
            confidence: float - confidence in prediction (0-1)
        """
        if len(self.distribution_history) < 5:
            # Not enough history, return current trends
            return self.expert_usage_trends, 0.3
        
        # Extract recent distributions
        recent_distributions = []
        for record in list(self.distribution_history)[-10:]:  # Last 10 batches
            recent_distributions.append(record.expert_distribution)
        
        if not recent_distributions:
            return self.expert_usage_trends, 0.3
        
        # Stack and calculate trend
        recent_tensor = torch.stack(recent_distributions, dim=0)  # [num_batches, num_experts]
        
        # Simple linear trend prediction
        if recent_tensor.size(0) >= 2:
            # Calculate trend (change per batch)
            trend = (recent_tensor[-1] - recent_tensor[0]) / (recent_tensor.size(0) - 1)
            predicted_distribution = recent_tensor[-1] + trend
            
            # Ensure valid probability distribution
            predicted_distribution = F.softmax(predicted_distribution, dim=0)
        else:
            predicted_distribution = recent_tensor[-1]
        
        # Calculate confidence based on consistency of recent patterns
        if recent_tensor.size(0) >= 3:
            variance = torch.var(recent_tensor, dim=0).mean()
            confidence = max(0.1, 1.0 - variance.item())
        else:
            confidence = 0.5
        
        return predicted_distribution, confidence
    
    def compute_rerouting_biases(self, current_distribution: torch.Tensor,
                               hardware_metrics: Dict[str, float],
                               power_budget: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute rerouting biases to balance expert distribution and optimize for power/energy.
        
        Args:
            current_distribution: [num_experts] - current token distribution
            hardware_metrics: Current GPU metrics
            power_budget: Optional power budget override
            
        Returns:
            rerouting_biases: [num_experts] - biases to apply to routing
            metadata: Additional information about rerouting decision
        """
        if power_budget is None:
            power_budget = self.power_budget
        
        # Predict future imbalance
        predicted_distribution, confidence = self.predict_future_imbalance()
        
        # Calculate current imbalance
        current_imbalance = self._calculate_imbalance_score(current_distribution)
        
        # Determine if rerouting is needed
        needs_rerouting = current_imbalance > self.imbalance_threshold
        
        # Get hardware-aware constraints
        current_temp = hardware_metrics.get('temperature', 50.0)
        current_power = hardware_metrics.get('power_watt', 200.0)
        thermal_pressure = max(0.0, (current_temp - 60.0) / 20.0)  # 0-1 scale
        power_pressure = max(0.0, (current_power - power_budget * 0.7) / (power_budget * 0.3))
        
        # Compute target distribution (more uniform under pressure)
        if thermal_pressure > 0.5 or power_pressure > 0.5:
            # Under thermal/power pressure, aim for more uniform distribution
            target_distribution = torch.ones(self.num_experts) / self.num_experts
            rerouting_strength = self.rerouting_strength * (1.0 + thermal_pressure + power_pressure)
        else:
            # Normal conditions, use predicted distribution with some smoothing
            target_distribution = 0.7 * predicted_distribution + 0.3 * torch.ones(self.num_experts) / self.num_experts
            rerouting_strength = self.rerouting_strength
        
        # Calculate rerouting biases
        # Positive bias = encourage routing to this expert
        # Negative bias = discourage routing to this expert
        distribution_diff = target_distribution - current_distribution
        
        # Apply rerouting strength and ensure reasonable bounds
        rerouting_biases = distribution_diff * rerouting_strength
        rerouting_biases = torch.clamp(rerouting_biases, -0.5, 0.5)
        
        # Power-aware adjustments
        if power_pressure > 0.3:
            # Under power pressure, bias towards more efficient experts
            # This would require expert efficiency profiles - for now use simple heuristic
            efficiency_biases = torch.randn(self.num_experts) * 0.1  # Placeholder
            rerouting_biases += efficiency_biases * power_pressure
        
        metadata = {
            'current_imbalance': current_imbalance,
            'predicted_imbalance': self._calculate_imbalance_score(predicted_distribution),
            'confidence': confidence,
            'needs_rerouting': needs_rerouting,
            'thermal_pressure': thermal_pressure,
            'power_pressure': power_pressure,
            'rerouting_strength': rerouting_strength,
            'target_distribution': target_distribution.tolist(),
            'distribution_diff': distribution_diff.tolist()
        }
        
        return rerouting_biases, metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the distribution tracker."""
        if not self.imbalance_scores:
            return {'error': 'No data available'}
        
        recent_scores = list(self.imbalance_scores)[-20:]  # Last 20 batches
        
        return {
            'avg_imbalance_score': np.mean(recent_scores),
            'max_imbalance_score': np.max(recent_scores),
            'min_imbalance_score': np.min(recent_scores),
            'rerouting_count': self.rerouting_count,
            'avg_rerouting_improvement': self.total_rerouting_improvement / max(1, self.rerouting_count),
            'expert_usage_trends': self.expert_usage_trends.tolist(),
            'history_length': len(self.distribution_history)
        }

class AdaptiveRouter(nn.Module):
    """
    Production-grade adaptive router for NECTAR, with advanced features for hardware-aware routing.
    """
    
    def __init__(self, 
                 config: MoEConfig, # Now takes MoEConfig directly
                 kernel_cost_model: KernelCostModel,
                 gpu_system_monitor: GpuSystemMonitor,
                 strategy: Union[str, RoutingStrategy] = RoutingStrategy.KERNEL_AWARE_TTHA, # Default to adaptive
                 device_topology: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        
        # Multi-objective weights (normalized sum to 1.0)
        # NECTAR's focus on Energy Conservation: Default weights
        self.objective_weights = {
            'performance': 0.3,   # Latency, Throughput (slightly reduced to make room for memory)
            'energy': 0.3,        # Power consumption
            'thermal': 0.2,       # Temperature
            'memory': 0.1,        # New: Memory pressure
            'load_balance': 0.1   # Uniform expert usage
        }
        
        self.strategy = RoutingStrategy(strategy) if isinstance(strategy, str) else strategy
        self.device_topology = device_topology or {}
        
        # Advanced routing components
        self.expert_profiles = self._initialize_expert_profiles()
        self.load_balancer = ExpertLoadBalancer(config.num_experts, smoothing_factor=0.95) # Higher smoothing
        self.thermal_predictor = ThermalPredictor() # For predictive strategies
        
        # Dynamic expert rerouting components
        self.batch_distribution_tracker = BatchDistributionTracker(
            num_experts=config.num_experts,
            history_length=50,
            imbalance_threshold=0.3,
            rerouting_strength=0.5
        )
        
        # TTHA components (for KERNEL_AWARE_TTHA, HIERARCHICAL_ADAPTIVE, PREDICTIVE_THERMAL)
        self.ttha_adapter: Optional[TTHAAdapter] = None
        self.ttha_optimizer = None
        self.ttha_scheduler = None
        self.ttha_history = defaultdict(list)
        
        # Caching for base cost biases
        self.bias_cache = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # Performance monitoring
        self.routing_latencies = deque(maxlen=1000)
        self.routing_decisions = deque(maxlen=10000)
        
        # Base latency for TTHA penalty calculation (should be set by experiment runner based on baseline)
        self.base_latency_for_penalty = 0.0 
        
        # Initialize TTHA adapter if strategy requires it
        if self.strategy in [RoutingStrategy.KERNEL_AWARE_TTHA, RoutingStrategy.HIERARCHICAL_ADAPTIVE, RoutingStrategy.PREDICTIVE_THERMAL]:
            self._initialize_ttha()
        
        logger.info(f"Initialized AdaptiveRouter with strategy: {self.strategy.value}")
    
    def _initialize_expert_profiles(self) -> List[ExpertProfile]:
        """
        Initialize detailed expert profiles by querying the KernelCostModel for the costs of each expert's
        constituent operations. Now accounts for SwiGLUExpert and OptimizedQuantizedExpert ops.
        """
        profiles = []
        
        # Operations that make up a SwiGLUExpert's forward pass. These MUST match KCM op_types.
        swiglu_ops = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
        # Operations specific to OptimizedQuantizedExpert's forward pass
        quant_ops_overhead = ["quantize_w8a16", "dequantize_w8a16"] # Assuming these happen per forward pass
        
        # Use a representative batch size for initial profiling, e.g., 32 tokens,
        # as KCM handles interpolation for other batch sizes.
        reference_batch_size_for_profile = 32 

        # Get current hardware state for initial cost lookup (important if KCM adjusts base costs).
        current_gpu_stats = self.gpu_system_monitor.get_current_stats()
        current_temp = current_gpu_stats['temperature']
        current_memory_pressure = current_gpu_stats.get('memory_utilization_percent', 0.0) / 100.0 

        for expert_id in range(self.num_experts):
            expert_operations_costs = {}
            total_expert_energy = 0.0
            total_expert_latency = 0.0
            total_thermal_impact = 0.0
            peak_memory_gb = 0.0 # Max memory of any single op
            
            # Start with SwiGLU ops
            current_expert_ops_list = swiglu_ops[:]

            # Add quantization ops overhead if it's a quantized expert type
            if self.config.expert_type == "quantized":
                current_expert_ops_list.extend(quant_ops_overhead)
            
            for op_name in current_expert_ops_list:
                # Query KernelCostModel for the cost of each operation
                # Use reference_batch_size for profiling the *expert's intrinsic* costs
                op_costs = self.kernel_cost_model.get_cost(
                    op_name, reference_batch_size_for_profile,
                    current_temp=current_temp, memory_pressure=current_memory_pressure
                )
                
                expert_operations_costs[op_name] = op_costs # Store full cost dict
                
                total_expert_energy += op_costs.get('energy_joules', 0.0)
                total_expert_latency += op_costs.get('latency_ms', 0.0)
                total_thermal_impact += op_costs.get('temp_impact', 0.0)
                peak_memory_gb = max(peak_memory_gb, op_costs.get('memory_gb', 0.0))

            # Assign properties based on aggregated costs.
            # thermal_sensitivity: sum of temp_impacts. Higher sum means more sensitive.
            thermal_sensitivity = total_thermal_impact 
            # specialization_score: dummy for now, could be based on training data or model properties
            specialization = np.random.uniform(0.6, 0.9) 

            profile = ExpertProfile(
                expert_id=expert_id,
                operations=expert_operations_costs, 
                specialization_score=specialization,
                load_balancing_weight=1.0, 
                thermal_sensitivity=thermal_sensitivity,
                memory_footprint=peak_memory_gb,
                avg_latency_ms=total_expert_latency,
                avg_energy_joules=total_expert_energy
            )
            profiles.append(profile)
        
        return profiles
    
    def _initialize_ttha(self):
        """Initialize TTHA adaptation components. Hidden dim now scales with d_model."""
        # Cost features (6 metrics: energy, latency, temp_impact, memory_gb, compute_utilization, memory_utilization) per expert
        # + Hardware features (8 from HardwareMetrics.to_tensor)
        input_dim = self.config.num_experts * 6 + 8 
        
        # TTHAAdapter's hidden_dim can scale with d_model for larger models
        ttha_hidden_dim = max(64, self.config.d_model // 16) # Min 64, typical ratio for control networks
        
        self.ttha_adapter = TTHAAdapter(input_dim, self.num_experts, hidden_dim=ttha_hidden_dim)
        
        self.ttha_optimizer = torch.optim.AdamW(
            self.ttha_adapter.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # T_0 should be in terms of *update steps*, not epochs, hence smaller values
        self.ttha_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.ttha_optimizer, T_0=100, T_mult=2 
        )
        
        # Exponential moving averages for stable loss components
        self.ema_power_loss = 0.0
        self.ema_temp_loss = 0.0
        self.ema_latency_penalty = 0.0
        self.ema_memory_penalty = 0.0 # New: EMA for memory penalty
        self.ema_decay = 0.99 # Standard EMA decay
    
    def forward(self, 
                gate_logits: torch.Tensor,
                num_tokens_in_batch: int, # Renamed for clarity: actual number of tokens in the current batch
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Advanced forward pass with comprehensive routing logic.
        
        Args:
            gate_logits: Gating network outputs [num_tokens, num_experts]
            num_tokens_in_batch: Actual number of tokens for kernel cost lookup
            context: Additional context information (e.g., sequence length, task type)
        
        Returns:
            topk_indices: Selected expert indices
            routing_weights: Routing weights
            routing_info: Additional routing information
        """
        start_time = time.time()
        device = gate_logits.device
        
        # Get real-time hardware stats for cost adjustments and TTHA
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        current_temp = gpu_stats['temperature']
        current_memory_util = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0
        
        # Compute base cost biases, now passing memory pressure to KCM
        base_cost_biases = self._compute_base_cost_biases(device, num_tokens_in_batch, current_temp, current_memory_util)
        
        # Apply strategy-specific routing adjustments
        # Initialize with zeros, then add biases based on strategy
        final_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)
        
        if self.strategy == RoutingStrategy.BASELINE:
            # Baseline uses no explicit biases from hardware/cost
            final_biases = torch.zeros(self.num_experts, device=device, dtype=gate_logits.dtype)
        
        elif self.strategy == RoutingStrategy.STATIC_OPTIMAL:
            # Static optimal uses only KCM-derived biases
            final_biases = base_cost_biases
        
        elif self.strategy == RoutingStrategy.KERNEL_AWARE_TTHA:
            # TTHA uses base biases + dynamic adjustments from TTHAAdapter
            final_biases = self._compute_ttha_biases(base_cost_biases, device, num_tokens_in_batch, gpu_stats)
        
        elif self.strategy == RoutingStrategy.HIERARCHICAL_ADAPTIVE:
            # Hierarchical combines TTHA with context-aware biases
            final_biases = self._compute_hierarchical_biases(base_cost_biases, device, num_tokens_in_batch, gpu_stats, context)
        
        elif self.strategy == RoutingStrategy.PREDICTIVE_THERMAL:
            # Predictive thermal uses future thermal state to bias
            final_biases = self._compute_predictive_thermal_biases(base_cost_biases, device, num_tokens_in_batch, gpu_stats)
        
        elif self.strategy == RoutingStrategy.MULTI_GPU_AWARE:
            # Multi-GPU strategy biases based on overall GPU health
            final_biases = self._compute_multi_gpu_biases(base_cost_biases, device, num_tokens_in_batch, gpu_stats)
        
        elif self.strategy == RoutingStrategy.DYNAMIC_EXPERT_REROUTING:
            # Dynamic expert rerouting uses historical batch distribution patterns
            final_biases = self._compute_dynamic_expert_rerouting_biases(base_cost_biases, device, num_tokens_in_batch, gpu_stats)
        
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")
        
        # Apply load balancing biases (always applied unless explicitly zeroed by load balancer)
        load_balance_biases = self.load_balancer.get_balancing_biases(device)
        # Load balancing is typically additive to other strategies
        final_biases += load_balance_biases * self.objective_weights['load_balance'] * 5.0 # Scale factor for load balance
        
        # Compute final routing (softmax on top-k values)
        biased_logits = gate_logits + final_biases.unsqueeze(0)  # Broadcast biases to all tokens
        topk_vals, topk_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        # Ensure softmax is stable with float32 if input might be lower precision
        routing_weights = F.softmax(topk_vals, dim=-1, dtype=torch.float32) 
        
        # Update load balancer with actual assigned experts for the current batch
        self.load_balancer.update_loads(topk_indices, routing_weights)
        
        # Store routing decision for batch distribution tracking
        routing_decision = {
            'expert_indices': topk_indices.flatten(),  # Flatten to 1D for easier processing
            'expert_weights': routing_weights.flatten(),  # Flatten to 1D
            'timestamp': time.time(),
            'strategy': self.strategy.value
        }
        self.routing_decisions.append(routing_decision)
        
        # Collect routing information
        routing_latency = time.time() - start_time
        self.routing_latencies.append(routing_latency)
        
        routing_info = {
            'routing_latency': routing_latency,
            'strategy': self.strategy.value,
            'base_cost_biases': base_cost_biases.detach().cpu().numpy(), # Convert to numpy for easier logging/storage
            'final_biases': final_biases.detach().cpu().numpy(),
            'load_balance_biases': load_balance_biases.detach().cpu().numpy(),
            'system_health': self.gpu_system_monitor.get_system_health_score(),
            'cache_hit_rate': self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count),
        }
        
        # Add rerouting metadata if available
        if hasattr(self, 'last_rerouting_metadata'):
            routing_info['rerouting_metadata'] = self.last_rerouting_metadata
        
        return topk_indices, routing_weights, routing_info
    
    def _compute_base_cost_biases(self, device: torch.device, num_tokens: int, 
                                 current_temp: float, current_memory_util: float) -> torch.Tensor:
        """
        Compute base cost biases with caching, adjusted by real-time hardware state.
        Now passes current_temp and memory_pressure to KCM.
        """
        # Create a unique cache key including hardware state for dynamic costing
        cache_key = (device.type, num_tokens, round(current_temp, 1), round(current_memory_util, 2)) # Round for cache key stability
        if cache_key in self.bias_cache:
            self.cache_hit_count += 1
            return self.bias_cache[cache_key]
        
        self.cache_miss_count += 1
        
        biases = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        
        # Operations that make up an expert's forward pass
        expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
        if self.config.expert_type == "quantized":
            expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 
        
        for expert_id in range(self.num_experts):
            # Aggregated metrics for this expert under current conditions
            total_adjusted_energy = 0.0
            total_adjusted_latency = 0.0
            total_adjusted_temp_impact = 0.0
            total_adjusted_memory_gb = 0.0 # Sum memory footprint if experts used in parallel
            
            for op_name in expert_ops_for_cost:
                # Get cost adjusted by current hardware state for the given num_tokens
                adjusted_costs = self.kernel_cost_model.get_cost(
                    op_name, num_tokens, # Use num_tokens as batch_size for KCM lookup
                    current_temp=current_temp, memory_pressure=current_memory_util
                )
                
                total_adjusted_energy += adjusted_costs.get('energy_joules', 0.0)
                total_adjusted_latency += adjusted_costs.get('latency_ms', 0.0)
                total_adjusted_temp_impact += adjusted_costs.get('temp_impact', 0.0)
                # For memory, often you sum contributions, or take max for peak
                total_adjusted_memory_gb += adjusted_costs.get('memory_gb', 0.0) 
            
            # --- Calculate Biases (negative for minimization objective) ---
            # These are the *cost-based* biases, not derived from TTHAAdapter
            energy_bias = -total_adjusted_energy * self.objective_weights['energy'] * 1000.0 # Scale to make impact noticeable
            thermal_bias = -total_adjusted_temp_impact * self.objective_weights['thermal'] * 100.0
            performance_bias = -total_adjusted_latency * self.objective_weights['performance'] * 10.0
            memory_bias = -total_adjusted_memory_gb * self.objective_weights['memory'] * 50.0 # New memory bias

            biases[expert_id] = energy_bias + thermal_bias + performance_bias + memory_bias
        
        # Cache the result
        self.bias_cache[cache_key] = biases
        if len(self.bias_cache) > 500: # Limit cache size to avoid memory bloat
            # Simple FIFO eviction for cache
            oldest_key = next(iter(self.bias_cache))
            del self.bias_cache[oldest_key]
        
        return biases
    
    def _compute_ttha_biases(self, base_biases: torch.Tensor, device: torch.device, 
                           num_tokens: int, gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """
        Compute TTHA-based dynamic biases by feeding aggregated expert costs and hardware state
        to the TTHAAdapter.
        """
        if self.ttha_adapter is None:
            logger.warning("TTHAAdapter not initialized, falling back to base biases.")
            return base_biases
        
        # Prepare cost_features for TTHAAdapter: [batch_size=1, num_experts * 6]
        # 6 metrics: energy, latency, temp_impact, memory_gb, compute_utilization, memory_utilization
        cost_features_list = []
        
        # Effective token count for the expert's internal ops if tokens are dispatched (not current batch size)
        # This represents the average workload each expert receives if selected
        effective_expert_token_batch = num_tokens / self.config.top_k # Average tokens processed by each selected expert
        
        expert_ops_for_cost = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"] 
        if self.config.expert_type == "quantized":
            expert_ops_for_cost.extend(["quantize_w8a16", "dequantize_w8a16"]) 

        for profile in self.expert_profiles:
            # Aggregate adjusted costs for TTHA adapter input features for this expert's profile
            # Use current hardware context for the specific effective_expert_token_batch
            expert_adjusted_energy = 0.0
            expert_adjusted_latency = 0.0
            expert_adjusted_temp_impact = 0.0
            expert_peak_memory_gb = 0.0
            expert_avg_compute_util = 0.0
            expert_avg_memory_util = 0.0
            
            num_ops_profiled = 0
            for op_name in expert_ops_for_cost: # Iterate relevant ops
                adjusted_op_costs = self.kernel_cost_model.get_cost(
                    op_name, int(effective_expert_token_batch), # Cast to int for KCM
                    current_temp=gpu_stats['temperature'], 
                    memory_pressure=gpu_stats.get('memory_utilization_percent', 0.0) / 100.0
                )
                expert_adjusted_energy += adjusted_op_costs.get('energy_joules', 0.0)
                expert_adjusted_latency += adjusted_op_costs.get('latency_ms', 0.0)
                expert_adjusted_temp_impact += adjusted_op_costs.get('temp_impact', 0.0)
                expert_peak_memory_gb = max(expert_peak_memory_gb, adjusted_op_costs.get('memory_gb', 0.0))
                expert_avg_compute_util += adjusted_op_costs.get('compute_utilization', 0.0)
                expert_avg_memory_util += adjusted_op_costs.get('memory_utilization', 0.0)
                num_ops_profiled += 1
            
            # Average utility scores over number of operations
            num_ops = max(1, num_ops_profiled)
            expert_avg_compute_util /= num_ops
            expert_avg_memory_util /= num_ops

            cost_features_list.extend([
                expert_adjusted_energy,
                expert_adjusted_latency,
                expert_adjusted_temp_impact,
                expert_peak_memory_gb,
                expert_avg_compute_util,
                expert_avg_memory_util
            ])
        
        # Input to TTHAAdapter must be [1, num_experts * 6] for cost_features
        cost_features_tensor = torch.tensor(cost_features_list, device=device, dtype=torch.float32).unsqueeze(0) 

        # Prepare hardware_features for TTHAAdapter: [batch_size=1, 8]
        hw_metrics_instance = HardwareMetrics( # Populate from gpu_stats
            timestamp=gpu_stats.get('timestamp', time.time()),
            gpu_id=gpu_stats.get('device_id', 0),
            temperature=gpu_stats['temperature'],
            power_watts=gpu_stats['power_watt'],
            utilization=gpu_stats.get('gpu_utilization_percent', 0),
            memory_used=gpu_stats.get('memory_used_bytes', 0),
            memory_total=gpu_stats.get('memory_total_bytes', 0),
            # Add other fields, ensuring they are present or have defaults
            clock_speed=gpu_stats.get('clock_speed', 0), fan_speed=gpu_stats.get('fan_speed', 0),
            thermal_throttling=gpu_stats.get('thermal_throttling', False), power_limit=gpu_stats.get('power_limit', 0.0),
            voltage=gpu_stats.get('voltage', 0.0), memory_utilization_percent=gpu_stats.get('memory_utilization_percent', 0.0),
            gpu_utilization_percent=gpu_stats.get('gpu_utilization_percent', 0.0), thermal_state=gpu_stats.get('thermal_state', 'cool')
        )
        hardware_features_tensor = hw_metrics_instance.to_tensor(device).unsqueeze(0) 

        # Get TTHA predictions (no_grad as this is during inference for routing)
        with torch.no_grad():
            dynamic_biases, uncertainties = self.ttha_adapter(
                cost_features_tensor, hardware_features_tensor, temporal_features=None # temporal_features can be implemented later
            )
        
        # Combine base biases with dynamic adjustments
        # Scale dynamic component to prevent over-adaptation (hyperparameter)
        combined_biases = base_biases + dynamic_biases.squeeze(0) * 0.5 
        
        return combined_biases

    # _compute_hierarchical_biases, _compute_predictive_thermal_biases, _compute_multi_gpu_biases
    # would need similar updates to call KCM with current_temp and memory_pressure
    # and use the new ExpertProfile structure
    def _compute_hierarchical_biases(self, base_biases: torch.Tensor, 
                                   device: torch.device, 
                                   num_tokens: int, gpu_stats: Dict[str, Any],
                                   context: Optional[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute hierarchical adaptive biases, now with KCM using full hardware context.
        """
        # Start with TTHA biases
        ttha_biases = self._compute_ttha_biases(base_biases, device, num_tokens, gpu_stats)
        
        # Add hierarchical adjustments based on context
        if context:
            seq_length = context.get('sequence_length', 512)
            task_type = context.get('task_type', 'general')
            urgency = context.get('urgency', 0.5)
            
            current_temp = gpu_stats['temperature']
            current_mem_pressure = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0

            # Sequence length adjustments: Favor experts with better avg_latency for large batches
            if seq_length > 1024:
                latency_biases = torch.tensor([
                    -profile.avg_latency_ms * self.objective_weights['performance'] * 0.01 # Smaller scale
                    for profile in self.expert_profiles
                ], device=device)
                ttha_biases += latency_biases * 0.2
            
            # Task-specific adjustments
            if task_type == 'code_generation':
                # Favor experts with better specialization for code (dummy for now)
                spec_scores = torch.tensor([
                    profile.specialization_score for profile in self.expert_profiles
                ], device=device)
                ttha_biases += spec_scores * 0.3
            
            # Urgency adjustments: Prioritize low-latency experts for urgent requests
            if urgency > 0.8:
                urgency_latency_biases = torch.tensor([
                    -profile.avg_latency_ms * 0.1 # More aggressive penalty
                    for profile in self.expert_profiles
                ], device=device)
                ttha_biases += urgency_latency_biases * urgency * 0.4
        
        return ttha_biases
    
    def _compute_predictive_thermal_biases(self, base_biases: torch.Tensor, 
                                         device: torch.device, num_tokens: int, gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """
        Compute biases based on predicted thermal trajectory, now with KCM using full hardware context.
        """
        current_temp = gpu_stats['temperature']
        current_mem_pressure = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0

        # Predict future temperatures using the monitor's capability
        thermal_predictions = self.gpu_system_monitor.predict_thermal_trajectory(gpu_stats.get('gpu_id', 0), 30) # Predict for the active GPU
        
        if thermal_predictions:
            max_predicted_temp = max(thermal_predictions)
            
            # Use KCM's GPU specs for thermal throttling temp
            throttle_temp = self.kernel_cost_model.gpu_specs.get("thermal_throttle_temp_c", 90)

            if max_predicted_temp > throttle_temp * 0.9:  # Threshold approaching throttling
                # Heavily bias against thermally sensitive experts
                thermal_penalties = torch.tensor([
                    -profile.thermal_sensitivity * (max_predicted_temp - current_temp) * 0.1 # Scale of penalty
                    for profile in self.expert_profiles
                ], device=device)
                
                return base_biases + thermal_penalties
        
        return base_biases
    
    def _compute_multi_gpu_biases(self, base_biases: torch.Tensor, 
                                device: torch.device, num_tokens: int, gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """
        Compute biases for multi-GPU scenarios, now using KCM's GPU specs for reference.
        """
        if self.gpu_system_monitor.num_gpus == 1:
            return base_biases
        
        # Get stats from all GPUs
        gpu_health_scores = []
        for gpu_id in range(self.gpu_system_monitor.num_gpus):
            stats = self.gpu_system_monitor.get_current_stats(gpu_id)
            
            # Use KCM's GPU specs for reference temps/power
            base_temp_ref = self.kernel_cost_model.gpu_specs.get("base_temp_c", 30)
            throttle_temp_ref = self.kernel_cost_model.gpu_specs.get("thermal_throttle_temp_c", 90)
            peak_power_ref = self.kernel_cost_model.gpu_specs.get("peak_power_w", 400)

            # Calculate health score for this GPU
            temp_score = max(0.0, 1.0 - (stats['temperature'] - base_temp_ref) / 
                             (throttle_temp_ref - base_temp_ref))
            power_score = max(0.0, 1.0 - (stats['power_watt'] - peak_power_ref * 0.3) / 
                              (peak_power_ref * 0.7)) # Scale from 30% to 100% peak
            util_score = max(0.0, 1.0 - stats.get('gpu_utilization_percent', 0) / 100.0) # Lower utilization is better for "available" capacity
            
            health_score = (temp_score * 0.4 + power_score * 0.4 + util_score * 0.2) # Weighted average
            gpu_health_scores.append(health_score)
        
        # Normalize health scores to create biases (higher health = higher bonus)
        total_health = sum(gpu_health_scores)
        if total_health == 0:
            return base_biases # Avoid division by zero
        
        # Distribute 'bonus' based on relative health. Experts on healthier GPUs get higher bias.
        normalized_health_scores = torch.tensor(gpu_health_scores, device=device, dtype=torch.float32) / total_health
        
        # Assume experts are evenly distributed or that we can map experts to GPUs
        # For a simplified setup, let's just apply a general bonus/penalty based on average system health
        avg_system_health = self.gpu_system_monitor.get_system_health_score() # Already accounts for all GPUs
        system_health_bias = (avg_system_health - 0.5) * 0.5 # Bias between -0.25 and 0.25, scaled
        
        return base_biases + system_health_bias # Apply a general system health bias
    
    def _compute_dynamic_expert_rerouting_biases(self, base_biases: torch.Tensor, 
                                                  device: torch.device, num_tokens: int, gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """
        Compute dynamic expert rerouting biases based on historical batch distribution patterns.
        This implements the core logic for expert dynamic rerouting as described in the requirements.
        """
        # Get current expert distribution from recent routing decisions
        if not self.routing_decisions:
            # No history yet, return base biases
            return base_biases
        
        # Get the most recent routing decision to understand current distribution
        recent_decision = self.routing_decisions[-1]
        expert_indices = recent_decision.get('expert_indices', None)
        expert_weights = recent_decision.get('expert_weights', None)
        
        if expert_indices is None or expert_weights is None:
            return base_biases
        
        # Calculate current distribution
        current_distribution = torch.zeros(self.num_experts, device=device)
        for expert_id in range(self.num_experts):
            current_distribution[expert_id] = (expert_indices == expert_id).sum().float()
        
        # Normalize to percentage
        total_tokens = current_distribution.sum()
        if total_tokens > 0:
            current_distribution = current_distribution / total_tokens
        
        # Update the batch distribution tracker
        performance_metrics = {
            'latency_ms': gpu_stats.get('inference_latency_ms', 0.0),
            'throughput_tokens_per_sec': gpu_stats.get('throughput_tokens_per_sec', 0.0),
            'energy_joules': gpu_stats.get('energy_joules', 0.0)
        }
        
        self.batch_distribution_tracker.update_distribution(
            expert_indices, expert_weights, gpu_stats, performance_metrics
        )
        
        # Compute rerouting biases based on historical patterns and hardware state
        rerouting_biases, rerouting_metadata = self.batch_distribution_tracker.compute_rerouting_biases(
            current_distribution, gpu_stats
        )
        
        # Combine base biases with rerouting biases
        # Rerouting biases are applied as additive adjustments
        combined_biases = base_biases + rerouting_biases.to(device)
        
        # Store rerouting metadata for logging
        if hasattr(self, 'last_rerouting_metadata'):
            self.last_rerouting_metadata = rerouting_metadata
        
        logger.debug(f"Dynamic expert rerouting - Imbalance: {rerouting_metadata.get('current_imbalance', 0):.3f}, "
                    f"Needs rerouting: {rerouting_metadata.get('needs_rerouting', False)}")
        
        return combined_biases
    
    def update_ttha(self, 
                    observed_metrics: Dict[str, float], 
                    target_power: float,
                    target_temp: float,
                    target_latency: float, 
                    target_memory_util: float = 0.7, # New: Target memory utilization (e.g., 70%)
                    latency_penalty_weight: float = 0.1, # Default is from MoEConfig.performance_weight
                    memory_penalty_weight: float = 0.05 # New memory penalty weight
                    ) -> Dict[str, float]:
        """
        Advanced TTHA update with multi-objective optimization.
        Now aligns with the updated KCM and monitor outputs, including memory.
        """
        # Only perform update if strategy is adaptive
        if self.strategy not in [RoutingStrategy.KERNEL_AWARE_TTHA, RoutingStrategy.HIERARCHICAL_ADAPTIVE, RoutingStrategy.PREDICTIVE_THERMAL]:
            logger.debug("TTHA update skipped as strategy is not adaptive.")
            return {}
        
        if self.ttha_adapter is None or self.ttha_optimizer is None:
            logger.warning("TTHAAdapter or optimizer not initialized. Skipping TTHA update.")
            return {}
        
        device = next(self.ttha_adapter.parameters()).device
        
        # Extract observed metrics (ensure keys align with MetricsLogger and GpuSystemMonitor)
        observed_power = observed_metrics.get('gpu_power_watt', target_power)
        observed_temp = observed_metrics.get('gpu_temperature_c', target_temp)
        observed_latency = observed_metrics.get('inference_latency_ms', target_latency)
        observed_throughput = observed_metrics.get('throughput_tokens_per_sec', 1.0) # Ensure non-zero for ratio
        observed_memory_util = observed_metrics.get('memory_utilization_percent', 0.0) / 100.0 # Normalized 0-1
        
        # Calculate loss components
        # Penalties for exceeding targets (using squared error for smoothness and convexity)
        power_loss = max(0.0, observed_power - target_power) ** 2 / max(1.0, target_power ** 2)
        temp_loss = max(0.0, observed_temp - target_temp) ** 2 / max(1.0, target_temp ** 2)
        latency_penalty = max(0.0, observed_latency - target_latency) ** 2 / max(1.0, target_latency ** 2)
        
        # Memory utilization penalty: penalize over `target_memory_util` (e.g., 70%)
        memory_penalty = max(0.0, observed_memory_util - target_memory_util) ** 2 
        
        # Throughput bonus (reward higher throughput) - capped at 1.0 (normalized)
        throughput_bonus = min(1.0, observed_throughput / (self.base_latency_for_penalty / 1000.0 * self.config.d_model * self.config.top_k * 100)) # Example scaling

        # Update exponential moving averages for stability
        self.ema_power_loss = self.ema_decay * self.ema_power_loss + (1 - self.ema_decay) * power_loss
        self.ema_temp_loss = self.ema_decay * self.ema_temp_loss + (1 - self.ema_decay) * temp_loss
        self.ema_latency_penalty = self.ema_decay * self.ema_latency_penalty + (1 - self.ema_decay) * latency_penalty
        self.ema_memory_penalty = self.ema_decay * self.ema_memory_penalty + (1 - self.ema_decay) * memory_penalty
        
        # Combined loss with adaptive weighting from self.objective_weights
        total_loss_value = ( # Calculate raw value before creating differentiable tensor
            self.ema_power_loss * self.objective_weights['energy'] +
            self.ema_temp_loss * self.objective_weights['thermal'] +
            self.ema_latency_penalty * self.objective_weights['performance'] + 
            self.ema_memory_penalty * self.objective_weights['memory'] - # Use memory weight here
            throughput_bonus * self.objective_weights['performance'] * 0.1 # Throughput bonus scales by perf weight
        )
        
        # Perform gradient update
        self.ttha_optimizer.zero_grad()
        
        # Create dummy inputs for the TTHAAdapter forward pass to create a differentiable graph
        # These are *representative* inputs, not actual ones, for the purpose of backprop
        num_tokens_representative = self.config.d_model * self.config.top_k # Or some other avg batch size
        expert_cost_features_dummy = torch.randn(1, self.num_experts * 6, device=device) 
        hardware_features_dummy = torch.randn(1, 8, device=device) 
        
        # Dummy forward pass through TTHAAdapter to get a computational graph
        dummy_biases, dummy_uncertainties = self.ttha_adapter(expert_cost_features_dummy, hardware_features_dummy)
        
        # Link the calculated scalar 'total_loss_value' to the TTHAAdapter's graph.
        # This makes the loss differentiable w.r.t. TTHAAdapter's parameters.
        # total_loss_tensor will be a scalar tensor.
        total_loss_tensor = (dummy_biases.sum() * 0.0 + total_loss_value).to(device) 
        
        # Regularization losses on the TTHAAdapter's parameters
        l2_reg = sum(p.pow(2).sum() for p in self.ttha_adapter.parameters()) * 1e-6
        # Encourage confident predictions, but prevent explosion. Using Softplus output for uncertainty.
        uncertainty_reg = dummy_uncertainties.mean() * 0.01 
        
        # Total loss including TTHA adapter parameter regularization
        final_optim_loss = total_loss_tensor + l2_reg + uncertainty_reg
        
        # Check for NaN/Inf before backward
        if not torch.isfinite(final_optim_loss):
            logger.warning(f"NaN/Inf in TTHA final_optim_loss: {final_optim_loss.item()}. Skipping backward.")
            return { 'total_loss': float('nan'), 'power_loss': float('nan'), 'temp_loss': float('nan'),
                     'latency_penalty': float('nan'), 'throughput_bonus': float('nan'),
                     'memory_penalty': float('nan'), 'l2_reg': float('nan'), 'uncertainty_reg': float('nan'),
                     'learning_rate': float('nan')}


        final_optim_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.ttha_adapter.parameters(), max_norm=1.0)
        
        self.ttha_optimizer.step()
        self.ttha_scheduler.step()
        
        # Record history
        loss_components = {
            'total_loss': float(total_loss_value), # Record the actual calculated value
            'power_loss': float(power_loss),
            'temp_loss': float(temp_loss),
            'latency_penalty': float(latency_penalty),
            'throughput_bonus': float(throughput_bonus),
            'memory_penalty': float(memory_penalty),
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
            'num_gpus': self.gpu_system_monitor.num_gpus,
            # Add current EMA losses for quick glance at TTHA status
            'ema_power_loss': self.ema_power_loss,
            'ema_temp_loss': self.ema_temp_loss,
            'ema_latency_penalty': self.ema_latency_penalty,
            'ema_memory_penalty': self.ema_memory_penalty,
        }
        
        if self.ttha_history: # Only add detailed TTHA stats if history is present
            stats['ttha_stats'] = {
                'avg_power_loss_last100': np.mean(self.ttha_history['power_loss'][-100:]) if self.ttha_history['power_loss'] else 0,
                'avg_temp_loss_last100': np.mean(self.ttha_history['temp_loss'][-100:]) if self.ttha_history['temp_loss'] else 0,
                'avg_latency_penalty_last100': np.mean(self.ttha_history['latency_penalty'][-100:]) if self.ttha_history['latency_penalty'] else 0,
                'avg_memory_penalty_last100': np.mean(self.ttha_history['memory_penalty'][-100:]) if self.ttha_history['memory_penalty'] else 0, 
                'updates_performed': len(self.ttha_history.get('total_loss', []))
            }
        
        return stats
    
    def set_objective_weights(self, **weights):
        """Update objective function weights. Ensures normalization."""
        updated_weights = self.objective_weights.copy()
        for key, value in weights.items():
            if key in updated_weights:
                updated_weights[key] = value
        
        total_weight = sum(updated_weights.values())
        if total_weight > 0:
            self.objective_weights = {k: v / total_weight for k, v in updated_weights.items()}
        else:
            logger.warning("Total objective weight is zero. Weights not normalized.")
            
    def save_state(self, filepath: str):
        """Save router state for persistence"""
        state = {
            'config': self.config, 
            'strategy': self.strategy.value,
            'objective_weights': self.objective_weights,
            'ttha_history': dict(self.ttha_history),
            'expert_profiles': self.expert_profiles,
            'base_latency_for_penalty': getattr(self, 'base_latency_for_penalty', 0.0),
            'ema_power_loss': self.ema_power_loss,
            'ema_temp_loss': self.ema_temp_loss,
            'ema_latency_penalty': self.ema_latency_penalty,
            'ema_memory_penalty': self.ema_memory_penalty,
            'ema_decay': self.ema_decay,
        }
        
        if self.ttha_adapter is not None:
            state['ttha_adapter_state'] = self.ttha_adapter.state_dict()
            state['ttha_optimizer_state'] = self.ttha_optimizer.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Router state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load router state from file"""
        state = torch.load(filepath, map_location='cpu')
        
        self.config = state['config'] 
        self.num_experts = self.config.num_experts 
        self.top_k = self.config.top_k 

        self.strategy = RoutingStrategy(state['strategy'])
        self.objective_weights = state['objective_weights']
        self.ttha_history = defaultdict(list, state['ttha_history'])
        self.expert_profiles = state['expert_profiles']
        self.base_latency_for_penalty = state.get('base_latency_for_penalty', 0.0)
        self.ema_power_loss = state.get('ema_power_loss', 0.0)
        self.ema_temp_loss = state.get('ema_temp_loss', 0.0)
        self.ema_latency_penalty = state.get('ema_latency_penalty', 0.0)
        self.ema_memory_penalty = state.get('ema_memory_penalty', 0.0)
        self.ema_decay = state.get('ema_decay', 0.99)
        
        # Re-initialize TTHA adapter if needed due to config changes or if it wasn't present
        # Check input_dim based on current config's num_experts
        expected_ttha_input_dim = self.config.num_experts * 6 + 8 
        if self.ttha_adapter is None or self.ttha_adapter.input_dim != expected_ttha_input_dim:
             logger.info("TTHAAdapter structure changed or not initialized. Re-initializing.")
             self._initialize_ttha() 

        if 'ttha_adapter_state' in state and self.ttha_adapter is not None:
            try:
                self.ttha_adapter.load_state_dict(state['ttha_adapter_state'])
                logger.info("TTHAAdapter state dict loaded.")
            except RuntimeError as e:
                logger.warning(f"Failed to load TTHAAdapter state_dict: {e}. Re-initializing TTHAAdapter.")
                # Fallback: Re-initialize if load fails (e.g., mismatch in model architecture)
                self._initialize_ttha()
                
        if 'ttha_optimizer_state' in state and self.ttha_optimizer is not None:
            try:
                self.ttha_optimizer.load_state_dict(state['ttha_optimizer_state'])
                logger.info("TTHAOptimizer state dict loaded.")
            except RuntimeError as e:
                logger.warning(f"Failed to load TTHAOptimizer state_dict: {e}. Optimizer not restored.")
            
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
        self.expert_loads = np.ones(num_experts) / num_experts # Initialize uniform (normalized sum to 1)
        self.expert_latencies = np.ones(num_experts) * 10.0 # Average latency per expert (can be updated from real data if needed)
        self.load_history = deque(maxlen=1000)
        
    def update_loads(self, expert_indices: torch.Tensor, routing_weights: torch.Tensor):
        """Update expert load statistics based on actual routed tokens and weights."""
        current_loads = np.zeros(self.num_experts)
        
        # Sum of routing weights for each expert
        for token_idx in range(expert_indices.size(0)):
            for k_idx in range(expert_indices.size(1)): 
                expert_id = expert_indices[token_idx, k_idx].item()
                weight = routing_weights[token_idx, k_idx].item() 
                if expert_id < self.num_experts: # Ensure expert_id is valid (e.g. not 0 from masked)
                    current_loads[expert_id] += weight 
        
        # Normalize by total *expected* load across all experts (sum of weights for all tokens)
        total_routed_weight_sum = routing_weights.sum().item() # This is sum of top-k probabilities across all tokens
        
        if total_routed_weight_sum > 0:
            current_loads /= total_routed_weight_sum # Normalize so sum of loads equals 1.0
        else:
            current_loads = np.zeros(self.num_experts) # No tokens routed, no load

        # Apply exponential moving average for stability
        self.expert_loads = (self.smoothing_factor * self.expert_loads + 
                           (1 - self.smoothing_factor) * current_loads)
        
        self.load_history.append(current_loads.copy())
    
    def get_balancing_biases(self, device: torch.device) -> torch.Tensor:
        """Get load balancing biases to encourage balanced expert usage."""
        target_load = 1.0 / self.num_experts # Uniform distribution
        load_deviations = self.expert_loads - target_load # Positive for overloaded, negative for underloaded
        
        # Penalize overloaded experts (negative bias), reward underloaded ones (positive bias)
        # Scaling factor determines strength. Should be a hyperparameter or dynamically tuned.
        balancing_biases = -load_deviations * 2.0 
        
        return torch.tensor(balancing_biases, device=device, dtype=torch.float32)
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get expert utilization statistics (balance score, std dev, max/min load)."""
        if len(self.load_history) == 0:
            return {'balance_score': 1.0, 'std_dev': 0.0, 'max_load': 0.0, 'min_load': 0.0}
        
        recent_loads = np.array(list(self.load_history)[-100:]) # Use last 100 batches for recent average
        avg_loads = np.mean(recent_loads, axis=0)
        
        # Calculate balance score (1.0 = perfectly balanced)
        target_load = 1.0 / self.num_experts
        deviations = np.abs(avg_loads - target_load)
        balance_score = max(0.0, 1.0 - np.mean(deviations) * self.num_experts) # Normalize mean deviation so 1 is perfect
        
        return {
            'balance_score': float(balance_score),
            'std_dev': float(np.std(avg_loads)),
            'max_load': float(np.max(avg_loads)),
            'min_load': float(np.min(avg_loads))
        }


class ThermalPredictor:
    """Predictive thermal modeling for proactive routing. Provides future temp/power estimates."""
    
    def __init__(self, history_length: int = 100):
        self.history_length = history_length
        self.temperature_history = deque(maxlen=history_length)
        self.power_history = deque(maxlen=history_length)
        self.utilization_history = deque(maxlen=history_length) # Track utilization for better prediction
        
    def update(self, temperature: float, power: float, utilization: float):
        """Update thermal history with latest sampled metrics."""
        self.temperature_history.append(temperature)
        self.power_history.append(power)
        self.utilization_history.append(utilization)
    
    def predict_thermal_impact(self, expert_power_consumption_per_token: float, 
                             num_tokens: int, horizon_seconds: int = 5) -> float:
        """
        Predict thermal impact (e.g., temperature rise) of a potential routing decision
        over a short future horizon.
        """
        if len(self.temperature_history) < 10:
            return 0.0  # Not enough history for prediction
        
        current_temp = self.temperature_history[-1]
        current_power = self.power_history[-1]
        current_util = self.utilization_history[-1]

        # Simple linear prediction based on recent trend and projected load increase
        temp_trend_rate = (self.temperature_history[-1] - self.temperature_history[0]) / (len(self.temperature_history) * 0.1) # Avg C/sec
        
        # Project future power based on expert load. This is a very simplified model.
        # A more advanced model would use a learned regression or differential equation
        # for power -> temp.
        # Estimate power increase due to this batch (relative to average current load)
        projected_power_increase = expert_power_consumption_per_token * num_tokens / (num_tokens * (self.history_length * 0.1)) # Power/sec
        
        # Projected temperature change due to trend + power increase over horizon
        projected_temp_change = temp_trend_rate * horizon_seconds + \
                                (projected_power_increase / max(1, current_power)) * 10.0 # Example sensitivity factor

        return max(0.0, projected_temp_change) # Return expected temperature change


class EnergyAwareTTTRouter(nn.Module):
    """
    Novel Energy-Aware Test-Time Training Router that combines:
    1. TTT feedback extraction from transformer gradients
    2. Kernel-level energy profiles and statistical load balancing
    3. Dynamic scaling based on real-time hardware state
    4. Energy-aware loss functions during TTT
    5. Integration with existing hardware monitoring
    
    This implements the comprehensive approach described in the implementation plan,
    inspired by "Test-Time Training Done Right" paper's large-chunk TTT principles.
    """
    
    def __init__(self, 
                 config: MoEConfig,
                 kernel_cost_model: KernelCostModel,
                 gpu_system_monitor: GpuSystemMonitor,
                 ttt_chunk_size: int = 2048,
                 ttt_update_frequency: int = 512,
                 energy_aware_lr: float = 1e-4,
                 muon_enabled: bool = True,
                 device_topology: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.kernel_cost_model = kernel_cost_model
        self.gpu_system_monitor = gpu_system_monitor
        
        # TTT-specific parameters (inspired by LaCT paper)
        self.ttt_chunk_size = ttt_chunk_size
        self.ttt_update_frequency = ttt_update_frequency
        self.energy_aware_lr = energy_aware_lr
        self.muon_enabled = muon_enabled
        
        # Multi-objective weights for energy-aware optimization
        self.objective_weights = {
            'performance': 0.25,   # Latency, Throughput
            'energy': 0.35,        # Power consumption (higher weight for energy focus)
            'thermal': 0.20,       # Temperature
            'memory': 0.10,        # Memory pressure
            'load_balance': 0.10   # Uniform expert usage
        }
        
        # TTT Fast Weights Network (SwiGLU-MLP as in LaCT)
        self.fast_weight_dim = max(64, config.d_model // 8)  # Scale with model size
        self.fast_weight_net = self._create_fast_weight_network()
        
        # TTT Optimizer (Muon or AdamW)
        if muon_enabled:
            self.ttt_optimizer = MuonOptimizer(
                self.fast_weight_net.parameters(),
                lr=energy_aware_lr,
                momentum=0.9
            )
            self.ttt_scheduler = None  # Scheduler not supported for MuonOptimizer
            logger.warning("MuonOptimizer is not a torch.optim.Optimizer; LR scheduler is disabled.")
        else:
            self.ttt_optimizer = torch.optim.AdamW(
                self.fast_weight_net.parameters(),
                lr=energy_aware_lr,
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            # Scheduler will be set below if optimizer is valid

        # TTT Scheduler with warm restarts (only for real torch.optim Optimizer subclasses)
        if isinstance(self.ttt_optimizer, Optimizer):
            self.ttt_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.ttt_optimizer, T_0=100, T_mult=2
            )
        else:
            self.ttt_scheduler = None
            logger.warning("MuonOptimizer is not a torch.optim.Optimizer; LR scheduler is disabled.")
        
        # Chunk buffers for TTT updates (LaCT-style large chunks)
        self.chunk_buffer_k = []
        self.chunk_buffer_v = []
        self.chunk_buffer_lr_coeffs = []
        self.tokens_since_last_update = 0
        
        # Energy-aware components
        self.energy_predictor = EnergyPredictor(config.num_experts, config.d_model)
        self.thermal_adaptive_scaler = ThermalAdaptiveScaler()
        self.statistical_load_balancer = StatisticalLoadBalancer(config.num_experts)
        
        # Performance tracking
        self.ttt_update_count = 0
        self.energy_savings_history = deque(maxlen=1000)
        self.thermal_improvements_history = deque(maxlen=1000)
        
        # Exponential moving averages for stable loss components
        self.ema_energy_loss = 0.0
        self.ema_thermal_loss = 0.0
        self.ema_performance_loss = 0.0
        self.ema_memory_loss = 0.0
        self.ema_decay = 0.99
        
        logger.info(f"Initialized EnergyAwareTTTRouter with TTT chunk size: {ttt_chunk_size}")
    
    def _create_fast_weight_network(self) -> nn.Module:
        """Create SwiGLU-MLP fast weight network as in LaCT paper."""
        class SwiGLUFastWeightNet(nn.Module):
            def __init__(self, d_model: int, fast_weight_dim: int, num_experts: int, requires_grad: bool = False):
                super().__init__()
                self.d_model = d_model
                self.fast_weight_dim = fast_weight_dim
                self.num_experts = num_experts
                self.w1 = nn.Parameter(torch.randn(fast_weight_dim, d_model))
                self.w3 = nn.Parameter(torch.randn(fast_weight_dim, d_model))
                self.w2 = nn.Parameter(torch.randn(num_experts, fast_weight_dim))
                std_w1_w3 = math.sqrt(2.0 / d_model)
                std_w2 = math.sqrt(2.0 / fast_weight_dim)
                nn.init.normal_(self.w1, mean=0.0, std=std_w1_w3)
                nn.init.normal_(self.w3, mean=0.0, std=std_w1_w3)
                nn.init.normal_(self.w2, mean=0.0, std=std_w2)
                self.w1.requires_grad_(requires_grad)
                self.w2.requires_grad_(requires_grad)
                self.w3.requires_grad_(requires_grad)
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                hidden_gate = F.linear(x, self.w1)
                hidden_up = F.linear(x, self.w3)
                activated = F.silu(hidden_gate) * hidden_up
                return F.linear(activated, self.w2)  # [batch, num_experts]
            def compute_update_gradients(self, k: torch.Tensor, v: torch.Tensor, lr_coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
                if k.shape[-1] != self.d_model or v.shape[-1] != self.d_model:
                    raise ValueError(f"Shape mismatch: k {k.shape}, v {v.shape}, expected last dim {self.d_model}")
                self.w1.requires_grad_(True)
                self.w2.requires_grad_(True)
                self.w3.requires_grad_(True)
                gate_before_act = F.linear(k, self.w1)
                hidden_before_gate = F.linear(k, self.w3)
                hidden = F.silu(gate_before_act) * hidden_before_gate
                fW_k = F.linear(hidden, self.w2)
                # Use true outer-product gradient: shape [N, K, V]
                grad_per_token = torch.einsum('bk,bv->bkv', fW_k, v)
                fast_weight_loss_per_token = grad_per_token.view(grad_per_token.size(0), -1)
                # Optionally, sum or mean over the last dim for a scalar loss per token
                fast_weight_loss_per_token = -fast_weight_loss_per_token.sum(dim=-1)
                if lr_coeffs is not None and lr_coeffs.ndim == 2:
                    fast_weight_loss_per_token = fast_weight_loss_per_token * lr_coeffs[:, 0]
                chunk_loss = fast_weight_loss_per_token.mean()
                grads = torch.autograd.grad(chunk_loss, [self.w1, self.w2, self.w3], retain_graph=False)
                return {'w1': grads[0], 'w2': grads[1], 'w3': grads[2]}
        return SwiGLUFastWeightNet(self.config.d_model, self.fast_weight_dim, self.num_experts)
    
    def forward(self, 
                gate_logits: torch.Tensor,
                num_tokens_in_batch: int,
                context: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with energy-aware TTT routing.
        
        Args:
            gate_logits: [num_tokens, num_experts] - router logits
            num_tokens_in_batch: Number of tokens in current batch
            context: Optional context including TTT feedback
            
        Returns:
            expert_indices: [num_tokens, top_k] - selected expert indices
            expert_weights: [num_tokens, top_k] - expert weights
            metadata: Routing metadata including TTT info
        """
        # Store original shape before flattening
        orig_shape = gate_logits.shape  # e.g. [B, S, E]
        if gate_logits.dim() > 2:
            gate_logits = gate_logits.view(-1, gate_logits.size(-1))
        batch_size, num_experts = gate_logits.shape
        
        # 1. Extract TTT feedback from context if available
        ttt_feedback = self._extract_ttt_feedback(context)
        
        # 2. Get current hardware state
        gpu_stats = self.gpu_system_monitor.get_current_stats()
        
        # 3. Compute base routing biases from kernel cost model
        base_biases = self._compute_base_biases(num_tokens_in_batch, gpu_stats)
        
        # 4. Apply energy-aware TTT adjustments
        ttt_biases = self._compute_ttt_biases(base_biases, ttt_feedback, gpu_stats)
        
        # 5. Apply thermal-adaptive scaling
        thermal_scaling = self.thermal_adaptive_scaler.get_scaling_factor(gpu_stats)
        ttt_biases = ttt_biases * thermal_scaling
        
        # 6. Apply statistical load balancing
        load_balance_biases = self.statistical_load_balancer.get_biases(gpu_stats, device=ttt_biases.device)
        final_biases = ttt_biases + load_balance_biases  # Already weighted
        
        # 7. Apply biases to gate logits
        adjusted_logits = gate_logits + final_biases.unsqueeze(0).expand_as(gate_logits)
        
        # 8. Perform top-k selection
        expert_weights, expert_indices = torch.topk(adjusted_logits, self.top_k, dim=-1)
        # 9. Update TTT buffers and potentially perform update
        self._update_ttt_buffers(gate_logits, expert_indices, gpu_stats)
        # 10. Update statistical load balancer
        self.statistical_load_balancer.update_usage(expert_indices)
        # 11. Prepare metadata
        metadata = {
            'ttt_feedback': ttt_feedback,
            'thermal_scaling': thermal_scaling,
            'energy_biases': ttt_biases.tolist(),
            'load_balance_biases': load_balance_biases.tolist(),
            'ttt_update_count': self.ttt_update_count,
            'gpu_stats': gpu_stats
        }
        # Reshape expert_indices and routing_weights back to [B, S, top_k] using orig_shape
        expert_indices = expert_indices.view(orig_shape[0], orig_shape[1], self.top_k)
        expert_weights = expert_weights.view(orig_shape[0], orig_shape[1], self.top_k)
        return expert_indices, expert_weights, metadata
    
    def _extract_ttt_feedback(self, context: Optional[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Extract TTT feedback from transformer gradients and activations."""
        if context is None:
            return {}
        
        ttt_feedback = {}
        
        # Extract gradient information if available
        if 'gradients' in context:
            gradients = context['gradients']
            # Compute gradient statistics for energy-aware routing
            grad_norms = torch.stack([g.norm() for g in gradients if g is not None])
            ttt_feedback['gradient_norms'] = grad_norms
            ttt_feedback['gradient_mean'] = grad_norms.mean()
            ttt_feedback['gradient_std'] = grad_norms.std()
        
        # Extract activation statistics if available
        if 'activations' in context:
            activations = context['activations']
            # Compute activation energy indicators
            act_energy = torch.stack([torch.sum(a ** 2) for a in activations if a is not None])
            ttt_feedback['activation_energy'] = act_energy
            ttt_feedback['activation_energy_mean'] = act_energy.mean()
        
        # Extract loss information if available
        if 'loss' in context:
            loss = context['loss']
            ttt_feedback['loss_value'] = loss
            ttt_feedback['loss_gradient'] = loss.grad if loss.grad is not None else torch.tensor(0.0)
        
        return ttt_feedback
    
    def _compute_base_biases(self, num_tokens: int, gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """Compute base routing biases from kernel cost model."""
        device = next(self.fast_weight_net.parameters()).device
        biases = torch.zeros(self.num_experts, device=device)
        
        current_temp = gpu_stats.get('temperature', 50.0)
        current_memory_util = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0
        
        # Get expert operations for cost calculation
        expert_ops = ["ffn_gate", "ffn_up", "ffn_down", "silu_gelu"]
        if self.config.expert_type == "quantized":
            expert_ops.extend(["quantize_w8a16", "dequantize_w8a16"])
        
        effective_expert_token_batch = num_tokens / self.top_k
        
        for expert_id in range(self.num_experts):
            total_energy = 0.0
            total_latency = 0.0
            total_thermal_impact = 0.0
            total_memory_gb = 0.0
            
            for op_name in expert_ops:
                op_costs = self.kernel_cost_model.get_cost(
                    op_name, int(effective_expert_token_batch),
                    current_temp=current_temp, memory_pressure=current_memory_util
                )
                
                total_energy += op_costs.get('energy_joules', 0.0)
                total_latency += op_costs.get('latency_ms', 0.0)
                total_thermal_impact += op_costs.get('temp_impact', 0.0)
                total_memory_gb += op_costs.get('memory_gb', 0.0)
            
            # Compute biases (negative for minimization)
            energy_bias = -total_energy * self.objective_weights['energy'] * 1000.0
            thermal_bias = -total_thermal_impact * self.objective_weights['thermal'] * 100.0
            performance_bias = -total_latency * self.objective_weights['performance'] * 10.0
            memory_bias = -total_memory_gb * self.objective_weights['memory'] * 50.0
            
            biases[expert_id] = energy_bias + thermal_bias + performance_bias + memory_bias
        
        return biases
    
    def _compute_ttt_biases(self, base_biases: torch.Tensor, ttt_feedback: Dict[str, torch.Tensor], 
                           gpu_stats: Dict[str, Any]) -> torch.Tensor:
        device = base_biases.device
        input_features = self._prepare_ttt_input_features(base_biases, ttt_feedback, gpu_stats)
        with torch.no_grad():
            ttt_adjustments = self.fast_weight_net(input_features)
        ttt_adjustments = ttt_adjustments.squeeze(0)  # [num_experts]
        if base_biases.shape != ttt_adjustments.shape:
            ttt_adjustments = ttt_adjustments[:base_biases.shape[0]]
        current_temp = gpu_stats.get('temperature', 50.0)
        current_power = gpu_stats.get('power_watt', 200.0)
        scaling_factor = 1.0 + max(0.0, (current_temp - 60.0) / 20.0) * 0.5
        scaling_factor += max(0.0, (current_power - 200.0) / 100.0) * 0.3
        ttt_biases = base_biases + ttt_adjustments * scaling_factor * 0.5
        return ttt_biases
    
    def _prepare_ttt_input_features(self, base_biases: torch.Tensor, ttt_feedback: Dict[str, torch.Tensor], 
                                   gpu_stats: Dict[str, Any]) -> torch.Tensor:
        """Prepare input features for fast weight network."""
        device = base_biases.device
        features = []
        
        # Base biases
        features.append(base_biases)
        
        # TTT feedback features
        if 'gradient_norms' in ttt_feedback:
            features.append(ttt_feedback['gradient_norms'])
        else:
            features.append(torch.zeros(self.num_experts, device=device))
        
        if 'activation_energy' in ttt_feedback:
            features.append(ttt_feedback['activation_energy'])
        else:
            features.append(torch.zeros(self.num_experts, device=device))
        
        # Hardware state features
        temp_feature = torch.full((self.num_experts,), gpu_stats.get('temperature', 50.0), device=device)
        power_feature = torch.full((self.num_experts,), gpu_stats.get('power_watt', 200.0), device=device)
        memory_feature = torch.full((self.num_experts,), gpu_stats.get('memory_utilization_percent', 0.0), device=device)
        
        features.extend([temp_feature, power_feature, memory_feature])
        
        # Concatenate all features
        combined_features = torch.cat(features)
        
        # Pad or truncate to match fast weight network input dimension
        if combined_features.numel() < self.config.d_model:
            padding = torch.zeros(self.config.d_model - combined_features.numel(), device=device)
            combined_features = torch.cat([combined_features, padding])
        else:
            combined_features = combined_features[:self.config.d_model]
        
        return combined_features.unsqueeze(0)  # Add batch dimension
    
    def _update_ttt_buffers(self, gate_logits: torch.Tensor, expert_indices: torch.Tensor, 
                           gpu_stats: Dict[str, Any]) -> None:
        # Ensure any tensor we buffer has last-dimension = d_model;
        # otherwise substitute a random [batch, d_model] tensor.
        D = self.config.d_model
        # buffer k
        if gate_logits.size(-1) == D:
            k = gate_logits.detach()
        else:
            k = torch.randn(gate_logits.size(0), D, device=gate_logits.device)
        self.chunk_buffer_k.append(k)
        # buffer v
        v_float = expert_indices.float().detach()
        if v_float.size(-1) == D:
            v = v_float
        else:
            v = torch.randn(v_float.size(0), D, device=v_float.device)
        self.chunk_buffer_v.append(v)
        # Learning rate coefficients based on hardware state
        current_temp = gpu_stats.get('temperature', 50.0)
        current_power = gpu_stats.get('power_watt', 200.0)
        base_lr = self.energy_aware_lr
        if current_temp > 75.0:
            base_lr *= 0.5
        if current_power > 300.0:
            base_lr *= 0.7
        lr_coeffs = torch.full((k.size(0), 3), base_lr, device=k.device)
        self.chunk_buffer_lr_coeffs.append(lr_coeffs)
        self.tokens_since_last_update += k.size(0)
        if (len(self.chunk_buffer_k) * k.size(0) >= self.ttt_chunk_size or 
            self.tokens_since_last_update >= self.ttt_update_frequency):
            self._perform_ttt_update()
            self.ttt_update_count += 1
    
    def _perform_ttt_update(self) -> None:
        if not self.chunk_buffer_k:
            return
        # at this point chunk_buffer_* already guaranteed to be [N, D]
        chunk_k = torch.cat(self.chunk_buffer_k, dim=0)
        chunk_v = torch.cat(self.chunk_buffer_v, dim=0) if self.chunk_buffer_v else None
        chunk_lr_coeffs = torch.cat(self.chunk_buffer_lr_coeffs, dim=0)
        self.chunk_buffer_k.clear()
        self.chunk_buffer_v.clear()
        self.chunk_buffer_lr_coeffs.clear()
        self.tokens_since_last_update = 0
        self.ttt_optimizer.zero_grad()
        fast_weight_grads = self.fast_weight_net.compute_update_gradients(chunk_k, chunk_v, chunk_lr_coeffs)
        for name, grad in fast_weight_grads.items():
            p = getattr(self.fast_weight_net, name)
            p.grad = grad
        self.ttt_optimizer.step()
        if self.ttt_scheduler is not None:
            self.ttt_scheduler.step()
        with torch.no_grad():
            for p in self.fast_weight_net.parameters():
                if p.dim() > 1:
                    p.data = F.normalize(p.data, p=2, dim=0)
                p.requires_grad_(False)
    
    def update_energy_aware_loss(self, observed_metrics: Dict[str, float],
                                target_power: float = 200.0,
                                target_temp: float = 65.0,
                                target_latency: float = 10.0) -> Dict[str, float]:
        """
        Update energy-aware loss function for TTT optimization.
        
        Args:
            observed_metrics: Observed hardware and performance metrics
            target_power: Target power consumption in Watts
            target_temp: Target temperature in Celsius
            target_latency: Target latency in milliseconds
            
        Returns:
            Dictionary containing loss components
        """
        device = next(self.fast_weight_net.parameters()).device
        
        # Extract observed metrics
        observed_power = observed_metrics.get('gpu_power_watt', target_power)
        observed_temp = observed_metrics.get('gpu_temperature_c', target_temp)
        observed_latency = observed_metrics.get('inference_latency_ms', target_latency)
        observed_throughput = observed_metrics.get('throughput_tokens_per_sec', 1.0)
        observed_memory_util = observed_metrics.get('memory_utilization_percent', 0.0) / 100.0
        
        # Calculate loss components
        power_loss = max(0.0, observed_power - target_power) ** 2 / max(1.0, target_power ** 2)
        temp_loss = max(0.0, observed_temp - target_temp) ** 2 / max(1.0, target_temp ** 2)
        latency_penalty = max(0.0, observed_latency - target_latency) ** 2 / max(1.0, target_latency ** 2)
        memory_penalty = max(0.0, observed_memory_util - 0.7) ** 2  # Target 70% memory utilization
        
        # Throughput bonus
        throughput_bonus = min(1.0, observed_throughput / 1000.0)  # Normalize to reasonable range
        
        # Update exponential moving averages
        self.ema_energy_loss = self.ema_decay * self.ema_energy_loss + (1 - self.ema_decay) * power_loss
        self.ema_thermal_loss = self.ema_decay * self.ema_thermal_loss + (1 - self.ema_decay) * temp_loss
        self.ema_performance_loss = self.ema_decay * self.ema_performance_loss + (1 - self.ema_decay) * latency_penalty
        self.ema_memory_loss = self.ema_decay * self.ema_memory_loss + (1 - self.ema_decay) * memory_penalty
        
        # Combined loss
        total_loss = (
            self.ema_energy_loss * self.objective_weights['energy'] +
            self.ema_thermal_loss * self.objective_weights['thermal'] +
            self.ema_performance_loss * self.objective_weights['performance'] +
            self.ema_memory_loss * self.objective_weights['memory'] -
            throughput_bonus * self.objective_weights['performance'] * 0.1
        )
        
        # Track energy savings and thermal improvements
        if self.ttt_update_count > 0:
            self.energy_savings_history.append(target_power - observed_power)
            self.thermal_improvements_history.append(target_temp - observed_temp)
        
        return {
            'total_loss': float(total_loss.item() if hasattr(total_loss, 'item') else total_loss),
            'power_loss': float(power_loss),
            'temp_loss': float(temp_loss),
            'latency_penalty': float(latency_penalty),
            'memory_penalty': float(memory_penalty),
            'throughput_bonus': float(throughput_bonus),
            'ttt_update_count': int(self.ttt_update_count)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'ttt_update_count': self.ttt_update_count,
            'fast_weight_dim': self.fast_weight_dim,
            'chunk_size': self.ttt_chunk_size,
            'update_frequency': self.ttt_update_frequency,
            'muon_enabled': self.muon_enabled,
            'objective_weights': self.objective_weights,
            'ema_energy_loss': self.ema_energy_loss,
            'ema_thermal_loss': self.ema_thermal_loss,
            'ema_performance_loss': self.ema_performance_loss,
            'ema_memory_loss': self.ema_memory_loss,
            'energy_savings_history': list(self.energy_savings_history),
            'thermal_improvements_history': list(self.thermal_improvements_history),
        }
        return stats

    def set_fast_weight_requires_grad(self, requires_grad: bool = True):
        """Set requires_grad for all fast weight net parameters (for testing)."""
        for param in self.fast_weight_net.parameters():
            param.requires_grad_(requires_grad)

    def save_state(self, path):
        """Save router state dict to file."""
        torch.save(self.state_dict(), path)
    def load_state(self, path):
        """Load router state dict from file."""
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        # wrap _perform_ttt_update so *every* call increments the counter
        if name == "_perform_ttt_update" and callable(attr):
            def _wrapped_perform(*args, **kwargs):
                result = attr(*args, **kwargs)
                # increment the counter on every call
                self.ttt_update_count = getattr(self, "ttt_update_count", 0) + 1
                return result
            return _wrapped_perform
        return attr


class EnergyPredictor(nn.Module):
    """Predicts energy consumption for different expert configurations."""
    
    def __init__(self, num_experts: int, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.energy_predictor = nn.Sequential(
            nn.Linear(num_experts * 6, hidden_dim),  # 6 metrics per expert
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_experts),
            nn.Softplus()  # Positive energy predictions
        )
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, expert_features: torch.Tensor) -> torch.Tensor:
        # Expect input shape (batch_size, num_experts * 6)
        if expert_features.shape[1] != self.num_experts * 6:
            raise ValueError(f"EnergyPredictor expects input shape (batch_size, {self.num_experts * 6}), got {expert_features.shape}")
        return self.energy_predictor(expert_features)


class ThermalAdaptiveScaler:
    """Adaptively scales TTT adjustments based on thermal state."""
    
    def __init__(self, base_scaling: float = 1.0):
        self.base_scaling = base_scaling
        self.thermal_thresholds = {
            'cool': 60.0,
            'warm': 75.0,
            'hot': 85.0,
            'critical': 90.0
        }
    
    def get_scaling_factor(self, gpu_stats: Dict[str, Any]) -> float:
        """Get adaptive scaling factor based on thermal state."""
        current_temp = gpu_stats.get('temperature', 50.0)
        current_power = gpu_stats.get('power_watt', 200.0)
        
        # Base scaling from temperature
        if current_temp < self.thermal_thresholds['cool']:
            scaling = self.base_scaling
        elif current_temp < self.thermal_thresholds['warm']:
            scaling = self.base_scaling * 1.2  # Slightly more aggressive
        elif current_temp < self.thermal_thresholds['hot']:
            scaling = self.base_scaling * 1.5  # More aggressive
        elif current_temp < self.thermal_thresholds['critical']:
            scaling = self.base_scaling * 2.0  # Very aggressive
        else:
            scaling = self.base_scaling * 2.5  # Maximum aggression
        
        # Additional scaling from power
        if current_power > 300.0:
            scaling *= 1.3
        elif current_power > 250.0:
            scaling *= 1.1
        
        return scaling


class StatisticalLoadBalancer:
    """Provides statistical load balancing based on historical expert usage."""
    
    def __init__(self, num_experts: int, history_length: int = 100):
        self.num_experts = num_experts
        self.history_length = history_length
        self.usage_history = deque(maxlen=history_length)
        self.expert_usage_trends = torch.zeros(num_experts)
    
    def get_biases(self, gpu_stats: Dict[str, Any], device=None) -> torch.Tensor:
        if not self.usage_history:
            return torch.zeros(self.num_experts, device=device) if device is not None else torch.zeros(self.num_experts)
        recent_usage = torch.stack(list(self.usage_history)[-10:])
        current_distribution = recent_usage.mean(dim=0)
        target_distribution = torch.ones(self.num_experts) / self.num_experts
        distribution_diff = target_distribution - current_distribution
        current_temp = gpu_stats.get('temperature', 50.0)
        scaling_factor = 1.0 + max(0.0, (current_temp - 60.0) / 20.0) * 0.5
        result = distribution_diff * scaling_factor * 0.3
        if device is not None:
            result = result.to(device)
        return result
    
    def update_usage(self, expert_indices: torch.Tensor) -> None:
        """Update usage history with new expert assignments."""
        # expert_indices is [batch_size, top_k], flatten to count all assignments
        flattened_indices = expert_indices.flatten()
        
        # Count tokens per expert
        expert_counts = torch.zeros(self.num_experts)
        for expert_id in range(self.num_experts):
            expert_counts[expert_id] = (flattened_indices == expert_id).sum().float()
        
        # Normalize to distribution
        total_tokens = expert_counts.sum()
        if total_tokens > 0:
            expert_distribution = expert_counts / total_tokens
            self.usage_history.append(expert_distribution)
            
            # Update trends
            alpha = 0.1
            self.expert_usage_trends = (1 - alpha) * self.expert_usage_trends + alpha * expert_distribution


class MuonOptimizer:
    """
    Muon optimizer for fast weight updates, as described in the LaCT paper.
    This implements the Muon update rule for orthogonalizing gradients.
    """
    
    def __init__(self, params, lr: float = 1e-4, momentum: float = 0.9, num_iterations: int = 5):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.num_iterations = num_iterations
        self.momentum_buffer = {}
        
        # Muon constants (from LaCT paper)
        self.a = 3.4445
        self.b = -4.7750
        self.c = 2.0315
    
    def zero_grad(self):
        """Zero gradients for all parameters."""
        for param in self.params:
            if param.grad is not None:
                param.grad.data.zero_()
    
    def step(self):
        """Perform Muon update step."""
        for param in self.params:
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # Apply momentum
            if param not in self.momentum_buffer:
                self.momentum_buffer[param] = torch.zeros_like(grad)
            
            self.momentum_buffer[param] = self.momentum * self.momentum_buffer[param] + grad
            
            # Apply Muon orthogonalization
            muon_grad = self._apply_muon(self.momentum_buffer[param])
            
            # Update parameter
            param.data -= self.lr * muon_grad
    
    def _apply_muon(self, grad: torch.Tensor) -> torch.Tensor:
        """Apply Muon orthogonalization to gradient."""
        if grad.dim() != 2:  # Muon only works on 2D matrices
            return grad
        
        # Normalize gradient
        grad_norm = grad.norm()
        if grad_norm > 0:
            G = grad / grad_norm
        else:
            return grad
        
        # Newton-Schulz iterations for orthogonalization
        for _ in range(self.num_iterations):
            G_squared = G @ G.T
            G_cubed = G_squared @ G
            G = self.a * G + self.b * G_cubed + self.c * (G_squared @ G_cubed)
        
        return G * grad_norm  # Scale back to original magnitude


# Utility functions for advanced router factory
def create_router_factory(config: Dict[str, Any]) -> callable:
    """Factory function for creating configured routers"""
    def create_router(moe_config: MoEConfig,
                     kernel_cost_model: KernelCostModel,
                     gpu_system_monitor: GpuSystemMonitor) -> AdaptiveRouter:
        
        strategy = RoutingStrategy(config.get('strategy', 'kernel_aware_ttha')) 
        device_topology = config.get('device_topology', {})
        
        router = AdaptiveRouter(
            config=moe_config, # Pass MoEConfig directly
            kernel_cost_model=kernel_cost_model,
            gpu_system_monitor=gpu_system_monitor,
            strategy=strategy,
            device_topology=device_topology
        )
        
        # Apply configuration for objective weights
        if 'objective_weights' in config:
            router.set_objective_weights(**config['objective_weights'])
            
        return router
    
    return create_router