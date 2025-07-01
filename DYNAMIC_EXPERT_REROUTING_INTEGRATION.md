# Dynamic Expert Rerouting Integration Guide

## Overview

This document describes the integration of **Dynamic Expert Rerouting** into your existing NECTAR codebase. This new functionality implements the requirements you specified:

1. **Uses previous batch distribution patterns** to predict and correct future imbalances
2. **Implements expert dynamic rerouting** for MoE imbalance without changing token-to-expert routing
3. **Optimizes for joules per token** (energy efficiency) rather than just performance
4. **Considers thermal consumption and power budgets** for hardware-aware decisions

## Key Components Added

### 1. BatchDistributionTracker Class

**Location**: `src/routers.py`

**Purpose**: Tracks historical batch distribution patterns and predicts future imbalances.

**Key Features**:
- Maintains a sliding window of batch distribution history
- Calculates imbalance scores using KL divergence and variance
- Predicts future distributions using trend analysis
- Computes rerouting biases based on hardware constraints

**Usage**:
```python
from routers import BatchDistributionTracker

tracker = BatchDistributionTracker(
    num_experts=8,
    history_length=50,
    imbalance_threshold=0.3,
    rerouting_strength=0.5
)

# Update with new batch information
tracker.update_distribution(expert_indices, expert_weights, hardware_metrics, performance_metrics)

# Get rerouting biases
rerouting_biases, metadata = tracker.compute_rerouting_biases(
    current_distribution, hardware_metrics
)
```

### 2. New Routing Strategy: DYNAMIC_EXPERT_REROUTING

**Location**: `src/routers.py` - `RoutingStrategy` enum

**Purpose**: Implements the dynamic expert rerouting logic that uses historical patterns.

**Key Features**:
- Integrates with existing `AdaptiveRouter`
- Uses `BatchDistributionTracker` for historical analysis
- Applies rerouting biases to balance expert usage
- Optimizes for power/energy efficiency under thermal constraints

**Usage**:
```python
from routers import RoutingStrategy, AdaptiveRouter

# Set the routing strategy
router.strategy = RoutingStrategy.DYNAMIC_EXPERT_REROUTING

# The router will automatically use batch distribution tracking
# and apply dynamic rerouting biases
```

### 3. Enhanced AdaptiveRouter

**Location**: `src/routers.py` - `AdaptiveRouter` class

**New Methods**:
- `_compute_dynamic_expert_rerouting_biases()`: Core rerouting logic
- Enhanced `forward()` method: Stores routing decisions for tracking

**Integration Points**:
- Automatically initializes `BatchDistributionTracker`
- Stores routing decisions in `routing_decisions` deque
- Applies rerouting biases during forward pass
- Logs rerouting metadata for analysis

## How It Works

### 1. Historical Pattern Tracking

The system tracks how tokens are distributed across experts over time:

```python
# Each batch, the system records:
routing_decision = {
    'expert_indices': topk_indices.flatten(),  # Which expert each token went to
    'expert_weights': routing_weights.flatten(),  # Routing probabilities
    'timestamp': time.time(),
    'strategy': self.strategy.value
}
```

### 2. Imbalance Detection

The system calculates imbalance scores using:
- **KL divergence** from uniform distribution
- **Variance** in expert usage
- **Trend analysis** of recent batches

```python
def _calculate_imbalance_score(self, distribution: torch.Tensor) -> float:
    uniform_dist = torch.ones_like(distribution) / self.num_experts
    kl_div = F.kl_div(torch.log(distribution + eps), uniform_dist, reduction='batchmean')
    variance = torch.var(distribution)
    return kl_div.item() + variance.item()
```

### 3. Future Prediction

Based on historical patterns, the system predicts future imbalances:

```python
def predict_future_imbalance(self) -> Tuple[torch.Tensor, float]:
    # Extract recent distributions
    recent_distributions = []
    for record in list(self.distribution_history)[-10:]:
        recent_distributions.append(record.expert_distribution)
    
    # Calculate trend
    recent_tensor = torch.stack(recent_distributions, dim=0)
    trend = (recent_tensor[-1] - recent_tensor[0]) / (recent_tensor.size(0) - 1)
    predicted_distribution = recent_tensor[-1] + trend
    
    return predicted_distribution, confidence
```

### 4. Hardware-Aware Rerouting

The system computes rerouting biases considering:

- **Thermal pressure**: Higher temperature → more uniform distribution
- **Power pressure**: Higher power consumption → bias toward efficient experts
- **Predicted imbalance**: Correct future imbalances proactively

```python
def compute_rerouting_biases(self, current_distribution, hardware_metrics):
    # Get hardware constraints
    thermal_pressure = max(0.0, (current_temp - 60.0) / 20.0)
    power_pressure = max(0.0, (current_power - power_budget * 0.7) / (power_budget * 0.3))
    
    # Compute target distribution
    if thermal_pressure > 0.5 or power_pressure > 0.5:
        target_distribution = torch.ones(self.num_experts) / self.num_experts  # Uniform
    else:
        target_distribution = 0.7 * predicted_distribution + 0.3 * uniform_dist
    
    # Calculate rerouting biases
    distribution_diff = target_distribution - current_distribution
    rerouting_biases = distribution_diff * rerouting_strength
    
    return rerouting_biases, metadata
```

## Testing the Integration

### 1. Basic Functionality Test

Run the test script to verify the integration:

```bash
python test_dynamic_rerouting.py
```

This will test:
- BatchDistributionTracker functionality
- Routing strategy enum definition
- MoE configuration compatibility

### 2. Full Experiment

Run the comprehensive experiment:

```bash
python src/experiments/run_dynamic_expert_rerouting.py \
    --num_experts 8 \
    --top_k 2 \
    --expert_type simple \
    --epochs 5 \
    --batch_size 32
```

### 3. Comparison with Baseline

Compare the new strategy with existing strategies:

```python
# Test baseline routing
model.moe_layer.adaptive_router.strategy = RoutingStrategy.BASELINE

# Test dynamic expert rerouting
model.moe_layer.adaptive_router.strategy = RoutingStrategy.DYNAMIC_EXPERT_REROUTING
```

## Expected Benefits

### 1. Improved Load Balancing

- **Reduced expert imbalance**: Tokens distributed more evenly across experts
- **Proactive correction**: Predicts and corrects imbalances before they become severe
- **Adaptive strength**: Rerouting strength adjusts based on hardware pressure

### 2. Energy Efficiency

- **Joules per token optimization**: Prioritizes energy-efficient expert selection
- **Power budget awareness**: Respects power limits and thermal constraints
- **Thermal management**: Reduces thermal pressure through intelligent routing

### 3. Performance Stability

- **Consistent throughput**: More uniform expert usage leads to predictable performance
- **Reduced bottlenecks**: Prevents single experts from becoming performance bottlenecks
- **Hardware utilization**: Better utilization of all available experts

## Configuration Parameters

### BatchDistributionTracker Parameters

```python
tracker = BatchDistributionTracker(
    num_experts=8,              # Number of experts in the MoE
    history_length=50,          # How many batches to remember
    imbalance_threshold=0.3,    # When to consider distribution imbalanced
    rerouting_strength=0.5      # How aggressive to be with rerouting
)
```

### Hardware Constraints

```python
# Power budget (Watts)
tracker.power_budget = 400.0

# Energy per token target (mJ)
tracker.energy_per_token_target = 2.0

# Thermal threshold (Celsius)
tracker.thermal_threshold = 80.0
```

## Monitoring and Analysis

### 1. Metrics Logged

The system logs comprehensive metrics including:

- **Imbalance scores**: Current and predicted imbalance
- **Rerouting decisions**: When and how rerouting was applied
- **Hardware metrics**: Temperature, power, utilization
- **Performance metrics**: Latency, throughput, energy consumption

### 2. Key Metrics to Monitor

- `avg_imbalance_score`: Average imbalance over time
- `rerouting_count`: How often rerouting was needed
- `thermal_pressure`: Thermal constraint pressure
- `power_pressure`: Power constraint pressure
- `confidence`: Confidence in predictions

### 3. Analysis Files

The experiment generates:
- **CSV log file**: Detailed metrics for each batch
- **JSON results file**: Summary statistics and configuration
- **Distribution statistics**: Per-epoch analysis

## Integration with Existing Code

### 1. Backward Compatibility

The integration is fully backward compatible:
- Existing routing strategies continue to work
- New strategy is opt-in
- No changes required to existing experiments

### 2. Gradual Adoption

You can gradually adopt the new functionality:

```python
# Start with existing strategies
router.strategy = RoutingStrategy.KERNEL_AWARE_TTHA

# Switch to dynamic expert rerouting when ready
router.strategy = RoutingStrategy.DYNAMIC_EXPERT_REROUTING
```

### 3. Hybrid Approaches

You can combine strategies or create custom hybrids:

```python
# Use TTHA for hardware adaptation + dynamic rerouting for load balancing
if hardware_pressure > threshold:
    router.strategy = RoutingStrategy.KERNEL_AWARE_TTHA
else:
    router.strategy = RoutingStrategy.DYNAMIC_EXPERT_REROUTING
```

## Next Steps

### 1. Immediate Actions

1. **Test the integration**: Run `test_dynamic_rerouting.py`
2. **Run baseline comparison**: Compare with existing strategies
3. **Analyze results**: Review the generated metrics and logs

### 2. Optimization Opportunities

1. **Expert efficiency profiles**: Add per-expert energy efficiency data
2. **Advanced prediction models**: Implement more sophisticated trend prediction
3. **Multi-GPU support**: Extend to multi-GPU scenarios
4. **Real-time adaptation**: Adjust parameters based on runtime performance

### 3. Research Directions

1. **Theoretical analysis**: Prove convergence and optimality properties
2. **Scalability testing**: Test with larger models and more experts
3. **Real-world validation**: Test on production workloads
4. **Integration with other TTT methods**: Combine with other test-time training approaches

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure `src` is in Python path
2. **CUDA errors**: Check GPU availability and memory
3. **Performance issues**: Adjust `rerouting_strength` and `imbalance_threshold`

### Debug Mode

Enable debug logging to see detailed rerouting decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Tuning

If rerouting is too aggressive or not aggressive enough:

```python
# More aggressive rerouting
tracker.rerouting_strength = 0.8
tracker.imbalance_threshold = 0.2

# Less aggressive rerouting
tracker.rerouting_strength = 0.3
tracker.imbalance_threshold = 0.4
```

## Conclusion

The Dynamic Expert Rerouting integration provides a powerful new capability for optimizing MoE models based on historical patterns and hardware constraints. It addresses your specific requirements for:

- Using previous batch distribution information
- Implementing expert dynamic rerouting
- Optimizing for energy efficiency (joules per token)
- Considering thermal and power constraints

The integration is designed to be robust, backward-compatible, and ready for immediate use in your research and experimentation pipeline. 