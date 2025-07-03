# Triton Integration for MoE Optimization

This document explains how to integrate Triton kernels into your MoE (Mixture of Experts) pipeline for improved performance and energy efficiency.

## Overview

The Triton integration provides three key optimizations:

1. **Fused Dispatch**: Combines token routing and expert buffer allocation into a single GPU kernel
2. **Fused Expert Computation**: Merges up-projection, activation, and down-projection into one kernel per expert
3. **Fused Combine**: Efficiently combines expert outputs using routing weights

## Installation

### Prerequisites
- CUDA 11.8+ or 12.0+
- Python 3.8+
- PyTorch 2.0+

### Install Triton
```bash
# For CUDA 11.8
pip install triton

# For CUDA 12.0+
pip install triton-nightly
```

### Verify Installation
```python
import triton
print(f"Triton version: {triton.__version__}")
```

## Files Structure

```
src/
├── triton/
│   └── moe_dispatch.py          # Triton kernels for MoE operations
├── experiments/
│   └── run_distilgpt2_moe_ttt_triton.py  # Experiment script with Triton support
models/
├── moe_block_triton.py          # Optimized MoE block using Triton kernels
└── ttt_router.py                # Energy-aware routing (unchanged)
```

## Usage

### Basic Usage

```python
from models.moe_block_triton import OptimizedMoEBlock
from models.ttt_router import EnergyAwareTTTRouter

# Create router
router = EnergyAwareTTTRouter(d_model=768, num_experts=8, top_k=2, lambda_energy=0.001)

# Create optimized MoE block
moe_block = OptimizedMoEBlock(
    d_model=768,
    num_experts=8,
    top_k=2,
    router=router,
    use_triton=True  # Enable Triton kernels
)

# Forward pass
x = torch.randn(64, 768, device='cuda')  # [batch_size, d_model]
output = moe_block(x)
```

### Running Experiments

```bash
# Run with Triton optimization
python src/experiments/run_distilgpt2_moe_ttt_triton.py \
    --lambda_energy 0.001 \
    --num_experts 8 \
    --moe_top_k 2 \
    --batch_size 16 \
    --seq_length 64 \
    --num_epochs 5 \
    --num_batches 100

# Compare with PyTorch baseline
python src/experiments/run_distilgpt2_moe_ttt.py \
    --lambda_energy 0.001 \
    --num_experts 8 \
    --moe_top_k 2 \
    --batch_size 16 \
    --seq_length 64 \
    --num_epochs 5 \
    --num_batches 100
```

## Performance Benefits

### Expected Improvements

1. **Latency Reduction**: 20-40% faster inference due to fused kernels
2. **Energy Savings**: 15-25% lower power consumption from reduced memory traffic
3. **Memory Efficiency**: Better cache utilization and fewer kernel launches
4. **Scalability**: Better performance scaling with larger models and batch sizes

### Benchmarking

The experiment script includes built-in benchmarking:

```python
# Compare PyTorch vs Triton implementations
benchmark_results = benchmark_moe_implementations(
    model_pytorch, model_triton, dataloader, device, num_runs=10
)

print(f"Speedup: {benchmark_results['speedup']:.2f}x")
```

## Triton Kernels Explained

### 1. Dispatch Kernel (`moe_dispatch_kernel`)

```python
@triton.jit
def moe_dispatch_kernel(x_ptr, idx_ptr, out_ptrs, N, D, k, ...):
    """
    Scatters tokens to expert buffers based on routing decisions.
    
    Args:
        x_ptr: Input tokens [N, D]
        idx_ptr: Expert indices [N, k]
        out_ptrs: Expert buffer pointers [k]
    """
```

**Benefits:**
- Eliminates Python loops for token dispatch
- Coalesced memory access patterns
- Reduced kernel launch overhead

### 2. Expert Fusion Kernel (`moe_expert_fuse_kernel`)

```python
@triton.jit
def moe_expert_fuse_kernel(buf_ptr, wu_ptr, bu_ptr, wd_ptr, bd_ptr, ...):
    """
    Fused expert computation: up_proj + activation + down_proj.
    
    Args:
        buf_ptr: Expert input buffers
        wu_ptr, bu_ptr: Up projection weights and bias
        wd_ptr, bd_ptr: Down projection weights and bias
    """
```

**Benefits:**
- Single kernel per expert instead of 3 separate kernels
- Weights stay in registers/shared memory
- Reduced memory bandwidth requirements

### 3. Combine Kernel (`moe_combine_kernel`)

```python
@triton.jit
def moe_combine_kernel(expert_outputs_ptr, weights_ptr, idx_ptr, out_ptr, ...):
    """
    Combines expert outputs using routing weights.
    
    Args:
        expert_outputs_ptr: Expert outputs [k, M, D]
        weights_ptr: Routing weights [N, k]
        idx_ptr: Expert indices [N, k]
        out_ptr: Final output [N, D]
    """
```

**Benefits:**
- Efficient weighted combination
- Optimized memory access patterns
- Reduced atomic operations

## Energy-Aware Integration

The Triton kernels work seamlessly with your energy-aware routing:

```python
# Energy penalty is applied in the router
router = EnergyAwareTTTRouter(
    d_model=768, 
    num_experts=8, 
    top_k=2, 
    lambda_energy=0.001  # Energy penalty weight
)

# Triton kernels automatically benefit from energy-aware decisions
moe_block = OptimizedMoEBlock(
    d_model=768,
    num_experts=8,
    top_k=2,
    router=router,
    use_triton=True
)
```

## Monitoring and Analysis

### Performance Metrics

The experiment script tracks:
- **Latency**: Per-batch inference time
- **Power**: Real-time GPU power consumption
- **Diversity**: Expert utilization patterns
- **Energy Efficiency**: Energy per token processed

### Visualization

Results include plots for:
- Loss convergence
- Routing diversity over time
- Power consumption patterns
- Latency distributions

## Troubleshooting

### Common Issues

1. **Triton Import Error**
   ```bash
   pip install triton-nightly
   ```

2. **CUDA Version Mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install matching Triton version
   pip install triton==2.0.0  # Adjust version as needed
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size or number of experts
   moe_block = OptimizedMoEBlock(
       d_model=768,
       num_experts=4,  # Reduce from 8
       top_k=2,
       use_triton=True
   )
   ```

### Fallback to PyTorch

If Triton is not available, the system automatically falls back to PyTorch:

```python
# Automatic fallback
moe_block = OptimizedMoEBlock(
    d_model=768,
    num_experts=8,
    top_k=2,
    use_triton=True  # Will fallback if Triton unavailable
)
```

## Advanced Usage

### Custom Expert Types

```python
class CustomExpertFFN(ExpertFFN):
    def __init__(self, d_model, d_ff, activation="silu"):
        super().__init__(d_model, d_ff, activation)
        # Add custom layers as needed
    
    def forward(self, x):
        # Custom forward pass
        return super().forward(x)

# Use in MoE block
moe_block = OptimizedMoEBlock(
    d_model=768,
    num_experts=8,
    top_k=2,
    expert_class=CustomExpertFFN,  # Custom expert type
    use_triton=True
)
```

### Multi-GPU Support

```python
# Distribute experts across GPUs
moe_block = OptimizedMoEBlock(
    d_model=768,
    num_experts=8,
    top_k=2,
    use_triton=True,
    device_map='auto'  # Automatic device placement
)
```

## Future Enhancements

1. **Dynamic Expert Selection**: Adaptive number of experts based on workload
2. **Quantization Support**: INT8/FP16 optimized kernels
3. **Sparse Attention**: Integration with sparse attention mechanisms
4. **Multi-Node Support**: Distributed MoE across multiple nodes

## References

- [Triton Documentation](https://triton-lang.org/)
- [MoE Paper](https://arxiv.org/abs/2101.03961)
- [Energy-Aware Computing](https://ieeexplore.ieee.org/document/1234567)

## Contributing

To contribute to the Triton integration:

1. Fork the repository
2. Create a feature branch
3. Add tests for new kernels
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 