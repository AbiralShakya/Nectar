# HPC Testing Guide for Energy-Aware TTT with Kernel Cost Model

This guide explains how to test the Energy-Aware TTT system with kernel cost model integration on HPC clusters.

## 🎯 What This Test Does

The HPC test compares three approaches on a single GPU:

1. **Simple Transformer (Baseline)** - No TTT, no hardware awareness
2. **Simple Transformer + TTT** - Traditional test-time training
3. **MoE + EnergyAwareTTTRouter** - Our novel hardware-aware approach

## 📊 Metrics Measured

- **Latency** (ms) - Inference time per batch
- **Memory Usage** (GB) - GPU memory consumption  
- **Power Consumption** (W) - Estimated GPU power usage
- **Temperature** (°C) - Estimated GPU temperature
- **Energy per Token** (J) - Energy efficiency metric
- **Throughput** (tokens/sec) - Processing speed
- **TTT Update Count** - Number of test-time training updates

## 🔧 **Kernel Cost Model Integration**

### **What is the Kernel Cost Model?**

The **Kernel Cost Model (KCM)** is a sophisticated system that predicts the energy, latency, and thermal impact of different GPU operations based on:

- **Operation Type**: FFN, attention, MoE routing, quantization, etc.
- **Batch Size**: How operations scale with input size
- **Hardware State**: Current temperature, memory pressure, GPU utilization
- **GPU Specifications**: A100, H100, V100 characteristics

### **How KCM Works in the HPC Test**

The enhanced HPC test script (`hpc_single_gpu_test.py`) now includes comprehensive KCM testing:

#### **1. KCM Initialization and Testing**
```python
# Initialize kernel cost model
kernel_cost_model = KernelCostModel(gpu_type="A100")

# Test different operation types
test_ops = ["ffn_gate", "ffn_up", "ffn_down", "attention_qk", "attention_av", "moe_router"]
for op in test_ops:
    for bs in [1, 8, 32, 128]:
        cost = kernel_cost_model.get_cost(op, bs)
        # Returns: energy_joules, latency_ms, temp_impact, memory_gb, etc.
```

#### **2. Thermal Throttling Effects**
```python
# Test thermal impact on performance
normal_cost = kernel_cost_model.get_cost("ffn_gate", 32, current_temp=50.0)
hot_cost = kernel_cost_model.get_cost("ffn_gate", 32, current_temp=85.0)
# Shows: +X% latency, +Y% energy at high temperatures
```

#### **3. Memory Pressure Effects**
```python
# Test memory pressure impact
low_mem_cost = kernel_cost_model.get_cost("ffn_gate", 32, memory_pressure=0.3)
high_mem_cost = kernel_cost_model.get_cost("ffn_gate", 32, memory_pressure=0.9)
# Shows: +X% latency, +Y% energy under memory pressure
```

#### **4. Energy Prediction Accuracy**
The test measures how well KCM predicts actual energy consumption:
```python
# Compare predicted vs actual energy
predicted_energy = kernel_cost_model.get_cost(op, batch_size)['energy_joules']
actual_energy = measured_power * latency_ms / 1000
accuracy = abs(predicted_energy - actual_energy) / actual_energy
```

### **KCM Integration in EnergyAwareTTTRouter**

The `EnergyAwareTTTRouter` uses KCM in several ways:

#### **1. Base Routing Biases**
```python
def _compute_base_biases(self, num_tokens: int, gpu_stats: Dict[str, Any]):
    # Calculate energy costs for each expert
    for expert_id in range(self.num_experts):
        for op_name in expert_ops:
            op_costs = self.kernel_cost_model.get_cost(
                op_name, int(effective_expert_token_batch),
                current_temp=current_temp, 
                memory_pressure=memory_pressure
            )
            total_energy += op_costs.get('energy_joules', 0.0)
        
        # Create bias favoring energy-efficient experts
        energy_bias = -total_energy * self.objective_weights['energy']
```

#### **2. Dynamic Adaptation**
```python
# Real-time hardware state monitoring
gpu_stats = self.gpu_system_monitor.get_current_stats()
current_temp = gpu_stats.get('temperature', 50.0)
memory_pressure = gpu_stats.get('memory_utilization_percent', 0.0) / 100.0

# Adjust costs based on current conditions
adjusted_costs = self.kernel_cost_model.get_cost(
    op_name, batch_size,
    current_temp=current_temp,
    memory_pressure=memory_pressure
)
```

#### **3. Thermal-Safe Batch Sizing**
```python
# Recommend safe batch sizes to avoid thermal throttling
safe_batch = kernel_cost_model.get_thermal_safe_batch_size(
    op_type, current_temp=80.0, max_temp_increase=3.0
)
```

## 🚀 **Running the Enhanced HPC Test**

### **Quick Start**

```bash
# Run on local machine
python hpc_single_gpu_test.py --test_moe --num_runs 20

# Run on SLURM cluster
sbatch run_hpc_test.slurm

# Run on PBS cluster  
qsub run_hpc_test.pbs
```

### **Test Parameters**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--d_model` | Model dimension | 256 |
| `--num_layers` | Number of transformer layers | 4 |
| `--batch_size` | Batch size for testing | 8 |
| `--seq_length` | Sequence length | 512 |
| `--num_runs` | Number of performance measurements | 20 |
| `--test_moe` | Enable MoE testing | False |
| `--num_experts` | Number of experts in MoE | 4 |
| `--top_k` | Top-k expert selection | 2 |
| `--ttt_chunk_size` | TTT chunk size | 512 |
| `--ttt_update_frequency` | TTT update frequency | 128 |
| `--energy_aware_lr` | Energy-aware learning rate | 1e-4 |
| `--muon_enabled` | Enable Muon TTT optimizer | False |

### **Expected Output**

The test will produce detailed output including:

#### **1. Kernel Cost Model Testing**
```
🔧 Testing Kernel Cost Model Integration
==================================================
Testing operation costs across batch sizes:

  ffn_gate:
    batch_size=  1: energy=0.000123J, latency=0.045ms, temp_impact=0.002°C
    batch_size=  8: energy=0.000456J, latency=0.123ms, temp_impact=0.008°C
    batch_size= 32: energy=0.001234J, latency=0.345ms, temp_impact=0.023°C
    batch_size=128: energy=0.003456J, latency=0.789ms, temp_impact=0.067°C

Testing thermal throttling effects:
  ffn_gate (batch_size=32):
    Normal temp (50°C): 0.345ms, 0.001234J
    Hot temp (85°C): 0.567ms, 0.001456J
    Thermal impact: +64.3% latency, +18.0% energy

Testing memory pressure effects:
  ffn_gate (batch_size=32):
    Low memory pressure (30%): 0.345ms, 0.001234J
    High memory pressure (90%): 0.456ms, 0.001345J
    Memory impact: +32.2% latency, +9.0% energy
```

#### **2. Performance Comparison**
```
📈 RESULTS COMPARISON
======================================================================
Simple Transformer:
  Latency: 12.34 ± 0.45 ms
  Memory: 0.234 GB
  Power: 185.6 W
  Temperature: 67.8°C
  Energy per token: 0.000123 J
  Throughput: 414.7 tokens/sec

Simple Transformer + TTT:
  Latency: 13.45 ± 0.52 ms
  Memory: 0.245 GB
  Power: 192.3 W
  Temperature: 69.2°C
  Energy per token: 0.000134 J
  Throughput: 380.7 tokens/sec
  TTT updates: 15
  Predicted energy: 0.000131 J
  Energy prediction accuracy: 2.2%

MoE + EnergyAwareTTTRouter:
  Latency: 11.23 ± 0.38 ms
  Memory: 0.198 GB
  Power: 178.9 W
  Temperature: 65.4°C
  Energy per token: 0.000098 J
  Throughput: 455.9 tokens/sec
  Predicted energy: 0.000101 J
  Energy prediction accuracy: 3.1%
```

#### **3. Improvement Analysis**
```
🎯 IMPROVEMENTS OVER BASELINE
======================================================================
MoE + EnergyAwareTTTRouter vs Baseline:
  Latency: +9.0%
  Power: +3.6%
  Energy per token: +20.3%
  Throughput: +9.9%
```

## 📊 **Output Files**

The test generates several output files:

### **1. `hpc_results.json`**
Detailed performance metrics for each model variant.

### **2. `kernel_cost_model_test_results.json`**
KCM testing results including:
- Operation costs across batch sizes
- Thermal throttling factors
- Memory pressure factors
- Thermal-safe batch size recommendations

### **3. `hpc_summary.json`**
Summary of the entire test including:
- Hardware information
- Test parameters
- Overall results
- KCM availability and test results

## 🔍 **Understanding KCM Results**

### **Operation Cost Breakdown**

The KCM provides costs for different operation types:

| Operation | Description | Typical Energy | Typical Latency |
|-----------|-------------|----------------|-----------------|
| `ffn_gate` | Feed-forward gate | 0.001-0.005J | 0.1-1.0ms |
| `ffn_up` | Feed-forward up projection | 0.002-0.008J | 0.2-1.5ms |
| `ffn_down` | Feed-forward down projection | 0.002-0.008J | 0.2-1.5ms |
| `attention_qk` | Attention Q*K computation | 0.001-0.004J | 0.1-0.8ms |
| `attention_av` | Attention A*V computation | 0.001-0.004J | 0.1-0.8ms |
| `moe_router` | MoE routing computation | 0.0001-0.001J | 0.01-0.1ms |

### **Thermal Impact Analysis**

The KCM models thermal effects:

- **Normal Temperature (50°C)**: Baseline performance
- **High Temperature (85°C)**: Up to 100% latency increase
- **Thermal Throttling**: Automatic performance reduction to prevent overheating

### **Memory Pressure Effects**

Memory pressure impacts performance:

- **Low Pressure (<50%)**: Normal performance
- **High Pressure (>80%)**: Up to 60% latency increase
- **Memory Bandwidth**: Becomes bottleneck under high pressure

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **KCM Import Error**
   ```
   Warning: KernelCostModel not available: No module named 'src.kernelcostmodel'
   ```
   **Solution**: Ensure the `src` directory is in your Python path.

2. **GPU Memory Issues**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Reduce batch size or sequence length.

3. **SLURM GPU Allocation**
   ```
   RuntimeError: CUDA error: no CUDA-capable device is detected
   ```
   **Solution**: Check SLURM GPU allocation with `squeue -j $SLURM_JOB_ID`.

### **Performance Tuning**

1. **For Better Energy Efficiency**:
   - Increase `--num_experts` to 8 or 16
   - Use `--top_k 1` for single expert selection
   - Enable `--muon_enabled` for optimized TTT

2. **For Higher Throughput**:
   - Increase `--batch_size` to 16 or 32
   - Reduce `--seq_length` to 256
   - Use smaller `--d_model` like 128

3. **For Thermal Management**:
   - Monitor temperature in output logs
   - Reduce batch size if thermal throttling occurs
   - Check thermal-safe batch size recommendations

## 📈 **Interpreting Results**

### **Key Metrics to Monitor**

1. **Energy per Token**: Lower is better for efficiency
2. **Energy Prediction Accuracy**: Should be <10% for good KCM
3. **Thermal Impact**: Should be minimal (<5°C increase)
4. **Memory Utilization**: Should be <80% to avoid pressure
5. **Throughput**: Higher is better for performance

### **Success Criteria**

- ✅ Energy per token reduced by >15% vs baseline
- ✅ Energy prediction accuracy >90%
- ✅ No thermal throttling observed
- ✅ Memory utilization <80%
- ✅ Throughput maintained or improved

## 🔬 **Advanced Testing**

### **Custom KCM Testing**

You can test specific KCM scenarios:

```python
# Test specific operation
cost = kernel_cost_model.get_cost("ffn_gate", 64, current_temp=75.0, memory_pressure=0.8)

# Test thermal-safe batch size
safe_batch = kernel_cost_model.get_thermal_safe_batch_size("attention_qk", 80.0, 2.0)

# Get cost breakdown
breakdown = kernel_cost_model.get_cost_breakdown("ffn_gate", 32)
```

### **Multi-GPU Testing**

For multi-GPU setups, modify the SLURM script:

```bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
```

And update the test script to handle multiple GPUs.

## 📚 **Further Reading**

- [Kernel Cost Model Documentation](src/kernelcostmodel.py)
- [EnergyAwareTTTRouter Implementation](src/routers.py)
- [MoE Models](src/moe_models.py)
- [Hardware Monitoring](src/monitor.py)

---

**Note**: The kernel cost model is essential for the energy-aware routing system. It provides the foundation for making intelligent routing decisions based on hardware state and operation characteristics. 