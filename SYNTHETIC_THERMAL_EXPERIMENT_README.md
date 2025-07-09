# Synthetic Thermal-Aware Energy-Adaptive Routing Experiment

This experiment tests energy-adaptive routing with thermal awareness using synthetic data and controlled thermal stress. The goal is to demonstrate how Test-Time Training (TTT) can adapt routing to conserve energy and manage thermal constraints.

## Overview

The experiment creates a synthetic environment with 8 experts where:
- **6 Cold Experts** (0-5): Operate at 0°C with low energy consumption
- **2 Hot Experts** (6-7): Operate at 60-80°C with high energy consumption

The router learns to prefer cold experts over time through TTT updates, demonstrating energy-aware adaptation.

## Key Features

### 1. Synthetic Thermal Stress
- **Cold Experts**: 0°C, 0.5x energy cost
- **Hot Experts**: 60-80°C, 2.0x energy cost
- Realistic thermal gradients for testing adaptation

### 2. Energy-Aware TTT Routing
- Per-expert energy penalties based on thermal state
- Adaptive penalty scaling with `lambda_energy` parameter
- Real-time feedback integration

### 3. Comprehensive Metrics
- **Expert Diversity**: Entropy of routing distribution
- **Thermal Imbalance**: Difference between cold/hot expert usage
- **Energy Savings**: Percentage reduction vs baseline
- **Routing Entropy**: Stability of routing decisions

### 4. Time-Series Analysis
- Expert usage patterns over time
- Temperature vs usage correlations
- Energy vs thermal imbalance relationships
- Cold vs hot expert comparison

## Files

### Core Experiment
- `src/experiments/run_synthetic_thermal_experiment.py`: Main experiment script
- `run_synthetic_thermal_experiment.slurm`: SLURM script for HPC execution
- `test_synthetic_thermal_local.py`: Local testing script

### Output
- `results/thermal_experiment/`: Experiment results and visualizations
- `logs/`: SLURM job logs

## Usage

### 1. Local Testing (Recommended First)

Test the experiment locally with smaller parameters:

```bash
python test_synthetic_thermal_local.py --lambda_energy 0.1 --num_batches 20
```

This will:
- Run a quick test with 20 batches
- Generate basic visualizations
- Save results to `test_results/`

### 2. Full Experiment on HPC

Submit the SLURM job for comprehensive testing:

```bash
sbatch run_synthetic_thermal_experiment.slurm
```

This will:
- Test multiple lambda values (0.0, 0.01, 0.1, 0.5, 1.0)
- Run 100 batches per lambda value
- Generate comprehensive visualizations
- Create summary report

### 3. Custom Parameters

Run with custom parameters:

```bash
python src/experiments/run_synthetic_thermal_experiment.py \
    --lambda_energy 0.2 \
    --num_experts 8 \
    --moe_top_k 2 \
    --batch_size 16 \
    --seq_length 128 \
    --d_model 512 \
    --num_batches 200 \
    --output_dir "results/custom_experiment"
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_energy` | 0.1 | Energy penalty weight (higher = more energy-aware) |
| `num_experts` | 8 | Number of experts (6 cold + 2 hot) |
| `moe_top_k` | 2 | Top-k experts to route to |
| `batch_size` | 8 | Batch size for processing |
| `seq_length` | 64 | Sequence length |
| `d_model` | 768 | Model dimension |
| `num_batches` | 100 | Number of batches to process |

## Expected Results

### Energy Awareness
- **Lambda = 0.0**: No energy penalty, uniform expert usage
- **Lambda > 0.0**: Increasing preference for cold experts
- **Higher Lambda**: More aggressive energy optimization

### Expert Diversity
- Should increase over time as router learns
- Cold experts should be used more frequently
- Hot experts should be used less frequently

### Energy Savings
- Positive energy savings with lambda > 0
- Savings increase with higher lambda values
- Trade-off between energy savings and routing diversity

## Visualizations

The experiment generates several key visualizations:

### 1. Expert Usage Over Time
- Shows how expert usage changes over time
- Separate plots for cold vs hot experts
- Demonstrates adaptation learning

### 2. Temperature vs Usage Scatter
- Correlation between expert temperature and usage
- Cold experts should cluster at low temperature, high usage
- Hot experts should cluster at high temperature, low usage

### 3. Expert Diversity Growth
- Entropy of routing distribution over time
- Should show increasing diversity as router adapts
- Trend line shows learning rate

### 4. Energy vs Thermal Imbalance
- Relationship between energy consumption and thermal balance
- Color-coded by time step
- Shows optimization trade-offs

### 5. Cold vs Hot Expert Comparison
- Four-panel comparison of cold vs hot experts
- Usage patterns, energy consumption, ratios, efficiency
- Clear demonstration of adaptation

## Analysis

### Key Metrics to Monitor

1. **Energy Savings**: Should be positive with lambda > 0
2. **Expert Diversity**: Should increase over time
3. **Thermal Imbalance**: Should decrease (more cold usage)
4. **TTT Updates**: Number of router updates performed

### Expected Trends

- **Early Stages**: Random routing, high thermal imbalance
- **Middle Stages**: Learning phase, increasing cold expert usage
- **Late Stages**: Stable adaptation, consistent energy savings

### Lambda Sensitivity

- **Low Lambda (0.01)**: Subtle energy awareness
- **Medium Lambda (0.1)**: Balanced optimization
- **High Lambda (1.0)**: Aggressive energy optimization

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Errors**: Check GPU availability and memory
3. **Memory Issues**: Reduce batch_size or seq_length
4. **No Energy Savings**: Check lambda_energy value

### Debug Mode

Add debug prints to the router:

```python
# In EnergyAwareTTTRouter.forward()
if self.ttt_update_count % 10 == 0:
    print(f"Energy penalty: {expert_penalties}")
    print(f"Expert usage: {expert_usage}")
```

## Future Extensions

1. **Real Hardware Integration**: Replace synthetic temperatures with real GPU monitoring
2. **Dynamic Lambda**: Adaptive lambda based on thermal state
3. **Multi-GPU**: Distributed expert placement
4. **Thermal Prediction**: Predictive thermal modeling
5. **Load Balancing**: Advanced load balancing strategies

## References

- Energy-Aware TTT Router: `models/ttt_router.py`
- Thermal Signal Processing: `src/thermal_signal.py`
- GPU Monitoring: `src/monitor.py`
- Previous Energy Tests: `test_energy_awareness_real.py` 