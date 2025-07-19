#!/bin/bash
#SBATCH --job-name=parallel_energy_moe
#SBATCH --output=logs/parallel_energy_moe_%j.out
#SBATCH --error=logs/parallel_energy_moe_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G

# Comprehensive SLURM script for parallel energy-aware MoE training
# Supports multi-GPU execution with energy optimization and dynamic expert rerouting

echo "=== Parallel Energy-Aware MoE Training Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Create necessary directories
mkdir -p logs
mkdir -p results/parallel_energy_moe

# Load modules (adjust for your cluster)
module load cuda/11.8
module load python/3.9
module load gcc/9.3.0

# Activate virtual environment (adjust path as needed)
source venv/bin/activate

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# GPU and system information
echo "=== System Information ==="
nvidia-smi
echo "CUDA Version: $(nvcc --version)"
echo "Python Version: $(python --version)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Configuration parameters
D_MODEL=768
NUM_EXPERTS=8
TOP_K=2
EXPERT_TYPE="swiglu"
BATCH_SIZE=32
SEQ_LENGTH=512
NUM_EPOCHS=10
NUM_SAMPLES=50000

# Energy optimization parameters
ENERGY_BUDGET=1600.0  # 400W per GPU * 4 GPUs
THERMAL_THRESHOLD=80.0
JOULES_PER_TOKEN_TARGET=0.002

# Parallelization parameters
WORLD_SIZE=4  # Number of GPUs
EXPERT_PARALLEL=2
DATA_PARALLEL=2

# TTT parameters
TTT_CHUNK_SIZE=2048

# Output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/parallel_energy_moe_${TIMESTAMP}"

echo "=== Training Configuration ==="
echo "Model dimension: $D_MODEL"
echo "Number of experts: $NUM_EXPERTS"
echo "Top-k: $TOP_K"
echo "Expert type: $EXPERT_TYPE"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LENGTH"
echo "Number of epochs: $NUM_EPOCHS"
echo "Number of samples: $NUM_SAMPLES"
echo "Energy budget: $ENERGY_BUDGET W"
echo "Thermal threshold: $THERMAL_THRESHOLD °C"
echo "Target J/token: $JOULES_PER_TOKEN_TARGET"
echo "World size: $WORLD_SIZE"
echo "Output directory: $OUTPUT_DIR"

# Run the main training script
echo "=== Starting Training ==="
python src/experiments/run_parallel_energy_moe.py \
    --d_model $D_MODEL \
    --num_experts $NUM_EXPERTS \
    --top_k $TOP_K \
    --expert_type $EXPERT_TYPE \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --num_epochs $NUM_EPOCHS \
    --num_samples $NUM_SAMPLES \
    --world_size $WORLD_SIZE \
    --expert_parallel $EXPERT_PARALLEL \
    --data_parallel $DATA_PARALLEL \
    --energy_budget $ENERGY_BUDGET \
    --thermal_threshold $THERMAL_THRESHOLD \
    --joules_per_token_target $JOULES_PER_TOKEN_TARGET \
    --enable_rerouting \
    --enable_ttt \
    --async_execution \
    --mixed_precision \
    --ttt_chunk_size $TTT_CHUNK_SIZE \
    --output_dir $OUTPUT_DIR

TRAINING_EXIT_CODE=$?

echo "=== Training Completed ==="
echo "Exit code: $TRAINING_EXIT_CODE"
echo "End time: $(date)"

# Post-training analysis and visualization
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "=== Running Post-Training Analysis ==="
    
    # Run energy analysis
    python -c "
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
if (output_dir / 'training_results.json').exists():
    with open(output_dir / 'training_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    batch_metrics = results.get('batch_metrics', [])
    if batch_metrics:
        power_values = [m.get('avg_power_watts', 0) for m in batch_metrics]
        temp_values = [m.get('avg_temperature', 0) for m in batch_metrics]
        joules_per_token = [m.get('joules_per_token', 0) for m in batch_metrics]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Power consumption over time
        axes[0, 0].plot(power_values)
        axes[0, 0].set_title('Power Consumption Over Time')
        axes[0, 0].set_ylabel('Power (W)')
        axes[0, 0].set_xlabel('Batch')
        axes[0, 0].axhline(y=$ENERGY_BUDGET, color='r', linestyle='--', label='Budget')
        axes[0, 0].legend()
        
        # Temperature over time
        axes[0, 1].plot(temp_values)
        axes[0, 1].set_title('Temperature Over Time')
        axes[0, 1].set_ylabel('Temperature (°C)')
        axes[0, 1].set_xlabel('Batch')
        axes[0, 1].axhline(y=$THERMAL_THRESHOLD, color='r', linestyle='--', label='Threshold')
        axes[0, 1].legend()
        
        # Energy efficiency over time
        axes[1, 0].plot(joules_per_token)
        axes[1, 0].set_title('Energy Efficiency Over Time')
        axes[1, 0].set_ylabel('Joules per Token')
        axes[1, 0].set_xlabel('Batch')
        axes[1, 0].axhline(y=$JOULES_PER_TOKEN_TARGET, color='r', linestyle='--', label='Target')
        axes[1, 0].legend()
        
        # Power vs Temperature scatter
        axes[1, 1].scatter(power_values, temp_values, alpha=0.6)
        axes[1, 1].set_title('Power vs Temperature')
        axes[1, 1].set_xlabel('Power (W)')
        axes[1, 1].set_ylabel('Temperature (°C)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Energy analysis plots saved to {output_dir}/energy_analysis.png')
        
        # Print summary statistics
        print('=== Energy Efficiency Summary ===')
        print(f'Average Power: {np.mean(power_values):.1f} W (Budget: $ENERGY_BUDGET W)')
        print(f'Max Temperature: {np.max(temp_values):.1f} °C (Threshold: $THERMAL_THRESHOLD °C)')
        print(f'Average J/token: {np.mean(joules_per_token):.6f} (Target: $JOULES_PER_TOKEN_TARGET)')
        print(f'Energy Efficiency: {(1 - np.mean(joules_per_token)/$JOULES_PER_TOKEN_TARGET) * 100:.1f}% improvement')
"
    
    # Generate comprehensive report
    python -c "
import json
import time
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
report_file = output_dir / 'experiment_report.md'

with open(report_file, 'w') as f:
    f.write('# Parallel Energy-Aware MoE Experiment Report\n\n')
    f.write(f'**Job ID:** $SLURM_JOB_ID\n')
    f.write(f'**Node:** $SLURM_NODELIST\n')
    f.write(f'**Timestamp:** $(date)\n\n')
    
    f.write('## Configuration\n\n')
    f.write(f'- Model Dimension: $D_MODEL\n')
    f.write(f'- Number of Experts: $NUM_EXPERTS\n')
    f.write(f'- Top-k: $TOP_K\n')
    f.write(f'- Expert Type: $EXPERT_TYPE\n')
    f.write(f'- Batch Size: $BATCH_SIZE\n')
    f.write(f'- Sequence Length: $SEQ_LENGTH\n')
    f.write(f'- Number of Epochs: $NUM_EPOCHS\n')
    f.write(f'- Energy Budget: $ENERGY_BUDGET W\n')
    f.write(f'- Thermal Threshold: $THERMAL_THRESHOLD °C\n')
    f.write(f'- Target J/token: $JOULES_PER_TOKEN_TARGET\n\n')
    
    f.write('## Features Enabled\n\n')
    f.write('- ✅ Dynamic Expert Rerouting\n')
    f.write('- ✅ Test-Time Training (TTT)\n')
    f.write('- ✅ Async Expert Execution\n')
    f.write('- ✅ Mixed Precision Training\n')
    f.write('- ✅ Multi-GPU Parallelization\n\n')
    
    f.write('## Key Innovations\n\n')
    f.write('1. **Previous Batch Distribution Analysis**: Uses historical routing patterns to predict and correct future expert imbalances\n')
    f.write('2. **Energy-Aware Optimization**: Optimizes for joules per token rather than just throughput\n')
    f.write('3. **Thermal-Aware Routing**: Considers GPU temperature and power consumption in routing decisions\n')
    f.write('4. **Dynamic Expert Rerouting**: Adjusts expert assignments based on hardware constraints and energy efficiency\n')
    f.write('5. **Parallel Execution**: Distributes experts across multiple GPUs with load balancing\n\n')
    
    # Load and include results if available
    if (output_dir / 'training_results.json').exists():
        with open(output_dir / 'training_results.json', 'r') as rf:
            results = json.load(rf)
        
        final_stats = results.get('final_moe_stats', {})
        f.write('## Results\n\n')
        f.write(f'- **Energy Efficiency**: {final_stats.get(\"joules_per_token\", 0):.6f} J/token\n')
        f.write(f'- **Average Power**: {final_stats.get(\"avg_power_watts\", 0):.1f} W\n')
        f.write(f'- **Average Temperature**: {final_stats.get(\"avg_temperature_celsius\", 0):.1f} °C\n')
        f.write(f'- **Energy Improvement**: {final_stats.get(\"energy_efficiency_improvement\", 0):.1f}%\n')
        f.write(f'- **Thermal Efficiency**: {final_stats.get(\"thermal_efficiency\", 0):.1f}%\n')
        f.write(f'- **Total Batches Processed**: {final_stats.get(\"batch_count\", 0)}\n\n')
    
    f.write('## Files Generated\n\n')
    f.write('- `training_results.json`: Detailed training metrics and configuration\n')
    f.write('- `energy_analysis.json`: Energy efficiency analysis\n')
    f.write('- `energy_analysis.png`: Visualization of energy metrics\n')
    f.write('- `experiment_report.md`: This comprehensive report\n\n')
    
    f.write('## Next Steps\n\n')
    f.write('1. Analyze the energy efficiency improvements compared to baseline\n')
    f.write('2. Examine the dynamic expert rerouting decisions and their impact\n')
    f.write('3. Evaluate the thermal management effectiveness\n')
    f.write('4. Scale to larger models and more GPUs\n')
    f.write('5. Compare with other energy optimization approaches\n')

print(f'Comprehensive experiment report generated: {report_file}')
"
    
    echo "=== Post-Training Analysis Completed ==="
else
    echo "=== Training Failed ==="
    echo "Check the error logs for details"
fi

# System resource usage summary
echo "=== Resource Usage Summary ==="
echo "GPU Memory Usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

echo "=== Job Completed ==="
echo "Total job time: $(($(date +%s) - $SLURM_JOB_START_TIME)) seconds"
echo "Output directory: $OUTPUT_DIR"
echo "Log files: logs/parallel_energy_moe_${SLURM_JOB_ID}.{out,err}"