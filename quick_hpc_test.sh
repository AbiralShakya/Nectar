#!/bin/bash
# Quick HPC Test Setup Script
# This script helps you quickly set up and run the EnergyAwareTTTRouter test on HPC clusters

set -e  # Exit on any error

echo "🚀 EnergyAwareTTTRouter HPC Test Setup"
echo "======================================"

# Check if we're on a cluster
if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "✅ Detected SLURM cluster environment"
    JOB_ID=$SLURM_JOB_ID
    CLUSTER_TYPE="SLURM"
elif [[ -n "$PBS_JOBID" ]]; then
    echo "✅ Detected PBS cluster environment"
    JOB_ID=$PBS_JOBID
    CLUSTER_TYPE="PBS"
elif [[ -n "$LSB_JOBID" ]]; then
    echo "✅ Detected LSF cluster environment"
    JOB_ID=$LSB_JOBID
    CLUSTER_TYPE="LSF"
else
    echo "ℹ️  Running on local machine"
    JOB_ID=$(date +%s)
    CLUSTER_TYPE="LOCAL"
fi

echo "Job ID: $JOB_ID"
echo "Cluster Type: $CLUSTER_TYPE"

# Create output directory
OUTPUT_DIR="results/hpc_test_${JOB_ID}"
mkdir -p $OUTPUT_DIR

echo "📁 Output directory: $OUTPUT_DIR"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Checking GPU availability..."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  nvidia-smi not available, using PyTorch GPU detection"
fi

# Check Python and PyTorch
echo "🐍 Checking Python environment..."
python --version

if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "✅ PyTorch is available"
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
        echo "✅ CUDA is available"
    else
        echo "⚠️  CUDA not available, will run on CPU"
    fi
else
    echo "❌ PyTorch not found. Installing..."
    pip install torch numpy pandas matplotlib
fi

# Run the test
echo "🧪 Running HPC TTT comparison test..."

python hpc_single_gpu_test.py \
    --d_model 256 \
    --num_layers 4 \
    --batch_size 8 \
    --seq_length 512 \
    --num_runs 20 \
    --test_moe \
    --num_experts 4 \
    --top_k 2 \
    --ttt_chunk_size 512 \
    --ttt_update_frequency 128 \
    --energy_aware_lr 1e-4 \
    --muon_enabled \
    --output_dir "$OUTPUT_DIR"

echo "✅ Test completed!"
echo "📊 Results saved to: $OUTPUT_DIR"

# Show summary if available
if [[ -f "$OUTPUT_DIR/hpc_summary.json" ]]; then
    echo "📋 Test Summary:"
    python -c "
import json
with open('$OUTPUT_DIR/hpc_summary.json', 'r') as f:
    summary = json.load(f)
print(f'Device: {summary[\"device\"]}')
print(f'GPU: {summary[\"gpu_name\"]}')
print(f'GPU Memory: {summary[\"gpu_memory_gb\"]:.1f} GB')
print(f'Test Parameters: {summary[\"test_parameters\"]}')
"
fi

echo "🎉 Setup and test completed successfully!" 