#!/bin/bash

# SLURM Directives for Princeton Della
#SBATCH --job-name=nectar_unit_tests     
#SBATCH --output=logs/nectar_unit_tests_%j.out 
#SBATCH --error=logs/nectar_unit_tests_%j.err  
#SBATCH --nodes=1                         
#SBATCH --ntasks-per-node=1               
#SBATCH --cpus-per-task=4               
#SBATCH --mem=16G                      
#SBATCH --time=00:15:00                  
#SBATCH --gres=gpu:1                      

# --- Setup Environment ---
# Purge existing modules for a clean environment
module purge

# Load necessary modules (adjust versions to Della's available ones)
module load anaconda3/2023.03              # Example Anaconda module
module load cuda/12.1                      # Example CUDA module

# Activate your Conda environment
source activate topological_ml   

PROJECT_ROOT="/home/as0714/hardware_efficient_ml"
cd "$PROJECT_ROOT" || { echo "ERROR: Failed to change directory to $PROJECT_ROOT"; exit 1; }

# Add your src directory to PYTHONPATH so Python can find your modules
# Test scripts are in src/experiments, so src/ should be in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/src

mkdir -p logs
mkdir -p kernel_cost_models # Needed for test_kcm and test_router

echo "--- IMPORTANT ---"
echo "Before running this script, ensure you have run:"
echo "  python src/experiments/expert_kernel_profiler.py --d_model 4096 --profile_base_dir profiling_data_for_tests --skip_ncu"
echo "  python src/experiments/parse_profiler_output.py --profile_base_dir profiling_data_for_tests --output_json kernel_cost_model_d4096.json"
echo "These steps generate the 'kernel_cost_model_d4096.json' required by test_kcm.py and test_router.py."
echo "-------------------"
echo ""


echo "--- Running test_monitor.py ---"
python src/experiments/test_monitor.py
echo ""

echo "--- Running test_kcm.py ---"
python src/experiments/test_kcm.py
echo ""

echo "--- Running test_experts.py ---"
python src/experiments/test_experts.py
echo ""

echo "--- Running test_router.py ---"
python src/experiments/test_router.py
echo ""

echo "All unit/component tests finished for job ID: ${SLURM_JOB_ID}"