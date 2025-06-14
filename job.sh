#!/bin/bash
#SBATCH --job-name=test-llm
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32          # Avoid --cpus-per-gpu to prevent implicit mem-per-cpu
#SBATCH --mem-per-gpu=124000        # ONLY use this for memory
#SBATCH --time=10:00:00
#SBATCH --output=test_output.log
#SBATCH --error=test_error.log

# Activate your virtual environment
source "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/liverenv/bin/activate"

# Set HF cache to scratch
export HF_HOME=/hkfs/work/workspace/scratch/st_st191428-LiverLLM/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Load Hugging Face token from your custom cache path
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)

# Confirm token presence (optional)
if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
  echo "? Hugging Face token not found or empty!"
  exit 1
else
  echo "? Hugging Face token is set."
fi


srun python "/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/main.py"
