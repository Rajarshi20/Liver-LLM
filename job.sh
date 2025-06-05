#!/bin/bash
#SBATCH --job-name=test-llm
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32          # Avoid --cpus-per-gpu to prevent implicit mem-per-cpu
#SBATCH --mem-per-gpu=193000        # ONLY use this for memory
#SBATCH --time=05:00:00
#SBATCH --output=test_output.log
#SBATCH --error=test_error.log

source liverenv/bin/activate
module load toolkit/nvidia-hpc-sdk/23.9 # if it is needed

srun python /hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver\ LLM/main.py
