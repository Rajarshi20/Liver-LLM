#!/bin/bash
#SBATCH --job-name=test-cl
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=2      
#SBATCH --mem=128G       # ONLY use this for memory
#SBATCH --gpus=A40:2
#SBATCH --time=0-03:00:00
#SBATCH --output=/mnt/beegfs/home/st191428/Liver-LLM-logs/%u_cl_output_%j.out
#SBATCH --error=/mnt/beegfs/home/st191428/Liver-LLM-logs/%u_cl_error_%j.err
#SBATCH --mail-user=st191428@stud.uni-stuttgart.de
#SBATCH --mail-type=END,FAIL
echo "[INFO] Job started"

# Activate env
module purge
module load Miniconda3
source ${EBROOTMINICONDA3}/bin/activate
conda activate venv || { echo "[ERROR] Conda activate failed"; exit 1; }

# HF setup
echo "[INFO] Setting up Hugging Face cache"
export HF_HOME=/mnt/beegfs/home/$USER/$SLURM_JOB_ID/hf_cache
mkdir -p $HF_HOME
cp /mnt/beegfs/home/st191428/hf_cache/token $HF_HOME/token || { echo "[ERROR] HF token copy failed"; exit 1; }
export HUGGINGFACE_HUB_TOKEN=$(cat $HF_HOME/token)

if [ -z "$HUGGINGFACE_HUB_TOKEN" ]; then
  echo "[ERROR] HF token is empty"
  exit 1
else
  echo "[INFO] HF token loaded"
fi

# Move code to scratch
echo "[INFO] Copying Liver-LLM to scratch"
SCRATCH_JOB_DIR=/mnt/beegfs/home/$USER/$SLURM_JOB_ID/job_run
mkdir -p $SCRATCH_JOB_DIR
cp -r /mnt/beegfs/home/st191428/Liver-LLM $SCRATCH_JOB_DIR || { echo "[ERROR] Code copy failed"; exit 1; }

cd $SCRATCH_JOB_DIR/Liver-LLM || { echo "[ERROR] cd to Liver-LLM failed"; exit 1; }


python continual_learning.py --pdf /mnt/beegfs/home/st191428/Treatment_of_Liver_Cancer.pdf