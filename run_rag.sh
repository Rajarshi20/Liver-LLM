#!/bin/bash
#SBATCH --job-name=llm-rag
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=2      
#SBATCH --mem=128G       # ONLY use this for memory
#SBATCH --gpus=A40:2
#SBATCH --time=0-12:00:00
#SBATCH --output=/mnt/beegfs/home/st191428/Liver-LLM-logs/%u_rag_output_%j.out
#SBATCH --error=/mnt/beegfs/home/st191428/Liver-LLM-logs/%u_rag_error_%j.err
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

# eecho "[INFO] Starting run_rag.py with streamlit"
# PORT=8501
# streamlit run run_rag.py --server.port $PORT --server.address=0.0.0.0 > streamlit.log 2>&1 &

# sleep 20

# # === Launch Ngrok and capture tunnel URL ===
# echo "[INFO] Launching ngrok on port $PORT..."
# ngrok http $PORT --log=stdout > ngrok.log 2>&1 &

# echo "[INFO] Waiting for ngrok tunnel to be established..."
# sleep 15

# # Try to extract URL from log
# NGROK_URL=$(grep -oE "https://[a-z0-9]+\.ngrok-free\.app" ngrok.log | head -n 1)
# echo "$NGROK_URL" > tunnel_url.txt

# if [ -n "$NGROK_URL" ]; then
#     echo "[INFO] Ngrok tunnel created: $NGROK_URL"
# else
#     echo "[ERROR] Ngrok tunnel URL not found after 30 seconds."
#     echo "Check ngrok.log for details."
# fi

python run_rag.py --pdf /mnt/beegfs/home/st191428/A_multicenter_prospective_study.pdf --question "What is HCC?"