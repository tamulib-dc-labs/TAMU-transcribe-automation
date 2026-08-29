#!/bin/bash
#SBATCH --job-name=asr-prepare
#SBATCH --output=prepare_Out.%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --time=02:00:00

# Step 1 of 3. No GPU.
#
# Reads the work list, lists what is already transcribed, and fills the queue.
# All network and filesystem work - compute nodes reach GitHub through
# WebProxy, and credentials come from config/local_settings.py on the shared
# filesystem, so nothing is passed through the job environment.
#
# One task on purpose: cloning the reviewer repo from several tasks at once
# would race on the same directory.

ml GCCcore/12.3.0 Python FFmpeg CUDA
{{WEB_PROXY}}

source {{VENV_ACTIVATE_PATH}}

export HF_HOME={{HF_CACHE}}
export TORCH_HOME={{HF_CACHE}}

python {{PREPARE_SCRIPT}}
