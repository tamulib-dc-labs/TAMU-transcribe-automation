#!/bin/bash
#SBATCH --job-name=asr-publish
#SBATCH --output=publish_Out.%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=short
#SBATCH --time=02:00:00

# Step 3 of 3. No GPU.
#
# Pushes the transcripts to GitHub and, in json mode, writes the resulting
# links back into the reviewer repo's config.json.
#
# Depends on the transcription job, so it starts once the GPUs are done.

ml GCCcore/12.3.0 Python FFmpeg CUDA
{{WEB_PROXY}}

source {{VENV_ACTIVATE_PATH}}

python {{PUBLISH_SCRIPT}}
