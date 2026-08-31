#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=asr-transcribe
#SBATCH --output=transcribe_Out.%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@900

# The whole pipeline, start to finish, on a compute node.
#
#   1. list the work        read the work list, skip what is already
#                           transcribed in the json-and-vtts repo
#   2. transcribe           Parakeet-TDT for words and timings, then
#                           Sortformer for speaker turns, then fuse them by
#                           time. One GPU, one model at a time - see
#                           parallel_models in src/config.py for why
#   3. upload               push the json and vtt files, then write their
#                           links into the reviewer app's config.json
#
# Nothing runs on a login node - `sbatch` is the only thing that happens there.
# Compute nodes reach GitHub and the media server through WebProxy.
#
# 4-hour wall clock. Interrupted? Submit again: finished interviews are
# skipped and any that were in flight are put back.

{{MODULE_LOAD}}
{{WEB_PROXY}}

source {{VENV_ACTIVATE_PATH}}

export HF_HOME={{HF_CACHE}}
export TORCH_HOME={{HF_CACHE}}
mkdir -p "$HF_HOME"

# Which interpreter actually ran, for the log. `source venv/bin/activate` uses
# whatever Python the venv was built against, NOT whatever `ml` just loaded -
# so this is the version that matters, and it is worth seeing it every run.
echo "python: $(which python)"
python -c "import sys; print('version:', sys.version.split()[0])"

python {{JOB_SCRIPT}} {{JOB_ARGS}}
