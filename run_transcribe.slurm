#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=asr-transcribe
#SBATCH --output=transcribe_Out.%j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@900

# The whole pipeline, start to finish, on a compute node.
#
#   1. list the work        read the work list, skip what is already
#                           transcribed in the json-and-vtts repo
#   2. transcribe           Parakeet-TDT for words and timings on one A100,
#                           Sortformer for speaker turns on the other, at the
#                           same time, then fuse them by time
#   3. upload               push the json and vtt files, then write their
#                           links into the reviewer app's config.json
#
# Nothing runs on a login node - `sbatch` is the only thing that happens there.
# Compute nodes reach GitHub and the media server through WebProxy.
#
# 4-hour wall clock. Interrupted? Submit again: finished interviews are
# skipped and any that were in flight are put back.

ml GCCcore/12.3.0 Python FFmpeg CUDA
ml WebProxy

source C:\lib_work\tamu-transcribe\venv/bin/activate

export HF_HOME=/scratch/group/tamu_libr_dc/cache\huggingface
export TORCH_HOME=/scratch/group/tamu_libr_dc/cache\huggingface
mkdir -p "$HF_HOME"

python C:\lib_work\tamu-transcribe\scripts\run_job.py --skip-upload --no-diarize --max-files 3
