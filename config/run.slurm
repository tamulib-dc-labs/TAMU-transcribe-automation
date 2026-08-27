#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=asr-transcribe
#SBATCH --output=transcribe_Out.%j
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@900

# The whole pipeline runs here. Nothing is staged in advance:
#
#   1. enumerate the collection into the shared queue  (no bytes moved)
#   2. each task fetches, transcribes and discards one interview at a time
#
# Parakeet-TDT produces words, timings and per-word confidence; Sortformer
# produces speaker turns. Both are NeMo and both derive time from the same
# 16 kHz waveform through the same FastConformer frontend, so the fusion joins
# two MEASURED time bases -- alignment error is bounded by Sortformer's 80 ms
# frame grid.
#
# A Grace GPU node has exactly 2x A100, so one model is pinned to each and they
# genuinely overlap.
#
# The gpu partition allows 4 days. Interrupted anyway? Just sbatch again:
# finished interviews are skipped, in-flight ones are reaped and redone.

# NeMo needs Python >= 3.10. GCCcore/10.3.0 Python is 3.9 and will NOT work.
ml GCCcore/12.3.0 Python FFmpeg CUDA

# Compute nodes have no direct internet. WebProxy sets http_proxy/https_proxy,
# which is what lets this job download its own audio and its own model weights
# instead of needing anything staged beforehand.
{{WEB_PROXY}}

# PATH WILL BE INJECTED BY PIPELINE
source {{VENV_ACTIVATE_PATH}}

# Shared model cache on scratch. NeMo and Hugging Face download on first use
# and reuse it after, so the first task to need a model pays for it and the
# rest hit the cache.
export HF_HOME={{HF_CACHE}}
export TORCH_HOME={{HF_CACHE}}
mkdir -p "$HF_HOME"

# Credentials for fetching audio. Passed through the environment so they never
# get written into the queue.
export SMB_USERNAME={{SMB_USERNAME}}

# ---------------------------------------------------------------------------
# 1. Enumerate the collection into the queue. Moves no audio -- it records
#    references. Idempotent, so every array task runs it and whichever gets
#    there first populates the queue for the rest.
# ---------------------------------------------------------------------------
python {{TRANSCRIBE_SCRIPT}} fill \
    --queue  "{{QUEUE_PATH}}" \
    --output "{{ORAL_OUTPUT_PATH}}" \
    --source {{SOURCE}} {{FILL_ARGS}}

# ---------------------------------------------------------------------------
# 2. Work the queue. Each task fetches only the interview it just claimed into
#    node-local $TMPDIR, transcribes it, writes the outputs and deletes the
#    copy -- so disk use is bounded by the number of workers, not by the size
#    of the collection.
# ---------------------------------------------------------------------------
python {{TRANSCRIBE_SCRIPT}} work \
    --queue        "{{QUEUE_PATH}}" \
    --output       "{{ORAL_OUTPUT_PATH}}" \
    --scratch      "$TMPDIR/audio" \
    --parakeet-model   "{{ASR_MODEL}}" \
    --sortformer-model "{{DIARIZATION_MODEL}}" \
    --words-device cuda:0 \
    --turns-device cuda:1 \
    --deadline-minutes {{DEADLINE_MINUTES}} \
    --lease {{LEASE_SECONDS}} \
    {{EXTRA_ARGS}}
