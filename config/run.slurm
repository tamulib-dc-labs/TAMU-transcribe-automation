#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=asr-transcribe
#SBATCH --output=transcribe_Out.%j
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@900

# Step 2 of 3. The GPU work.
#
# Four tasks take interviews from the queue, one at a time. Each fetches only
# the recording it just claimed, transcribes it, writes the outputs, and
# deletes its copy - so disk use is bounded by the number of workers, not by
# the size of the collection.
#
# Parakeet-TDT produces the words, timings and per-word confidence; Sortformer
# produces the speaker turns. Both are NeMo and both take their time from the
# same 16 kHz waveform through the same FastConformer frontend, so the fusion
# joins two MEASURED time bases. A Grace GPU node has 2x A100, so one model is
# pinned to each and they overlap.
#
# 4-hour wall clock. Interrupted? Just sbatch again: finished interviews are
# skipped, in-flight ones are reaped and redone.

ml GCCcore/12.3.0 Python FFmpeg CUDA
{{WEB_PROXY}}

source {{VENV_ACTIVATE_PATH}}

export HF_HOME={{HF_CACHE}}
export TORCH_HOME={{HF_CACHE}}
mkdir -p "$HF_HOME"

python {{TRANSCRIBE_SCRIPT}} work \
    --queue        "{{QUEUE_PATH}}" \
    --output       "{{ORAL_OUTPUT_PATH}}" \
    --scratch      "$TMPDIR/audio" \
    --parakeet-model   "{{ASR_MODEL}}" \
    --sortformer-model "{{DIARIZATION_MODEL}}" \
    --words-device {{WORDS_DEVICE}} \
    --turns-device {{TURNS_DEVICE}} \
    --deadline-minutes {{DEADLINE_MINUTES}} \
    --lease {{LEASE_SECONDS}} \
    --max-attempts {{MAX_ATTEMPTS}} \
    --max-line-width {{MAX_LINE_WIDTH}} \
    --max-line-count {{MAX_LINE_COUNT}} \
    {{EXTRA_ARGS}}
