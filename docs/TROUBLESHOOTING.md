# Troubleshooting

Common failures and what to do about them.

---

## NeMo will not install

```
ERROR: Could not build wheels for ...
```

**Check your Python version first.** This is the usual cause.

```bash
python3 --version        # must be 3.10 or newer
```

If it says 3.9, you loaded the wrong module. Use:

```bash
ml GCCcore/12.3.0 Python FFmpeg CUDA
```

If that module does not exist on your cluster, find one that does:

```bash
module spider Python
```

Then update `module_load_command` in `src/config.py` and in
`config/run.slurm`.

**Other things to try:**

```bash
ml WebProxy                      # the installer needs internet
pip install --upgrade pip
pip install Cython packaging     # NeMo wants these present first
pip install -r requirements.txt
```

If it still fails, ask <help@hprc.tamu.edu> — they can tell you which toolchain
combinations are known to work.

---

## `ModuleNotFoundError: No module named 'src'`

You are running from the wrong folder. Run from the repository root:

```bash
cd /scratch/user/$USER/asr/repo
python scripts/transcribe.py status --queue ../data/queue
```

---

## The job starts, then dies immediately

Read the log:

```bash
tail -50 transcribe_Out.*
```

**"CUDA out of memory"** — two models on one GPU. Either request two
(`--gres=gpu:a100:2`, the default) or run them one after the other:

```python
# src/config.py
turns_device = "cuda:0"
parallel_models = False
```

**"No such file or directory: .../venv/bin/activate"** — the virtual environment
is somewhere else than the job expects. Check `venv_name` and `working_dir` in
`src/config.py`.

**"command not found: python"** — the module load line in `config/run.slurm`
did not work on the compute node. Check the module name.

---

## Nothing happens — the queue stays empty

```bash
python scripts/transcribe.py status --queue data/queue
```

```json
{ "pending": 0, "claimed": 0, "done": 0, "failed": 0 }
```

Nothing was found to do. Possible reasons:

- **Everything is already transcribed.** `fill` skips interviews that already
  have a JSON file. Look in `data/oral_output/json/`.
- **The spreadsheet has no rows to process.** Check the `as` column in your
  tracking sheet.
- **Wrong source.** `--source local` needs `--input` pointing at a folder that
  actually contains audio.

---

## Everything fails with "SMB_PASSWORD is not set"

The job needs the password in its environment. Export it before submitting:

```bash
export SMB_PASSWORD=your_netid_password
python scripts/run_pipeline.py
```

If you submit with `sbatch` directly, pass it through:

```bash
sbatch --export=ALL,SMB_PASSWORD="$SMB_PASSWORD" config/run.slurm
```

---

## Everything fails with an SMB connection error

The compute nodes cannot reach the file share. The web proxy only carries web
traffic, and the file share does not use it.

**Work around it:** copy the audio to scratch once from a login node, then run
in local mode.

```bash
# on a login node, where the share is reachable
python scripts/transcribe.py fill \
    --queue  data/queue \
    --output data/oral_output \
    --source local \
    --input  data/oral_input
```

Nothing else changes.

---

## Some interviews failed

See why:

```bash
python scripts/transcribe.py status --queue data/queue --failures
```

```
iv_014: RuntimeError: no speech recognised
iv_022: FileNotFoundError: source audio missing: /scratch/.../iv_022.mp3
```

| Message | Meaning |
|---|---|
| `no speech recognised` | The file is silent, corrupt, or not really audio |
| `source audio missing` | The path in the queue no longer exists. Not retried — it will never succeed |
| `CUDA out of memory` | Usually a very long file. Try `--sequential-models` |
| `yt-dlp failed` | The streaming URL is dead or needs a login |

Retry them after fixing the cause:

```bash
python scripts/transcribe.py requeue --queue data/queue
sbatch config/run.slurm
```

---

## The same interview keeps being redone

`lease_seconds` is shorter than the file takes. Another worker assumes the first
one died and starts over.

Raise it in `src/config.py`:

```python
lease_seconds = 10800     # 3 hours
```

It must be longer than your slowest single recording.

---

## The job ran out of time

Expected on a big collection. Submit again:

```bash
sbatch config/run.slurm
```

Finished interviews are skipped. Interviews that were in progress go back in the
queue automatically.

---

## Speaker labels look wrong

**Everything is `S01`.** The model heard only one voice. Common when one speaker
is much quieter — check the recording levels.

**More speakers than there really are.** Background noise or crosstalk being
read as another person. The model handles up to 4; beyond that it degrades.

**Speakers swapped part-way through.** Labels are assigned in order of first
speech and can drift on long recordings.

**To check whether diarization is the problem**, turn it off and compare the
words alone:

```bash
python scripts/transcribe.py work --queue data/queue --output data/oral_output --no-diarize
```

If the words are fine without it, the transcription is good and only the speaker
assignment needs attention.

---

## Transcription quality is poor

Before changing anything, look at `word_score_buckets` in the JSON. If the
scores are low across the board, it is the audio, not the model.

Things worth trying, in order:

1. **Listen to the recording.** Very quiet, very noisy, or heavily clipped audio
   is hard for any model.
2. **Try the noise reduction.** Off by default, but old tape is the case where
   it might help:
   ```bash
   python scripts/transcribe.py work --queue Q --output OUT --denoise
   ```
   Run one file both ways and compare.
3. **Check the language.** These models are English-first.

---

## Checking on a running job

```bash
squeue -u $USER                              # is it running?
tail -f transcribe_Out.*                     # live log
python scripts/transcribe.py status --queue data/queue
scancel <jobid>                              # stop it
```

Stopping a job is safe. Interviews in progress go back in the queue and are
picked up next time.

---

## Starting completely over

```bash
rm -rf data/queue          # forget all progress
rm -rf data/oral_output    # delete transcripts — they will be redone
```

To keep the transcripts but rebuild the queue, delete only `data/queue`.
Finished interviews are then skipped, because `fill` checks the output folder.

---

## Still stuck

- HPRC support: <help@hprc.tamu.edu> — for account, module and cluster problems
- Include your job ID, the module load line, and the last 50 lines of the log
