# Configuration

Every setting lives in `src/config.py`. This page says what each one does and
when you would change it.

---

## Must change — your own details

| Setting | Example | What it is |
|---|---|---|
| `smb_username` | `"jdoe"` | Your NetID, for the file share |
| `git_owner` | `"tamulib-dc-labs"` | GitHub org or user that owns the transcripts repo |
| `git_repo_name` | `"edge-grant-json-and-vtts"` | Repo transcripts are pushed to |
| `git_username` | `"JaneDoe"` | Your GitHub username |
| `sheet_url` | `"https://docs.google.com/..."` | Tracking spreadsheet listing which folders to process |
| `cache_dir` | `"/scratch/user/jdoe/asr/cache"` | Where model weights are stored. **Use scratch, not home** |

Passwords and tokens are **not** in this file. They come from the environment:

```bash
export GIT_TOKEN=...          # GitHub personal access token
export SMB_PASSWORD=...       # your NetID password, SMB mode only
```

---

## The models

| Setting | Default | Notes |
|---|---|---|
| `asr_model` | `nvidia/parakeet-tdt-0.6b-v3` | Produces the words, their timings, and confidence |
| `diarization_model` | `nvidia/diar_streaming_sortformer_4spk-v2.1` | Produces speaker turns. **Handles up to 4 speakers** — accuracy drops beyond that |
| `diarize` | `True` | Set `False` to skip speaker labels. About 7× faster, but the JSON has no `speaker` field |

---

## GPUs

| Setting | Default | Notes |
|---|---|---|
| `words_device` | `"cuda:0"` | Which GPU runs Parakeet |
| `turns_device` | `"cuda:1"` | Which GPU runs Sortformer |
| `parallel_models` | `True` | Run both at the same time |

A Grace GPU node has exactly two A100s, so the defaults put one model on each
and they run simultaneously.

**If you only request one GPU**, set both devices to `cuda:0` *and* set
`parallel_models = False`. Two models on one GPU do not go faster in parallel —
they just compete for it, and both sit in memory at once. The code warns you if
you get this combination wrong.

---

## Audio preparation

| Setting | Default | Notes |
|---|---|---|
| `denoise` | `False` | Runs a noise-reduction pass before transcription |

**Why it is off.** It was tuned for Whisper, in v2.0. Both current models are
already trained on noisy audio and normalise their own input, and the
noise-reduction can introduce artefacts that make an end-to-end model *worse*.

Turn it on if a side-by-side test on your own recordings shows it helps. Old
tape hiss is the case where it might.

---

## The queue

| Setting | Default | Notes |
|---|---|---|
| `lease_seconds` | `5400` (90 min) | How long a worker "holds" an interview before others assume it died |
| `max_attempts` | `3` | Tries before giving up on a file |
| `deadline_minutes` | `2820` (47 h) | Workers stop taking new work this far into the job |
| `max_files` | `0` | Cap interviews per run. `0` = no cap. Useful for a small test |

**`lease_seconds` must be longer than your slowest interview.** If a file takes
2 hours and the lease is 90 minutes, another worker will assume the first one
died and start the same file again. Raise it if your recordings are long.

**`deadline_minutes` must be below the job's time limit.** The default 47 hours
sits under the 48-hour limit in `config/run.slurm`, leaving time to finish the
file in hand and exit cleanly. If you shorten the job's `--time`, shorten this
too.

---

## Network

| Setting | Default | Notes |
|---|---|---|
| `use_web_proxy` | `True` | Loads the `WebProxy` module in the job |

Grace compute nodes have no direct internet. `WebProxy` gives them a route out,
which is how the job downloads its own audio and model weights.

**Leave this on.** With it off, the job cannot download anything, and it will
only work if all the audio is already on scratch and the models are already
cached.

It carries **web traffic only** (HTTP and HTTPS). The SMB file share uses a
different kind of connection and does not go through it — see
[SETUP.md step 8](SETUP.md#step-8--check-the-file-share-is-reachable-smb-mode-only).

---

## Subtitles

| Setting | Default | Notes |
|---|---|---|
| `max_line_width` | `42` | Characters per subtitle line |
| `max_line_count` | `2` | Lines shown at once |
| `language` | `"en"` | Written into the JSON. The models detect language themselves |

---

## Where files go

These are worked out from `working_dir` — you rarely change them.

| Property | Path |
|---|---|
| `oral_input_path` | `data/oral_input` — audio, in local mode |
| `oral_output_path` | `data/oral_output` — the JSON and VTT files |
| `queue_path` | `data/queue` — the work queue |
| `hf_cache` | `<cache_dir>/huggingface` — model weights |

---

## The Slurm job

Job size is in `config/run.slurm`, not in `config.py`:

```bash
#SBATCH --array=0-3            # 4 workers at once
#SBATCH --gres=gpu:a100:2      # 2 GPUs each
#SBATCH --time=48:00:00        # Grace allows up to 4 days
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
```

**More workers = faster**, up to the number of interviews you have. `--array=0-7`
gives 8. Each takes a whole GPU node, so ask for what you will use.

If you change `--time`, change `deadline_minutes` to match.
