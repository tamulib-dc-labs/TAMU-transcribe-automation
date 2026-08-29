# Configuration

Every setting lives in `src/config.py`. This page says what each one does and
when you would change it.

## Where to put your own values

**Do not edit `src/config.py` for anything private.** This repository is
public — a token committed there is pushed to GitHub and revoked within
minutes.

Put your settings in `config/local_settings.py` instead. It is gitignored, so
it never leaves your machine, and anything in it overrides `src/config.py`.

```bash
cp config/local_settings.example.py config/local_settings.py
# then edit it
```

```python
# config/local_settings.py
smb_username = "your_netid"
smb_password = "your_netid_password"

git_owner    = "tamulib-dc-labs"
git_username = "YourGitHubUsername"
git_token    = "ghp_xxxxxxxxxxxx"

cache_dir = "/scratch/user/your_netid/asr/cache"
```

Leave out anything you do not need — the defaults still apply. On startup the
pipeline prints which settings it picked up (names only, never values).

Credentials can still come from the environment instead, if you prefer:
`GIT_TOKEN` and `SMB_PASSWORD`. `local_settings.py` wins if both are set.

---

## Settings — your own details

Set these in `config/local_settings.py`.

| Setting | Example | What it is |
|---|---|---|
| `smb_username` | `"jdoe"` | Your NetID, for the file share |
| `git_owner` | `"tamulib-dc-labs"` | GitHub org or user that owns the transcripts repo |
| `git_repo_name` | `"edge-grant-json-and-vtts"` | Repo transcripts are pushed to |
| `git_username` | `"JaneDoe"` | Your GitHub username |
| `sheet_url` | `"https://docs.google.com/..."` | Tracking spreadsheet listing which folders to process |
| `cache_dir` | `"/scratch/user/jdoe/asr/cache"` | Where model weights are stored. **Use scratch, not home** |

| `git_token` | `"ghp_..."` | GitHub token with `repo` scope |
| `smb_password` | `"..."` | Your NetID password |

Or export `GIT_TOKEN` and `SMB_PASSWORD` instead of writing them down.

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
| `deadline_minutes` | `225` (3 h 45 m) | Workers stop taking new work this far into the job |
| `max_files` | `0` | Cap interviews per run. `0` = no cap. Useful for a small test |

## Not redoing finished work

| Setting | Default | Notes |
|---|---|---|
| `check_transcripts_repo` | `True` | Before queueing, check the transcripts repo for interviews that already have a JSON |

`/scratch` is purged periodically, so `data/oral_output` is a cache, not a
record. The GitHub transcripts repository is the record. Leave this on — with it
off, a purge means transcribing and uploading the whole collection again.

---

## Where the audio comes from

| Setting | Default | Notes |
|---|---|---|
| `source` | `"auto"` | `smb`, `json`, or `local`. `auto` follows `from_json` |
| `input_dir` | `""` | Folder to read with `source = "local"`. Defaults to `data/oral_input` |

Or from the command line:

```bash
python scripts/run_pipeline.py --source local --input data/oral_input
python scripts/run_pipeline.py --source json      # same as --from-json
```

Use `local` for your first test run, and as the fallback if compute nodes
cannot reach the file share.

---

## Reading the list from the reviewer app

Set `from_json = True` (or pass `--from-json`) and the interviews come from the
reviewer repo's `config-to-process.json` instead of the tracking spreadsheet.

| Setting | Default | Notes |
|---|---|---|
| `from_json` | `False` | Take the work list from the reviewer repo |
| `config_repo_name` | `"edge-grant-reviewer"` | The repo holding the list |
| `config_json_path` | `"public/config-to-process.json"` | The list of work to read, inside that repo |
| `output_config_path` | `"public/config.json"` | Written after a run, with the transcript links |

**`config_json_path` is read; `output_config_path` is written.** They do
different jobs and neither affects the other:

- `config_json_path` is the list of interviews to transcribe. Every entry in it
  is queued.
- `output_config_path` is written at the *end* of a run, adding the JSON and
  VTT links for the transcripts that were produced, so the reviewer app can
  show them on the next visit. It is **not** consulted beforehand and does not
  decide what gets skipped — that comes from the transcripts repository (see
  `check_transcripts_repo`).

Entries are merged into the output by `name`, so re-transcribing an interview
replaces its row rather than adding a second one. Interviews with no transcript
yet are left untouched.

You *can* point both at the same file. The run then reads it, transcribes
everything listed, and overwrites it with the links — the pipeline prints a note
when you do. Two files is still clearer.

The repo is cloned by the prepare job, which reaches GitHub through WebProxy.
The entries are written to `data/work_list.json` for the filling step.

Each entry needs a **`name`** and an **`audio`** URL. Transcripts are named
after `name`, so `02_00113` produces `02_00113.json` and `02_00113.vtt`.

**`lease_seconds` must be longer than your slowest interview.** If a file takes
2 hours and the lease is 90 minutes, another worker will assume the first one
died and start the same file again. Raise it if your recordings are long.

**`deadline_minutes` must be below the job's time limit.** The default 3 h 45 m
sits under the 4-hour limit in `config/run.slurm`, leaving time to finish the
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

Job size is in the Slurm files, not in `config.py`. There are three:
`config/prepare.slurm` and `config/publish.slurm` are small CPU jobs;
`config/run.slurm` is the GPU one worth tuning:

```bash
#SBATCH --array=0-3            # 4 workers at once
#SBATCH --gres=gpu:a100:2      # 2 GPUs each
#SBATCH --time=04:00:00        # Grace allows up to 4 days on the gpu partition
#SBATCH --cpus-per-task=24
#SBATCH --mem=360G
```

**More workers = faster**, up to the number of interviews you have. `--array=0-7`
gives 8. Each takes a whole GPU node, so ask for what you will use.

If you change `--time`, change `deadline_minutes` to match.
