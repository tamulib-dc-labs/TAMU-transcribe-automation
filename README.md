# Oral History Transcription — v3.0

Transcribes oral history recordings on TAMU HPRC **Grace**, with speaker labels,
and uploads the results to a GitHub repository for review.

Everything runs on Grace. Nothing is downloaded to your own computer.

---

## What it does

You point it at a collection of interview recordings. For each one it gives you:

- a **transcript** with a timestamp on every single word
- a **confidence score** on every word, so a reviewer can see what to check
- **speaker labels** — who was talking, and when
- a **subtitle file** (`.vtt`) that plays alongside the audio

## Which models

| Job | Model |
|---|---|
| Words, timings, confidence | `nvidia/parakeet-tdt-0.6b-v3` |
| Speaker turns | `nvidia/diar_streaming_sortformer_4spk-v2.1` |

Both are NVIDIA NeMo models. They install together and run in one process.

> **Why these two?** They measure time the same way — both read the audio at
> 16 kHz through the same front end. So word times and speaker times line up,
> and speakers get attached to the right words.
> See [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

## Versions

| Version | What it uses |
|---|---|
| **v3.0** (this) | Parakeet + Sortformer. Speaker labels. One Slurm job does everything. |
| **v2.0** (`git checkout v2.0`) | WhisperX. No speaker labels. Downloads everything up front. |

Both write the **same JSON and VTT format**, so the reviewer app works with
either.

---

## Getting started

New to this? Follow **[docs/SETUP.md](docs/SETUP.md)** — it goes from getting an
HPRC account to your first transcript.

Already set up:

```bash
cp config/local_settings.example.py config/local_settings.py   # once, then edit

python scripts/run_pipeline.py                       # from the tracking spreadsheet
python scripts/run_pipeline.py --from-json           # from the reviewer app's list
python scripts/run_pipeline.py --source local --max-files 1 --skip-upload
```

Your NetID, GitHub token and paths go in `config/local_settings.py`. It is
gitignored, so nothing private is committed.

That submits the job to Grace, waits for it to finish, and uploads the results.

---

## Documentation

| Guide | Read it when |
|---|---|
| **[SETUP.md](docs/SETUP.md)** | Setting this up on your own HPRC account |
| **[CONFIGURATION.md](docs/CONFIGURATION.md)** | Changing a setting and wanting to know what it does |
| **[HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)** | Understanding the design, or changing the code |
| **[OUTPUT.md](docs/OUTPUT.md)** | Working with the JSON or VTT files |
| **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Something went wrong |
| **[architecture.html](docs/architecture.html)** | Wanting the diagrams |

---

## How a run works

One `sbatch` does the whole thing:

1. **List the work.** Reads the tracking spreadsheet (or the reviewer app's
   list) and writes one entry per interview into a queue folder on `/scratch`.
   No audio is downloaded yet.
2. **Do the work.** Four jobs run at once. Each takes the next interview from
   the queue, downloads just that file, transcribes it, saves the results, and
   deletes its copy of the audio.
3. **Upload.** After the job finishes, the transcripts are pushed to the
   reviewer repository.

If the job is interrupted, run `sbatch config/run.slurm` again. Finished
interviews are skipped; unfinished ones are picked back up.

---

## What you get

```
data/oral_output/json/02_00113_a_01.json     every word, with time + confidence
data/oral_output/vtts/02_00113_a_01.vtt      subtitles
```

```
00:00:04.180 --> 00:00:09.310
<v S02>Well, we were still out on the Brazos
place then. My father worked the land and
```

Full description in [docs/OUTPUT.md](docs/OUTPUT.md).

---

## Requirements

- A TAMU HPRC account with access to **Grace** ([apply here](https://hprc.tamu.edu/apply/))
- An allocation with GPU hours
- A GitHub personal access token, if you want results uploaded

---

## Tests

```bash
pip install pytest
pytest -q
```

233 tests. They need no GPU, no model weights and no internet — the models are
stubbed, so they run on a login node or your laptop.

---

## Status

**This has not yet been run on Grace.** Two things to confirm on your first run,
both covered in [SETUP.md](docs/SETUP.md):

1. That `nemo_toolkit[asr]` installs on Grace.
2. That a compute node can reach the SMB file share — only matters if you read
   audio from there.
