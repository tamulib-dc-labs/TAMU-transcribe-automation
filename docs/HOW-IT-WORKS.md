# How it works

The design, and why it is built this way. Read this before changing the code.

For diagrams, open [architecture.html](architecture.html).

---

## The shape of a run

One job does everything. `run_pipeline.py` submits it and exits.

```
run_pipeline.py   login node   →  sbatch, then exit

config/run.slurm  compute node →  1. list the work, skip what is done
                                     (no audio moves)
                                  2. transcribe, one interview at a time
                                  3. push the results, write back the links
```

The three steps are `scripts/run_job.py`, in one process. There is one job to
submit, one log to read and one thing to re-run.

**The login node only runs `sbatch`.** It is a shared machine that everyone
logs into, and running real work there slows it down for everyone — TAMU asks
you not to. `sbatch` is the one thing that *has* to happen there, because that
is the only place you can submit from. Everything else — cloning repositories,
downloading audio, transcribing, uploading — happens on a compute node.

Compute nodes reach the internet through `module load WebProxy`, which is why
they can do the network steps at all.

---

## Step 1 — listing the work

Reads the tracking spreadsheet (or the reviewer app's JSON list) and writes one
list of interviews still to do.

Each file is a **reference**, not audio:

```json
{ "id": "02_00113_a_01",
  "kind": "smb",
  "remote_path": "//cifs.library.tamu.edu/share/02_00113/02_00113_a_01.mp3" }
```

Three kinds: `local` (already on disk), `url` (downloadable), `smb` (the file
share).

**In `--from-json` mode there is one extra step first.** The list of interviews
lives in a GitHub repo, so the job clones it, reads `config-to-process.json`,
and writes the entries to `data/work_list.json`. The filling step then reads
that file. Still no audio — only the list.

Each entry's audio comes from its **`audio`** field. Note that these entries
also have a `url` field, which is the *transcript* link the reviewer app shows
— not the recording. Using it would download a JSON file and hand it to the
speech model.

Output files are named after the entry's `name`, so you can predict a
transcript's filename from the config without looking at the audio.

This step is **idempotent** — running it twice changes nothing, which is what
makes an interrupted run safe to re-submit.

### What counts as "already done"

Two checks:

1. **The transcripts repository.** The job lists what already has a JSON in
   `edge-grant-json-and-vtts` and writes the names to `data/completed.json`.
2. **The local output folder**, `data/oral_output/json/`.

The first check matters because **`/scratch` is purged periodically**. The local
folder is a cache; the GitHub repository is the record. Without check 1, a purge
would mean re-transcribing and re-uploading the entire collection.

If the repository cannot be reached the run continues on check 2 alone — some
work may be redone, which is better than refusing to start.

**The reviewer repo's `config.json` is not one of the checks.** That file is
*written* at the end of a run so the reviewer app has links to the new
transcripts; it is not a record of what has been transcribed. Reading it to
decide what to skip made the two jobs of that file contradict each other, and
made `config.json` unusable as both input and output.

---

## Step 2 — transcribing

`scripts/transcribe.py run`, on a GPU node. For each interview in turn:

1. **Take** the next interview on the list.
2. **Download** just that one file into node-local `$TMPDIR`.
3. **Transcribe** it — Parakeet and Sortformer, on separate GPUs, at the same time.
4. **Fuse** their outputs by time.
5. **Write** the JSON and VTT.
6. **Delete** its copy of the audio and mark the interview done.

Downloading per interview rather than all up front has two consequences worth
knowing: **disk use depends on the number of workers, not the size of the
collection**, and a failure part-way through costs one interview instead of the
whole download phase.

---

## The two models

| | Model | Produces | Time per 90 min |
|---|---|---|---|
| Words | `parakeet-tdt-0.6b-v3` | text, word timings, confidence | ~1.6 s |
| Speakers | `diar_streaming_sortformer_4spk-v2.1` | speaker turns | ~11 s |

Both are NVIDIA NeMo models, so they install together and run in one process.

They are pinned to separate GPUs and run at the same time. A Grace GPU node has
exactly two A100s, which is what the defaults assume.

> Both models are fast. Almost all the time in a run goes to downloading and
> decoding audio, not to the GPUs.

---

## Why these two models specifically

This is the part worth understanding before substituting anything.

**They measure time the same way.** Both read the audio at 16 kHz through the
same kind of front end. Parakeet's word timings and Sortformer's speaker turns
therefore sit on the same clock, and the error when matching them is bounded by
Sortformer's 80-millisecond resolution.

An alternative we tried and rejected was a large language model that transcribes
*and* diarizes. It writes its timestamps as text — it literally types the
characters `[0.48]` the way it types any other word. Those times are
**predicted**, not measured, and nothing ties them to the recording. Matching
predicted times against measured ones makes speakers drift out of sync with the
words, and there is no limit to how far.

If you replace either model, check that the replacement *measures* time.

---

## Putting words and speakers together

Parakeet gives words with times. Sortformer gives speaker turns with times.
Neither knows about the other.

Each word is given the speaker of the turn it **overlaps most in time**. A word
that falls in a gap between turns takes the nearest one, if it is within a
second; further away than that, it gets no speaker rather than a guess.

Then consecutive words with the same speaker are grouped into segments, and
those segments are broken into subtitle lines.

Code: `src/asr/fusion.py`, `src/asr/lines.py`.

---

## When something fails

Failures are handled by kind, not uniformly:

| Failure | What happens |
|---|---|
| Speaker detection fails | The run continues **without speaker labels**. A transcript with no speakers is still useful; no transcript is not |
| Audio file missing | Failed immediately, not retried — it will never succeed, and retrying wastes GPU time |
| Anything else | Logged with the reason and counted; the run carries on. Re-run to retry it |
| Worker killed | Its claim expires and another worker picks the interview up |

---

## Where the code lives

```
config/run.slurm         the one job, submitted by run_pipeline.py
scripts/run_job.py       what that job runs: list, transcribe, upload
scripts/transcribe.py    run / status
scripts/run_pipeline.py  submit, wait, upload

src/asr/sources.py       where audio comes from; fetching one file
src/asr/parakeet.py      words, timings, confidence
src/asr/sortformer.py    speaker turns
src/asr/fusion.py        matching words to speakers
src/asr/lines.py         words → subtitle lines
src/asr/output.py        writing the JSON and VTT
src/asr/preprocess.py    decode and optional noise reduction

src/config.py            every setting
src/pipeline.py          submit, monitor, upload
src/git/, src/utils/     unchanged from v2.0
```

---

## One rule if you edit `src/asr/`

**Never import `torch` or `nemo` at the top of a file.** Put the import inside
the function that needs it.

That keeps the package importable on a login node with no GPU software
installed, which is what lets `status` run there.

Breaking it does not fail on a machine where everything is installed. It fails
later, on the cluster — so if you add an import, put it inside the function
that needs it, not at the top of the module.

---

## What changed from v2.0

| v2.0 | v3.0 |
|---|---|
| Whisper `large-v3` via WhisperX | Parakeet-TDT |
| Silero voice detection, plus a workaround for short segments | Not needed — removed |
| wav2vec2 forced alignment for word timings | Not needed — Parakeet times its own words |
| No speaker labels | Sortformer |
| Download the whole collection, then transcribe | Download one interview at a time, inside the job |
| Static split of files across GPUs | One list, worked through in order |
| `whisperx`, `ctranslate2`, `nltk`, `langchain` | One package: `nemo_toolkit[asr]` |

Removed along the way: a `torch.load` monkeypatch and a cuDNN library-path fix,
both of which existed only to work around `ctranslate2`.

**The output format did not change.** v2.0 and v3.0 files are interchangeable
for the reviewer app.
