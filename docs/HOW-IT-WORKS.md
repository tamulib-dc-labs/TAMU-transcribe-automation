# How it works

The design, and why it is built this way. Read this before changing the code.

For diagrams, open [architecture.html](architecture.html).

---

## The shape of a run

One `sbatch` does everything.

```
1. list the work   →  queue/pending/    (no audio moves)
2. do the work     →  four jobs, each taking one interview at a time
3. upload          →  after the job, from a login node
```

---

## Step 1 — listing the work

Reads the tracking spreadsheet (or the reviewer app's JSON list) and writes one
small file per interview into `data/queue/pending/`.

Each file is a **reference**, not audio:

```json
{ "id": "02_00113_a_01",
  "kind": "smb",
  "remote_path": "//cifs.library.tamu.edu/share/02_00113/02_00113_a_01.mp3" }
```

Three kinds: `local` (already on disk), `url` (downloadable), `smb` (the file
share).

**In `--from-json` mode there is one extra step first.** The list of interviews
lives in a private GitHub repo, so `run_pipeline.py` clones it on the login
node, drops anything already transcribed, and writes what is left to
`data/work_list.json`. The job reads that file. Still no audio — only the list.

Each entry's audio comes from its **`audio`** field. Note that these entries
also have a `url` field, which is the *transcript* link the reviewer app shows
— not the recording. Using it would download a JSON file and hand it to the
speech model.

Output files are named after the entry's `name`, so you can predict a
transcript's filename from the config without looking at the audio.

This step is **idempotent** — running it twice changes nothing. That is why
every one of the four jobs runs it at startup: whichever gets there first fills
the queue, and the others find it already done. No coordination needed.

---

## Step 2 — doing the work

Each job loops:

1. **Claim** the next interview from the queue.
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

## The queue

Four folders on `/scratch`:

```
queue/pending/     waiting
queue/claimed/     someone is working on it
queue/done/        finished
queue/failed/      gave up after 3 tries
```

That is the entire state of a run. No database, no server.

### Why a queue and not "worker 1 takes files 1–10"

Interviews range from twenty minutes to three hours. Split the list four ways
and three GPUs finish early while one grinds through the long recordings.
Taking the next file when you finish the last one keeps everyone busy.

### Why claims expire

A claim carries a deadline (`lease_seconds`). If a job is killed by the time
limit, its interviews are left *claimed* rather than lost. The next job to start
notices the expired claims and puts them back.

That is why resuming is just `sbatch` again.

### How two workers never take the same file

Claiming creates a lock file with `O_CREAT | O_EXCL` — an operation that
succeeds for exactly one caller, on every filesystem Grace uses. Only that
caller then moves the interview into `claimed/`.

This matters more than it sounds. Two earlier designs let the same interview be
claimed twice under eight workers — 3% of the time in one, 17% in the other.
That means two GPUs transcribing the same recording and both writing the same
output file. `tests/test_workqueue.py` has a test that would catch it coming
back.

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
| Anything else | Retried up to `max_attempts`, then moved to `failed/` with the reason |
| Worker killed | Its claim expires and another worker picks the interview up |

---

## Where the code lives

```
config/run.slurm         the job: list the work, then do it
scripts/transcribe.py    fill / work / status / requeue
scripts/run_pipeline.py  submit, wait, upload

src/asr/sources.py       where audio comes from; fetching one file
src/asr/parakeet.py      words, timings, confidence
src/asr/sortformer.py    speaker turns
src/asr/fusion.py        matching words to speakers
src/asr/lines.py         words → subtitle lines
src/asr/output.py        writing the JSON and VTT
src/asr/workqueue.py     claim / lease / reap
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
installed, which is what lets `fill`, `status`, `requeue` and the whole test
suite run there.

`tests/test_import_isolation.py` checks every file in the package and fails if
this is broken — which is deliberate, because breaking it does not fail on your
machine where everything is installed. It fails later, on the cluster.

---

## Tests

```bash
pytest -q          # 210 tests, no GPU, no weights, no network
```

The models are stubbed, so what is tested is everything around them: the queue
under concurrent workers, matching words to speakers, decoding audio, subtitle
line breaks, the output format, and the worker loop.

The two model calls themselves are not covered. That is what your first real run
on Grace is for.

---

## What changed from v2.0

| v2.0 | v3.0 |
|---|---|
| Whisper `large-v3` via WhisperX | Parakeet-TDT |
| Silero voice detection, plus a workaround for short segments | Not needed — removed |
| wav2vec2 forced alignment for word timings | Not needed — Parakeet times its own words |
| No speaker labels | Sortformer |
| Download the whole collection, then transcribe | Download one interview at a time, inside the job |
| Static split of files across GPUs | Shared queue |
| `whisperx`, `ctranslate2`, `nltk`, `langchain` | One package: `nemo_toolkit[asr]` |

Removed along the way: a `torch.load` monkeypatch and a cuDNN library-path fix,
both of which existed only to work around `ctranslate2`.

**The output format did not change.** v2.0 and v3.0 files are interchangeable
for the reviewer app.
