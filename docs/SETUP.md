# Setup — running this on your own HPRC account

From nothing to your first transcript. Allow about an hour, most of it waiting
for things to install.

Everything happens on Grace. You never download audio or models to your own
machine.

---

## Before you start

You need:

1. **An HPRC account with Grace access.** Apply at
   <https://hprc.tamu.edu/apply/>. Accounts expire every September and must be
   renewed.
2. **An allocation with GPU hours.** Your PI or department usually holds this.
   Check yours with `myproject -l`.
3. **A GitHub personal access token** — only if you want transcripts uploaded
   automatically. Create one at <https://github.com/settings/tokens> with
   `repo` scope.

---

## Step 1 — Log in

```bash
ssh yournetid@grace.hprc.tamu.edu
```

You land on a **login node**. Login nodes are for setting things up and
submitting jobs. Never run transcription here — it belongs in a job.

---

## Step 2 — Pick a working folder

Use `/scratch`, not your home directory. Home is limited to 10 GB; scratch gives
you 1 TB.

```bash
mkdir -p /scratch/user/$USER/asr
cd /scratch/user/$USER/asr
```

Everything from here lives under that folder.

> **Note:** files on `/scratch` are purged after a period of inactivity. Your
> transcripts get uploaded to GitHub, so that is fine — but do not leave the
> only copy of anything important there.

---

## Step 3 — Get the code

```bash
git clone https://github.com/tamulib-dc-labs/TAMU-transcribe-automation.git repo
cd repo
git checkout v3.0
```

---

## Step 4 — Load the right software modules

```bash
ml GCCcore/12.3.0 Python FFmpeg CUDA
```

**This matters.** The models need Python 3.10 or newer. The module the old
version used (`GCCcore/10.3.0 Python`) is Python 3.9 and will not work.

Check what you got:

```bash
python3 --version        # should say 3.11 or 3.12
```

If that module name does not exist on your cluster, find one that works:

```bash
module spider Python
```

Then update `module_load_command` in `src/config.py` to match.

---

## Step 5 — Make a virtual environment and install

```bash
cd /scratch/user/$USER/asr
python -m venv venv
source venv/bin/activate

# Compute nodes and login nodes both need the proxy for downloads
ml WebProxy

pip install --upgrade pip
pip install -r repo/requirements.txt
```

**This is the step most likely to fail.** `nemo_toolkit[asr]` is a large
package. If it errors, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#nemo-will-not-install).

Confirm it worked:

```bash
python -c "import nemo.collections.asr; print('NeMo OK')"
```

---

## Step 6 — Check the tests pass

```bash
cd repo
pip install pytest
pytest -q
```

You should see `210 passed`. These do not use a GPU or download anything, so
they are safe to run on a login node. If they pass, the code is installed
correctly.

---

## Step 7 — Set your own details

Open `src/config.py` and change these to yours:

```python
smb_username: str = "your_netid"

git_owner:    str = "your_github_org"
git_repo_name: str = "your_transcripts_repo"
git_username: str = "your_github_username"

sheet_url: str = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

cache_dir: str = "/scratch/user/YOUR_NETID/asr/cache"
```

Every setting is explained in [CONFIGURATION.md](CONFIGURATION.md).

---

## Step 8 — Check the file share is reachable *(SMB mode only)*

Skip this if your audio comes from URLs (`--from-json` mode).

Compute nodes reach the internet through a proxy, but that proxy only carries
web traffic. The file share uses a different kind of connection. So test it
from an actual compute node before relying on it:

```bash
srun --partition=gpu --gres=gpu:a100:1 --time=00:30:00 --pty bash

ml GCCcore/12.3.0 Python
source /scratch/user/$USER/asr/venv/bin/activate
python -c "
import smbclient, getpass
smbclient.register_session('cifs.library.tamu.edu',
                           username='your_netid',
                           password=getpass.getpass())
print('SMB reachable from compute node')
"
exit
```

**If it prints the success message**, you are done — everything runs on the
compute node.

**If it fails**, the compute nodes cannot reach the share. That is fine, it just
means you copy the audio to scratch once from a login node and tell the job to
read it from there:

```bash
# on a login node
python repo/scripts/transcribe.py fill \
    --queue  /scratch/user/$USER/asr/data/queue \
    --output /scratch/user/$USER/asr/data/oral_output \
    --source local \
    --input  /scratch/user/$USER/asr/data/oral_input
```

Nothing else changes.

---

## Step 9 — Try one interview first

Do not start with the whole collection. Put one recording somewhere on scratch
and run it through:

```bash
cd /scratch/user/$USER/asr

mkdir -p data/oral_input
cp /path/to/one_interview.mp3 data/oral_input/

python repo/scripts/transcribe.py fill \
    --queue  data/queue \
    --output data/oral_output \
    --source local \
    --input  data/oral_input

sbatch repo/config/run.slurm
```

Watch it:

```bash
squeue -u $USER                 # is it running?
tail -f transcribe_Out.*        # what is it doing?
```

The **first** run is slow — it downloads about 3 GB of model weights. Later runs
reuse them from the cache.

When it finishes:

```bash
cat data/oral_output/json/one_interview.json | head -40
cat data/oral_output/vtts/one_interview.vtt
```

Check that the words are right and the speaker labels make sense before you run
the whole collection.

---

## Step 10 — Run the collection

```bash
export GIT_TOKEN=your_github_token
export SMB_PASSWORD=your_netid_password     # SMB mode only

python repo/scripts/run_pipeline.py
```

This submits the job, waits, and uploads results when it is done.

Useful flags:

| Flag | Does |
|---|---|
| `--from-json` | Take the list from the reviewer app instead of the spreadsheet |
| `--max-files 5` | Only do five interviews — good for a first real run |
| `--skip-upload` | Leave transcripts on disk instead of pushing to GitHub |
| `--no-diarize` | Words only, no speaker labels |

Re-running is always safe. Interviews that already have a transcript are
skipped, so nothing is redone and nothing is deleted.

Check on it any time:

```bash
python repo/scripts/transcribe.py status --queue data/queue --failures
```

```json
{ "pending": 41, "claimed": 4, "done": 12, "failed": 0 }
```

- **pending** — waiting
- **claimed** — being worked on right now
- **done** — finished
- **failed** — gave up after 3 tries; `--failures` shows why

---

## If a job runs out of time

Grace allows 4 days per GPU job. If a run is cut off, just submit again:

```bash
sbatch repo/config/run.slurm
```

Finished interviews are skipped. Interviews that were half-done when the job
died are put back in the queue automatically. Nothing is lost and nothing is
transcribed twice.

---

## Retrying failures

```bash
python repo/scripts/transcribe.py status  --queue data/queue --failures
python repo/scripts/transcribe.py requeue --queue data/queue
sbatch repo/config/run.slurm
```

---

## Quick reference

| Command | Does |
|---|---|
| `sbatch config/run.slurm` | Start (or resume) transcription |
| `squeue -u $USER` | Are my jobs running? |
| `scancel <jobid>` | Stop a job |
| `transcribe.py status --queue Q` | How far along am I? |
| `transcribe.py status --queue Q --failures` | What failed and why |
| `transcribe.py requeue --queue Q` | Try the failures again |
| `tail -f transcribe_Out.*` | Live log |

Something not working? → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
