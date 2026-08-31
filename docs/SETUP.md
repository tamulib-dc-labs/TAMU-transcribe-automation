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

## Step 6 — Check the code loads

```bash
cd repo
python -c "import src.pipeline, src.config; print('ok')"
```

`ok` means the dependencies are installed and the package imports. This does
not use a GPU or download anything, so it is safe on a login node.

---

## Step 7 — Set your own details

Copy the example settings file and edit it:

```bash
cp config/local_settings.example.py config/local_settings.py
```

```python
# config/local_settings.py
smb_username = "your_netid"
smb_password = "your_netid_password"

git_owner    = "your_github_org"
git_username = "your_github_username"
git_token    = "ghp_xxxxxxxxxxxx"          # needs 'repo' scope
git_repo_name = "your_transcripts_repo"

sheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"
cache_dir = "/scratch/user/your_netid/asr/cache"
```

**Edit this file, not `src/config.py`.** This repository is public, so a token
committed to `src/config.py` would be pushed to GitHub and revoked within
minutes. `config/local_settings.py` is gitignored and never leaves your machine.

Anything you leave out keeps its default. Every setting is explained in
[CONFIGURATION.md](CONFIGURATION.md).

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

python repo/scripts/run_pipeline.py \
    --source local \
    --input data/oral_input \
    --max-files 1 \
    --skip-upload
```

That submits one job and exits. `--skip-upload` keeps this test off GitHub.

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
python repo/scripts/run_pipeline.py
```

(If you would rather not write credentials down, leave them out of
`local_settings.py` and `export GIT_TOKEN=...` and `export SMB_PASSWORD=...`
instead.)

This submits one job and exits straight away:

```
  Submitted job 1234567
  squeue -u $USER
  tail -f transcribe_Out.1234567
```

You can log out; Slurm keeps it going. Nothing runs on the login node.

Useful flags:

| Flag | Does |
|---|---|
| `--from-json` | Take the list from the reviewer app instead of the spreadsheet |
| `--max-files 5` | Only do five interviews — good for a first real run |
| `--skip-upload` | Leave transcripts on disk instead of pushing to GitHub |
| `--no-diarize` | Words only, no speaker labels |
| `--wait` | Stay attached and print job status until it finishes |

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

The GPU job asks for 4 hours. If a run is cut off, just start it again:

```bash
python repo/scripts/run_pipeline.py
```

You can also re-submit the filled copy the last run left behind, which skips
re-reading the command line:

```bash
sbatch run_transcribe.slurm
```

(`config/run.slurm` itself is a template with `{{PLACEHOLDER}}` in it, so it
cannot be submitted directly. `run_pipeline.py` fills it in and writes
`run_transcribe.slurm` next to it.)

Finished interviews are skipped. Interviews that were half-done when the job
died are put back in the queue automatically. Nothing is lost and nothing is
transcribed twice.

---

## Retrying failures

```bash
python repo/scripts/transcribe.py status  --queue data/queue --failures
python repo/scripts/transcribe.py requeue --queue data/queue
sbatch run_transcribe.slurm
```

---

## Quick reference

| Command | Does |
|---|---|
| `python scripts/run_pipeline.py` | Start (or resume) the pipeline |
| `sbatch run_transcribe.slurm` | Re-submit the last filled job |
| `squeue -u $USER` | Are my jobs running? |
| `scancel <jobid>` | Stop a job |
| `transcribe.py status --queue Q` | How far along am I? |
| `transcribe.py status --queue Q --failures` | What failed and why |
| `transcribe.py requeue --queue Q` | Try the failures again |
| `tail -f transcribe_Out.*` | Live log |

Something not working? → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
