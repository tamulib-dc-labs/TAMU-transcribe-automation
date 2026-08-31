"""Your own settings. Copy this to local_settings.py and edit.

    cp config/local_settings.example.py config/local_settings.py

local_settings.py is gitignored, so anything here stays on your machine.
This repository is public - a token written into src/config.py would be
pushed to GitHub and revoked within minutes.

Anything you set here overrides src/config.py. Leave out what you do not
need; the defaults still apply.
"""

# --- who you are -----------------------------------------------------------

smb_username = "your_netid"
smb_password = "your_netid_password"        # or leave out and export SMB_PASSWORD

# --- GitHub ----------------------------------------------------------------

git_owner = "tamulib-dc-labs"               # org or user that owns the repos
git_username = "YourGitHubUsername"
git_token = "ghp_xxxxxxxxxxxxxxxxxxxx"      # needs 'repo' scope

git_repo_name = "edge-grant-json-and-vtts"  # transcripts are pushed here
config_repo_name = "edge-grant-reviewer"    # the reviewer app's repo

# --- what to transcribe ----------------------------------------------------

sheet_url = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit"

# source = "local"                          # smb | json | local
# input_dir = "/scratch/user/YOUR_NETID/asr/data/oral_input"
# max_files = 1                             # cap a test run

# --- environment modules ---------------------------------------------------

# Goes into the Slurm job verbatim. Pin the Python version: NeMo needs 3.11.4
# or later to unpack its own model files, and an unversioned `ml Python` under
# GCCcore/12.3.0 gives 3.11.3. Check `ml spider Python` on your cluster.
# Rebuild the venv after changing this - the venv decides the interpreter.
# module_load_command = "ml GCCcore/13.2.0 Python/3.11.5 FFmpeg CUDA"  # the default

# --- where things live -----------------------------------------------------

cache_dir = "/scratch/user/YOUR_NETID/asr/cache"

# --- tuning (optional) -----------------------------------------------------

# diarize = True                            # speaker labels
# denoise = False                           # noise reduction before ASR
# words_device = "cuda:0"
# turns_device = "cuda:1"
