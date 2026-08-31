"""
Pipeline orchestrator.

Two halves, and they run in different places:

    submit()   on a login node. Fills in config/run.slurm and sbatch's it.
               That is all a login node ever does.

    execute()  inside the Slurm job. The actual pipeline: list the work,
               transcribe it, upload the results.

Transcription itself is scripts/transcribe.py - Parakeet-TDT for words and
timings, Sortformer for speaker turns.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.cli import quote
from src.config import get_config
from src.utils.file_manager import FileManager, CommandRunner
from src.utils.logger import Logger
from src.git.uploader import GitUploader
from src.git.config_repo import ConfigRepoManager


class TranscriptionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, skip_upload: bool = False, wait: bool = False,
                 job_args: list | None = None):
        """Initialize the pipeline with configuration."""
        self.config = get_config()
        self.skip_upload = skip_upload
        self.wait = wait
        # Command-line options to hand to the job, so a flag typed on the
        # login node is not silently lost on the way to the compute node.
        self.job_args = list(job_args or [])
        self.file_manager = FileManager()
        self.command_runner = CommandRunner()
        
        # For from_json mode: store processed entries and their info
        self._processed_entries: List[Dict[str, Any]] = []
        self._config_repo_manager = None
    
    def submit(self) -> int:
        """Fill in the job template and hand it to Slurm. Login node only.

        `sbatch` is the one thing that has to happen on a login node, so it is
        the only thing that does. Everything else is in execute(), which runs
        inside the job.
        """
        Logger.log_info("Submitting the transcription pipeline")
        Logger.log_info(f"Working directory: {self.config.working_dir}")
        Logger.log_info(f"Source: {self.config.resolved_source}")

        with open(self.config.slurm_job_path, "r") as handle:
            content = self._fill_template(handle.read())

        written = os.path.join(self.config.working_dir, "run_transcribe.slurm")
        with open(written, "w") as handle:
            handle.write(content)

        job_id = self.command_runner.submit_slurm_job(written)
        if not job_id:
            Logger.log_error("Nothing was submitted")
            return 1

        Logger.log_info("")
        Logger.log_info(f"  Submitted job {job_id}")
        Logger.log_info(f"  squeue -u $USER")
        Logger.log_info(f"  tail -f transcribe_Out.{job_id}")
        Logger.log_info("")

        if self.wait:
            self._monitor_slurm_job(job_id)

        return 0

    # Kept so `run()` still means "do the normal thing" from a login node.
    run = submit

    def execute(self) -> int:
        """The pipeline itself. Runs inside the Slurm job, never on a login node.

        List the work, transcribe it, upload the results - in one process, so
        there is one job to submit, one log to read and one thing to re-run.
        """
        Logger.log_info("Starting the transcription pipeline")
        Logger.log_info(f"Source: {self.config.resolved_source}")

        self._prepare_directories()

        if self.config.resolved_source == "json" and not self._prepare_config_repo():
            Logger.log_warning("No work list, so there is nothing to do.")
            return 0

        self._collect_completed()

        if self._fill_queue() != 0:
            Logger.log_error("Could not build the work queue")
            return 1

        if self._transcribe() != 0:
            Logger.log_error("The transcription step failed")
            return 1

        if self.skip_upload:
            Logger.log_info(
                f"--skip-upload: transcripts left in {self.config.oral_output_path}"
            )
            return 0

        self._upload_to_github()
        if self.config.resolved_source == "json":
            self._update_config_repo()

        Logger.log_info("Done")
        return 0

    def _fill_queue(self) -> int:
        """Enumerate the source into the work queue. Moves no audio."""
        Logger.log_step(1, "List the work", "STARTED")

        command = [
            sys.executable, self.config.transcribe_script_path, "fill",
            "--queue", self.config.queue_path,
            "--output", self.config.oral_output_path,
            "--source", self.config.resolved_source,
            "--skip-list", self.config.completed_list_path,
        ]
        if self.config.resolved_source == "json":
            command += ["--config-json", self.config.work_list_path]
        else:
            if self.config.resolved_source == "local":
                command += ["--input", self.config.resolved_input_dir]
            if self.config.max_files:
                command += ["--max-files", str(self.config.max_files)]

        code = subprocess.run(command).returncode
        Logger.log_step(1, "List the work", "COMPLETED" if code == 0 else "FAILED")
        return code

    def _transcribe(self) -> int:
        """Work the queue until it is empty or the deadline is reached."""
        Logger.log_step(2, "Transcribe", "STARTED")

        command = [
            sys.executable, self.config.transcribe_script_path, "work",
            "--queue", self.config.queue_path,
            "--output", self.config.oral_output_path,
            "--scratch", os.path.join(
                os.environ.get("TMPDIR", self.config.data_dir), "audio"
            ),
            "--parakeet-model", self.config.asr_model,
            "--sortformer-model", self.config.diarization_model,
            "--words-device", self.config.words_device,
            "--turns-device", self.config.turns_device,
            "--deadline-minutes", str(self.config.deadline_minutes),
            "--lease", str(self.config.lease_seconds),
            "--max-attempts", str(self.config.max_attempts),
            "--max-line-width", str(self.config.max_line_width),
            "--max-line-count", str(self.config.max_line_count),
        ]
        if self.config.language:
            command += ["--language", self.config.language]
        if not self.config.diarize:
            command.append("--no-diarize")
        if self.config.denoise:
            command.append("--denoise")
        if not self.config.parallel_models:
            command.append("--sequential-models")

        code = subprocess.run(command).returncode
        Logger.log_step(2, "Transcribe", "COMPLETED" if code == 0 else "FAILED")
        return code

    def _prepare_directories(self):
        """Make sure the working directories exist.

        Deliberately does NOT clear them, unlike v2.0. In v3.0 the output
        directory is what makes a run resumable - `fill` skips any interview
        that already has a JSON there - and in local mode the input directory
        holds the audio itself. Wiping either would re-transcribe the whole
        collection on every run, or delete the source.

        To start over deliberately, remove data/queue and data/oral_output by
        hand.
        """
        for path in (
            self.config.oral_input_path,
            self.config.oral_output_path,
            self.config.queue_path,
        ):
            self.file_manager.ensure_directory(path)
        Logger.log_step(3, "Prepared working directories", "COMPLETED")
    

    def _prepare_config_repo(self) -> bool:
        """from_json mode: fetch the work list from the reviewer repo.

        Clones (or updates) the reviewer repository, reads
        config-to-process.json, drops anything already transcribed, and writes
        what is left to a plain file for the SLURM job to enumerate.

        No audio is downloaded here - each worker fetches its own interview.
        """
        Logger.log_step(4, "Read work list from reviewer repo", "STARTED")

        self._config_repo_manager = ConfigRepoManager(
            repo_folder=self.config.config_repo_path,
            owner=self.config.git_owner,
            repo_name=self.config.config_repo_name,
            username=self.config.git_username,
            token=self.config.get_git_token(),
            config_json_path=self.config.config_json_path,
            output_config_path=self.config.output_config_path,
        )

        if not self._config_repo_manager.setup_repository():
            Logger.log_error("Could not clone or update the reviewer repository")
            return False

        entries = self._config_repo_manager.read_config_to_process()
        if not entries:
            Logger.log_warning("Nothing to process in config-to-process.json")
            return False

        if self.config.max_files:
            entries = entries[: self.config.max_files]

        # Imported here, not at the top: this runs inside the job, and pulling
        # in src.asr on the login node would load numpy for one 3-line
        # function.
        from src.asr.sources import slug

        # The output files are named after the entry's `name`, so record that
        # here for the upload step to match them up afterwards.
        for entry in entries:
            entry["_folder_name"] = slug(entry.get("name", ""))
        self._processed_entries = entries

        work_list = Path(self.config.work_list_path)
        work_list.parent.mkdir(parents=True, exist_ok=True)
        work_list.write_text(json.dumps(entries, indent=2), encoding="utf-8")

        Logger.log_step(
            4, f"Queued {len(entries)} interview(s) from the reviewer repo", "COMPLETED"
        )
        return True

    def _collect_completed(self) -> int:
        """List interviews that already have a transcript in the GitHub repo.

        data/oral_output lives on /scratch, which is purged periodically, so it
        is a cache rather than a record. The transcripts repository is the
        record. Without this check a purge means re-transcribing and
        re-uploading the whole collection.

        Only filenames are needed, so this uses a blobless, checkout-free clone
        in its own directory. It never touches the uploader's working copy -
        an earlier version reused it and hit "untracked working tree files
        would be overwritten by checkout", which failed the whole step.

        Never fatal: if the repository cannot be reached the run proceeds using
        the local folder alone and may redo some work.
        """
        completed_list = Path(self.config.completed_list_path)
        completed_list.parent.mkdir(parents=True, exist_ok=True)

        if not self.config.check_transcripts_repo:
            completed_list.write_text("[]", encoding="utf-8")
            return 0

        Logger.log_step(5, "Check which interviews are already transcribed", "STARTED")
        try:
            done = self._list_transcribed()
        except Exception as exc:  # noqa: BLE001 - a slow start beats a failed run
            Logger.log_warning(
                f"Could not read {self.config.git_repo_name} ({exc}). Continuing "
                "with the local output folder only - some interviews may be redone."
            )
            completed_list.write_text("[]", encoding="utf-8")
            return 0

        completed_list.write_text(json.dumps(done, indent=2), encoding="utf-8")
        Logger.log_step(
            5,
            f"{len(done)} interview(s) already transcribed in {self.config.git_repo_name}",
            "COMPLETED",
        )
        return len(done)

    def _list_transcribed(self) -> list:
        """Filenames under json/ on the transcripts repo's main branch.

        Reads the remote's file list without creating a working tree, so it
        cannot collide with anything already on disk.
        """
        index_dir = os.path.join(self.config.data_dir, ".transcripts-index")
        url = (
            f"https://{self.config.git_username}:{self.config.get_git_token()}"
            f"@github.com/{self.config.git_owner}/{self.config.git_repo_name}.git"
        )

        def git(*args, cwd=None):
            result = subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip().splitlines()[-1:] or "git failed")
            return result.stdout

        if not os.path.isdir(os.path.join(index_dir, ".git")):
            os.makedirs(os.path.dirname(index_dir), exist_ok=True)
            # --filter=blob:none fetches the file *names* without their contents;
            # --no-checkout means no working tree is ever written.
            git("clone", "--filter=blob:none", "--no-checkout", url, index_dir)
        else:
            git("remote", "set-url", "origin", url, cwd=index_dir)
            git("fetch", "--filter=blob:none", "origin", "main", cwd=index_dir)

        listing = git("ls-tree", "-r", "--name-only", "origin/main", "json/", cwd=index_dir)
        return sorted(
            Path(line).stem
            for line in listing.splitlines()
            if line.strip().endswith(".json")
        )

    def _fill_template(self, slurm_content: str) -> str:
        """Substitute the {{PLACEHOLDER}}s in config/run.slurm.

        Only four, because the job runs Python that imports src/config.py for
        itself. Settings do not have to be threaded through the shell - change
        config.py and the next job picks it up.
        """
        replacements = {
            "{{VENV_ACTIVATE_PATH}}": f"{self.config.venv_path}/bin/activate",
            "{{HF_CACHE}}": self.config.hf_cache,
            "{{MODULE_LOAD}}": self.config.module_load_command,
            "{{JOB_SCRIPT}}": self.config.job_script_path,
            "{{JOB_ARGS}}": quote(self.job_args),
            "{{WEB_PROXY}}": (
                "ml WebProxy" if self.config.use_web_proxy
                else "# WebProxy disabled in config"
            ),
        }
        for placeholder, value in replacements.items():
            slurm_content = slurm_content.replace(placeholder, value)
        return slurm_content

    def _monitor_slurm_job(self, job_id: str):
        """Monitor SLURM job until completion."""
        Logger.log_step(9, f"Monitor SLURM job {job_id}", "STARTED")
        
        Logger.log_info(f"Job ID: {job_id}")
        
        while True:
            status = self.command_runner.check_slurm_job_status(job_id)
            
            if status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
                Logger.log_info(f"Job {job_id} - Status: {status}")
                break
            
            Logger.log_info(f"Job {job_id} - Status: {status}")
            time.sleep(self.config.check_interval_mins * 60)
    
    def _upload_to_github(self):
        """Upload transcription results to GitHub."""
        Logger.log_step(10, "Upload results to GitHub", "STARTED")
        
        token = self.config.get_git_token()
        
        self._uploader = GitUploader(
            source_folder=self.config.oral_output_path,
            repo_folder=self.config.git_repo_path,
            owner=self.config.git_owner,
            repo_name=self.config.git_repo_name,
            username=self.config.git_username,
            token=token
        )
        
        if self._uploader.upload():
            Logger.log_step(10, "Upload results to GitHub", "COMPLETED")
        else:
            Logger.log_error("GitHub upload failed")
    

    def _update_config_repo(self):
        """Write the transcript links back into the reviewer repo (json mode).

        Only interviews that actually have files on GitHub Pages get a row.
        The work list is no longer filtered before a run, so it names every
        interview in the collection - writing all of them here would put empty
        `url` and `vtt` fields against interviews that were never transcribed,
        and the reviewer app would show them as broken links.
        """
        Logger.log_step(11, "Update config.json in reviewer repo", "STARTED")

        if not self._processed_entries:
            Logger.log_warning("No entries to record")
            return

        if not self._config_repo_manager:
            Logger.log_error("Config repo manager not initialised")
            return

        uploader = getattr(self, "_uploader", None)
        if not uploader:
            Logger.log_error("Uploader not available, cannot build the links")
            return

        # Output files are named after the entry's folder name, not its `name`.
        folder_names = [
            entry["_folder_name"]
            for entry in self._processed_entries
            if entry.get("_folder_name")
        ]
        file_urls = uploader.get_uploaded_file_urls(folder_names)

        new_entries = []
        skipped = 0
        for entry in self._processed_entries:
            name = entry.get("name", "")
            urls = file_urls.get(entry.get("_folder_name", name), {})
            json_url = urls.get("json_url") or ""
            vtt_url = urls.get("vtt_url") or ""

            if not json_url and not vtt_url:
                skipped += 1
                continue

            new_entries.append({
                "audio": entry.get("audio", ""),
                "url": json_url,
                "vtt": vtt_url,
                "name": name,
            })
            Logger.log_info(f"  {name}: JSON={json_url}, VTT={vtt_url}")

        if skipped:
            Logger.log_info(
                f"  {skipped} entr(ies) have no transcript yet and were left alone"
            )

        if not new_entries:
            Logger.log_warning("No transcripts to record in config.json")
            return

        if not self._config_repo_manager.update_config_json(new_entries):
            Logger.log_error("Failed to write config.json")
            return

        if not self._config_repo_manager.commit_and_push():
            Logger.log_error("Failed to push config.json")
            return

        Logger.log_step(11, "Update config.json in reviewer repo", "COMPLETED")
