"""
Main pipeline orchestrator.

Coordinates: environment setup, audio download, model cache check, filling the
work queue, SLURM submission, and git upload.

Transcription itself is Parakeet-TDT (words) plus Sortformer (speaker turns),
run by scripts/transcribe.py inside the SLURM job.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.config import get_config
from src.utils.file_manager import FileManager, CommandRunner
from src.asr.sources import slug
from src.utils.logger import Logger
from src.git.uploader import GitUploader
from src.git.config_repo import ConfigRepoManager


class TranscriptionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, skip_upload: bool = False):
        """Initialize the pipeline with configuration."""
        self.config = get_config()
        self.skip_upload = skip_upload
        self.file_manager = FileManager()
        self.command_runner = CommandRunner()
        
        # For from_json mode: store processed entries and their info
        self._processed_entries: List[Dict[str, Any]] = []
        self._config_repo_manager = None
    
    def run(self):
        """Execute the complete pipeline."""
        Logger.log_info("Starting Transcription Automation Pipeline")
        Logger.log_info(f"Working directory: {self.config.working_dir}")
        Logger.log_info(f"Mode: {'FROM_JSON' if self.config.from_json else 'SMB'}")
        
        # Step 1: Load environment modules
        self._load_modules()
        
        # Step 2: Setup Python environment
        self._setup_python_environment()
        
        # Step 3: Prepare directories
        self._prepare_directories()
        
        # Audio and model downloads happen INSIDE the SLURM job - compute nodes
        # reach the network through WebProxy, so nothing is staged here.
        #
        # from_json mode is the one exception: the list of interviews lives in
        # a private GitHub repo, so it is cloned here and written out as a plain
        # file the job can read. Still no audio - just the work list.
        if self.config.resolved_source == "json" and not self._prepare_config_repo():
            return 1

        # Step 4: Submit SLURM job
        job_id = self._submit_slurm_job()
        
        if not job_id:
            Logger.log_error("No job was submitted; nothing to wait for")
            return 1

        # Step 5: Monitor job
        self._monitor_slurm_job(job_id)

        if self.skip_upload:
            Logger.log_info(
                f"--skip-upload: transcripts left in {self.config.oral_output_path}"
            )
        else:
            # Step 6: Upload results to GitHub
            self._upload_to_github()

            # Step 7: Update config repo (only for from_json mode)
            if self.config.from_json:
                self._update_config_repo()

        Logger.log_info("Pipeline execution completed")
        return 0
    
    def _load_modules(self):
        """Load required environment modules."""
        Logger.log_step(1, "Load environment modules", "STARTED")
        try:
            subprocess.run(
                self.config.module_load_command,
                shell=True,
                check=True,
                executable='/bin/bash'
            )
            Logger.log_step(1, "Load environment modules", "COMPLETED")
        except Exception as e:
            Logger.log_warning(f"Failed to load modules: {e}. Continuing anyway...")
    
    def _setup_python_environment(self):
        """Setup or verify Python virtual environment."""
        venv_exists = os.path.exists(self.config.venv_python)
        
        if not venv_exists:
            # Create venv
            if not self.command_runner.run(
                f"python -m venv {self.config.venv_path}",
                3,
                f"Create virtual environment at {self.config.venv_path}",
                shell=True
            ):
                sys.exit(1)
            
            # Install requirements
            if os.path.exists(self.config.requirements_path):
                Logger.log_step("3.5", "Install Python dependencies", "STARTED")
                if not self.command_runner.run(
                    f"{self.config.venv_pip} install -r {self.config.requirements_path}",
                    "3.5",
                    "Install dependencies from requirements.txt",
                    shell=True
                ):
                    Logger.log_warning("Failed to install some dependencies")
        else:
            Logger.log_step(3, "Using existing virtual environment", "COMPLETED")
    
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

    def _submit_slurm_job(self) -> str:
        """Inject paths and submit SLURM job."""
        Logger.log_step(8, "Update and submit SLURM job", "STARTED")
        
        # Read SLURM template
        with open(self.config.slurm_job_path, "r") as f:
            slurm_content = f.read()
        
        # Inject paths
        slurm_content = slurm_content.replace("{{VENV_ACTIVATE_PATH}}", f"{self.config.venv_path}/bin/activate")
        slurm_content = slurm_content.replace("{{HF_CACHE}}", self.config.hf_cache)
        slurm_content = slurm_content.replace("{{TRANSCRIBE_SCRIPT}}", self.config.transcribe_script_path)
        slurm_content = slurm_content.replace("{{QUEUE_PATH}}", self.config.queue_path)
        slurm_content = slurm_content.replace("{{ORAL_OUTPUT_PATH}}", self.config.oral_output_path)
        slurm_content = slurm_content.replace("{{ASR_MODEL}}", self.config.asr_model)
        slurm_content = slurm_content.replace("{{DIARIZATION_MODEL}}", self.config.diarization_model)
        slurm_content = slurm_content.replace("{{DEADLINE_MINUTES}}", str(self.config.deadline_minutes))
        slurm_content = slurm_content.replace("{{LEASE_SECONDS}}", str(self.config.lease_seconds))
        slurm_content = slurm_content.replace("{{WORDS_DEVICE}}", self.config.words_device)
        slurm_content = slurm_content.replace("{{TURNS_DEVICE}}", self.config.turns_device)
        slurm_content = slurm_content.replace("{{MAX_ATTEMPTS}}", str(self.config.max_attempts))
        slurm_content = slurm_content.replace("{{MAX_LINE_WIDTH}}", str(self.config.max_line_width))
        slurm_content = slurm_content.replace("{{MAX_LINE_COUNT}}", str(self.config.max_line_count))
        slurm_content = slurm_content.replace(
            "{{WEB_PROXY}}",
            "ml WebProxy" if self.config.use_web_proxy else "# WebProxy disabled in config",
        )
        slurm_content = slurm_content.replace("{{SMB_USERNAME}}", self.config.smb_username)
        source = self.config.resolved_source
        slurm_content = slurm_content.replace("{{SOURCE}}", source)

        fill_args = []
        if source == "json":
            # The list was read on the login node; the job just enumerates it.
            fill_args.append(f'--config-json "{self.config.work_list_path}"')
        else:
            if source == "local":
                fill_args.append(f'--input "{self.config.resolved_input_dir}"')
            if self.config.max_files:
                fill_args.append(f"--max-files {self.config.max_files}")
        slurm_content = slurm_content.replace("{{FILL_ARGS}}", " ".join(fill_args))

        # Optional worker flags, assembled from config.
        extra = []
        if self.config.language:
            extra.append('--language "%s"' % self.config.language)
        if not self.config.diarize:
            extra.append("--no-diarize")
        if self.config.denoise:
            extra.append("--denoise")
        if not self.config.parallel_models:
            extra.append("--sequential-models")
        slurm_content = slurm_content.replace("{{EXTRA_ARGS}}", " ".join(extra))
        
        # Write updated SLURM file
        updated_slurm = os.path.join(self.config.working_dir, "run_job.slurm")
        with open(updated_slurm, "w") as f:
            f.write(slurm_content)
        
        # Submit job
        job_id = self.command_runner.submit_slurm_job(updated_slurm)
        
        if job_id:
            Logger.log_info(f"")
            Logger.log_info(f"{'='*80}")
            Logger.log_info(f"  SLURM JOB SUBMITTED SUCCESSFULLY")
            Logger.log_info(f"  Job ID: {job_id}")
            Logger.log_info(f"{'='*80}")
            Logger.log_info(f"")
        else:
            Logger.log_error("Failed to submit SLURM job")
        
        return job_id
    
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
        """Update config.json with new entries (from_json mode only)."""
        Logger.log_step(11, "Update config.json in reviewer repo", "STARTED")
        
        if not self._processed_entries:
            Logger.log_warning("No processed entries to update")
            return
        
        if not self._config_repo_manager:
            Logger.log_error("Config repo manager not initialized")
            return
        
        # Use folder names for URL lookup since files have been renamed to use folder names
        folder_names = [entry.get("_folder_name") for entry in self._processed_entries if entry.get("_folder_name")]
        
        if hasattr(self, '_uploader') and self._uploader:
            file_urls = self._uploader.get_uploaded_file_urls(folder_names)
        else:
            Logger.log_warning("Uploader not available, cannot get file URLs")
            file_urls = {}
        
        # Build new config entries
        new_entries = []
        for entry in self._processed_entries:
            name = entry.get("name", "")
            folder_name = entry.get("_folder_name", name)
            audio_url = entry.get("audio", "")
            
            # Get uploaded URLs using folder name as key (files were renamed)
            urls = file_urls.get(folder_name, {})
            json_url = urls.get("json_url", "")
            vtt_url = urls.get("vtt_url", "")
            
            new_entry = {
                "audio": audio_url,
                "url": json_url,
                "vtt": vtt_url,
                "name": name
            }
            new_entries.append(new_entry)
            Logger.log_info(f"  {name}: JSON={json_url}, VTT={vtt_url}")
        
        # Update config.json
        if not self._config_repo_manager.update_config_json(new_entries):
            Logger.log_error("Failed to update config.json")
            return
        
        # Commit and push changes
        if not self._config_repo_manager.commit_and_push():
            Logger.log_error("Failed to push config.json changes")
            return
        
        Logger.log_step(11, "Update config.json in reviewer repo", "COMPLETED")
    
