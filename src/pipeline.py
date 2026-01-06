"""
Main pipeline orchestrator for WhisperX Transcription Automation.

This module coordinates all pipeline steps including:
- Environment setup
- File downloads
- Model preparation
- SLURM job submission
- Git upload
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from src.config import get_config
from src.utils.file_manager import FileManager, CommandRunner
from src.utils.logger import Logger
from src.git.uploader import GitUploader
from src.git.config_repo import ConfigRepoManager
from src.utils.json_downloader import JsonAudioDownloader


class TranscriptionPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the pipeline with configuration."""
        self.config = get_config()
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
        
        # Step 4: Download audio files (different sources based on mode)
        if self.config.from_json:
            self._download_audio_from_json()
        else:
            self._download_audio_files()
        
        # Step 5: Download models
        if self.config.download_models_before_slurm:
            self._download_models()
        
        # Step 6: Download NLTK data
        self._download_nltk_data()
        
        # Step 7: Submit SLURM job
        job_id = self._submit_slurm_job()
        
        if job_id:
            # Step 8: Monitor job
            self._monitor_slurm_job(job_id)
            
            # Step 8.5: Rename output files for from_json mode
            if self.config.from_json:
                self._rename_output_files()
            
            # Step 9: Upload results to GitHub
            self._upload_to_github()
            
            # Step 10: Update config repo (only for from_json mode)
            if self.config.from_json:
                self._update_config_repo()
        
        Logger.log_info("Pipeline execution completed")
    
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
        """Clear and prepare input/output directories."""
        # Clear input directory
        self.file_manager.clear_directory(self.config.oral_input_path)
        Logger.log_step(4, f"Cleared {self.config.oral_input_path}", "COMPLETED")
        
        # Clear output directory
        self.file_manager.clear_directory(self.config.oral_output_path)
        Logger.log_step(5, f"Cleared {self.config.oral_output_path}", "COMPLETED")
    
    def _download_audio_files(self):
        """Download audio files using download script."""
        Logger.log_step(6, "Download audio files from network share", "STARTED")
        
        password = self.config.get_smb_password()
        
        download_cmd = [
            self.config.venv_python,
            self.config.download_script_path,
            "--server", self.config.smb_server,
            "--share", self.config.smb_share,
            "--username", self.config.smb_username,
            "--password", password,
            "--base-path", self.config.smb_base_path,
            "--local-path", self.config.oral_input_path,
            "--sheet-url", self.config.sheet_url,
            "--max-folders", str(self.config.max_folders)
        ]
        
        if not self.command_runner.run(download_cmd, 6, "Download audio files"):
            sys.exit(1)
    
    def _download_models(self):
        """Download WhisperX models for offline use via venv."""
        Logger.log_step(7, "Download WhisperX models", "STARTED")
        
        # Create a script to download models using the venv Python
        download_script = f"""
import os
import sys
from pathlib import Path

# Set cache directories
os.environ['HF_HOME'] = '{self.config.hf_cache}'
os.environ['HF_HUB_OFFLINE'] = '0'
os.makedirs('{self.config.hf_cache}', exist_ok=True)

# Check if model already exists
model_name = "{self.config.whisper_model}"
cache_path = Path('{self.config.hf_cache}')

# WhisperX models are stored in hub/models--Systran--faster-whisper-<model>
model_cache_pattern = f"**/models--Systran--faster-whisper-*{{model_name}}*"
existing_models = list(cache_path.glob(model_cache_pattern))

if existing_models:
    print(f"Model '{{model_name}}' already exists in cache:")
    print(f"  {{existing_models[0]}}")
    print(f"  Skipping download.")
    sys.stdout.flush()
    model_exists = True
else:
    print(f"Model '{{model_name}}' not found in cache, downloading...")
    sys.stdout.flush()
    model_exists = False

# Import after environment is set
import torch
import whisperx
import functools

# Apply PyTorch 2.6+ compatibility patch
try:
    torch.serialization.add_safe_globals = lambda x: None
    _original_torch_load = torch.load
    
    @functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    
    torch.load = _patched_torch_load
    print("PyTorch 2.6+ compatibility patch applied")
    sys.stdout.flush()
except Exception as e:
    print(f"WARNING: Failed to apply PyTorch patch: {{e}}")
    sys.stdout.flush()

# Download models
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8"  # Safe for both CPU and GPU
if device == "cpu":
    print(f"Using CPU mode with compute_type={{compute_type}}")
else:
    print(f"Using GPU mode with compute_type={{compute_type}}")
sys.stdout.flush()

# Download WhisperX model only if not exists
if not model_exists:
    print(f"Downloading WhisperX model: {{model_name}}...")
    sys.stdout.flush()
    
    try:
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        print(f"WhisperX model '{{model_name}}' downloaded successfully!")
        sys.stdout.flush()
        del model
    except Exception as e:
        print(f"Error downloading WhisperX model: {{e}}")
        sys.stdout.flush()
        sys.exit(1)

# Download alignment models
languages = {self.config.alignment_languages}
print(f"\\nChecking alignment models for languages: {{', '.join(languages)}}...")
sys.stdout.flush()

for lang in languages:
    # Check if alignment model exists
    align_cache_pattern = f"**/models--*alignment*{{lang}}*"
    existing_align = list(cache_path.glob(align_cache_pattern))
    
    if existing_align:
        print(f"  Alignment model for '{{lang}}' already exists, skipping.")
        sys.stdout.flush()
    else:
        try:
            print(f"  Downloading alignment model for '{{lang}}'...")
            sys.stdout.flush()
            align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
            print(f"  Alignment model for '{{lang}}' downloaded!")
            sys.stdout.flush()
            del align_model
        except Exception as e:
            print(f"  Could not download alignment for '{{lang}}': {{e}}")
            sys.stdout.flush()

print(f"\\n{{'='*60}}")
print(f"All models cached at: {self.config.hf_cache}")
print(f"{{'='*60}}")
"""
        
        # Run via venv Python
        download_cmd = [self.config.venv_python, "-c", download_script]
        if not self.command_runner.run(download_cmd, 7, "Download WhisperX models"):
            Logger.log_warning("Model download failed")
    
    def _download_nltk_data(self):
        """Download NLTK punkt_tab data."""
        Logger.log_step("7.5", "Download NLTK data", "STARTED")
        
        nltk_script = f"""
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab', download_dir='{self.config.nltk_cache}', quiet=False)
print("NLTK punkt_tab downloaded successfully")
"""
        
        nltk_cmd = [self.config.venv_python, "-c", nltk_script]
        self.command_runner.run(nltk_cmd, "7.5", "Download NLTK punkt_tab data")
    
    def _submit_slurm_job(self) -> str:
        """Inject paths and submit SLURM job."""
        Logger.log_step(8, "Update and submit SLURM job", "STARTED")
        
        # Read SLURM template
        with open(self.config.slurm_job_path, "r") as f:
            slurm_content = f.read()
        
        # Inject paths
        slurm_content = slurm_content.replace("{{VENV_ACTIVATE_PATH}}", f"{self.config.venv_path}/bin/activate")
        slurm_content = slurm_content.replace("{{HF_CACHE}}", self.config.hf_cache)
        slurm_content = slurm_content.replace("{{NLTK_CACHE}}", self.config.nltk_cache)
        slurm_content = slurm_content.replace("{{TRANSCRIBE_SCRIPT}}", self.config.transcribe_script_path)
        slurm_content = slurm_content.replace("{{WHISPER_MODEL}}", self.config.whisper_model)
        slurm_content = slurm_content.replace("{{ORAL_INPUT_PATH}}", self.config.oral_input_path)
        slurm_content = slurm_content.replace("{{ORAL_OUTPUT_PATH}}", self.config.oral_output_path)
        
        # Inject language argument (or omit if None for auto-detection)
        if self.config.language:
            language_arg = f'--language "{self.config.language}" \\\n    '
        else:
            language_arg = ''
        slurm_content = slurm_content.replace("{{LANGUAGE_ARG}}", language_arg)
        
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
    
    def _download_audio_from_json(self):
        """Download audio files from JSON config (from_json mode)."""
        Logger.log_step(6, "Download audio files from JSON config", "STARTED")
        
        token = self.config.get_git_token()
        
        # Step 1: Setup config repository (edge-grant-reviewer)
        Logger.log_info("Setting up config repository...")
        self._config_repo_manager = ConfigRepoManager(
            repo_folder=self.config.config_repo_path,
            owner=self.config.git_owner,
            repo_name=self.config.config_repo_name,
            username=self.config.git_username,
            token=token,
            config_json_path=self.config.config_json_path,
            output_config_path=self.config.output_config_path
        )
        
        if not self._config_repo_manager.setup_repository():
            Logger.log_error("Failed to setup config repository")
            sys.exit(1)
        
        # Step 2: Read config-to-process.json
        entries = self._config_repo_manager.read_config_to_process()
        
        if not entries:
            Logger.log_warning("No entries found in config-to-process.json")
            return
        
        Logger.log_info(f"Found {len(entries)} entries to process")
        
        # Step 3: Download audio files
        # Pass module load command for HPC FFmpeg loading
        downloader = JsonAudioDownloader(
            input_dir=self.config.oral_input_path,
            module_load_command=self.config.module_load_command
        )
        self._processed_entries = downloader.process_config_entries(
            entries,
            max_files=self.config.max_json_files
        )
        
        if not self._processed_entries:
            Logger.log_warning("No audio files were successfully downloaded")
            return
        
        Logger.log_step(6, f"Downloaded {len(self._processed_entries)} audio files", "COMPLETED")
    
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
    
    def _rename_output_files(self):
        """Rename output files from audio filename to folder name (from_json mode only).
        
        This ensures the JSON/VTT files use the 'name' field from config instead of
        the audio filename (which may include qualifiers like '-medium').
        """
        Logger.log_step("8.5", "Renaming output files to use folder names", "STARTED")
        
        if not self._processed_entries:
            Logger.log_warning("No processed entries for renaming")
            return
        
        json_dir = os.path.join(self.config.oral_output_path, "json")
        vtt_dir = os.path.join(self.config.oral_output_path, "vtts")
        
        renamed_count = 0
        
        for entry in self._processed_entries:
            folder_name = entry.get("_folder_name", "")  # This is the 'name' field
            audio_filename = entry.get("_audio_filename", "")
            
            if not folder_name or not audio_filename:
                continue
            
            # Get the base names
            audio_base = os.path.splitext(audio_filename)[0]  # e.g., "02_00113-a_01-medium"
            target_base = folder_name  # e.g., "02_00113-a_01"
            
            # Skip if they're the same
            if audio_base == target_base:
                continue
            
            # Rename JSON file
            old_json = os.path.join(json_dir, f"{audio_base}.json")
            new_json = os.path.join(json_dir, f"{target_base}.json")
            if os.path.exists(old_json) and not os.path.exists(new_json):
                os.rename(old_json, new_json)
                Logger.log_info(f"Renamed: {audio_base}.json -> {target_base}.json")
                renamed_count += 1
            
            # Rename VTT file
            old_vtt = os.path.join(vtt_dir, f"{audio_base}.vtt")
            new_vtt = os.path.join(vtt_dir, f"{target_base}.vtt")
            if os.path.exists(old_vtt) and not os.path.exists(new_vtt):
                os.rename(old_vtt, new_vtt)
                Logger.log_info(f"Renamed: {audio_base}.vtt -> {target_base}.vtt")
                renamed_count += 1
        
        Logger.log_step("8.5", f"Renamed {renamed_count} output files", "COMPLETED")

