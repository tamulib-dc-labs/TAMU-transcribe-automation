"""
Configuration management for WhisperX Transcription Automation Pipeline.
All configurable parameters are defined here as a centralized Config class.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Main configuration class for the transcription pipeline."""
    
    # --- Working Directory ---
    working_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # --- Pipeline Settings ---
    check_interval_mins: int = 5  # SLURM job status check interval
    
    # --- Virtual Environment ---
    venv_name: str = "venv"
    
    # --- Environment Modules ---
    module_load_command: str = "ml GCCcore/10.3.0 Python FFmpeg CUDA"
    
    # --- Data Directories ---
    data_folder: str = "data"
    oral_input_folder: str = "oral_input"
    oral_output_folder: str = "oral_output"
    
    # --- SMB Network Share Settings ---
    smb_server: str = "cifs.library.tamu.edu"
    smb_share: str = "digital_project_management"
    smb_base_path: str = "edge-grant/GB_38253_MP3s"
    smb_username: str = "jvk_chaitanya"
    smb_password: str = ""  # Leave empty to prompt
    
    # --- Google Sheets Settings ---
    sheet_url: str = "https://docs.google.com/spreadsheets/d/16cHa57n7rJmS744nMH2dY2H4IKLP5fMeHJ0iY8w85EM/edit?usp=sharing"
    max_folders: int = 20
    
    # --- GitHub Settings ---
    git_owner: str = "tamulib-dc-labs"
    git_repo_name: str = "edge-grant-json-and-vtts"
    git_username: str = "JvkChaitanya"
    git_token: str = ""  # Set via environment variable GIT_TOKEN
    
    # --- From JSON Mode Settings ---
    from_json: bool = False  # Flag to enable JSON-based processing
    config_repo_name: str = "edge-grant-reviewer"  # Reviewer repo name
    config_json_path: str = "public/config-to-process.json"  # Input config in reviewer repo
    output_config_path: str = "public/config.json"  # Output config to update in reviewer repo
    max_json_files: int = 20  # Maximum number of files to process from JSON (0 = no limit)
    
    # --- Cache Directories ---
    cache_dir: str = "/scratch/group/tamu_libr_dc/cache"
    
    # --- Transcription Settings ---
    whisper_model: str = "large-v3"
    batch_size: int = 16
    compute_type: str = "float16"
    language: Optional[str] = None  # None for auto-detect
    alignment_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    max_line_width: int = 42
    max_line_count: int = 2
    highlight_words: bool = False
    
    # --- Model Download Settings ---
    download_models_before_slurm: bool = True
    
    # --- Derived Properties ---
    @property
    def venv_path(self) -> str:
        """Full path to virtual environment."""
        return os.path.join(self.working_dir, self.venv_name)
    
    @property
    def venv_python(self) -> str:
        """Path to Python executable in virtual environment."""
        return os.path.join(self.venv_path, "bin", "python")
    
    @property
    def venv_pip(self) -> str:
        """Path to pip executable in virtual environment."""
        return os.path.join(self.venv_path, "bin", "pip")
    
    @property
    def git_repo_path(self) -> str:
        """Path to git repository (one level above working directory)."""
        return os.path.join(os.path.dirname(self.working_dir), self.git_repo_name)
    
    @property
    def config_repo_path(self) -> str:
        """Path to config repository (edge-grant-reviewer, one level above working directory)."""
        return os.path.join(os.path.dirname(self.working_dir), self.config_repo_name)
    
    @property
    def data_dir(self) -> str:
        """Path to data directory."""
        return os.path.join(self.working_dir, self.data_folder)
    
    @property
    def oral_input_path(self) -> str:
        """Path to oral input directory."""
        return os.path.join(self.data_dir, self.oral_input_folder)
    
    @property
    def oral_output_path(self) -> str:
        """Path to oral output directory."""
        return os.path.join(self.data_dir, self.oral_output_folder)
    
    @property
    def hf_cache(self) -> str:
        """HuggingFace models cache path."""
        return os.path.join(self.cache_dir, "huggingface")
    
    @property
    def nltk_cache(self) -> str:
        """NLTK data cache path."""
        return os.path.join(self.cache_dir, "nltk_data")
    
    @property
    def model_dir(self) -> Optional[str]:
        """Optional specific model directory path."""
        return None  # Can be overridden if needed
    
    # --- Script Paths ---
    @property
    def download_script_path(self) -> str:
        """Path to download automation script."""
        return os.path.join(self.working_dir, "legacy", "download_automation_3.py")
    
    @property
    def model_download_script_path(self) -> str:
        """Path to model download script (legacy - pipeline uses inline venv script)."""
        return os.path.join(self.working_dir, "legacy", "d_whisperx.py")
    
    @property
    def transcribe_script_path(self) -> str:
        """Path to transcription script."""
        return os.path.join(self.working_dir, "scripts", "transcribe.py")
    
    @property
    def git_upload_script_path(self) -> str:
        """Path to git upload script (not used - GitUploader class used instead)."""
        return os.path.join(self.working_dir, "legacy", "git_upload.py")
    
    @property
    def slurm_job_path(self) -> str:
        """Path to SLURM job template."""
        return os.path.join(self.working_dir, "config", "run.slurm")
    
    @property
    def requirements_path(self) -> str:
        """Path to requirements.txt file."""
        return os.path.join(self.working_dir, "requirements.txt")
    
    def get_smb_password(self) -> str:
        """Get SMB password from config, environment, or prompt."""
        import getpass
        
        if self.smb_password:
            return self.smb_password
        
        env_password = os.environ.get('SMB_PASSWORD')
        if env_password:
            return env_password
        
        try:
            return getpass.getpass(f"Password for {self.smb_username}: ")
        except Exception as e:
            raise ValueError(f"Error getting password: {e}")
    
    def get_git_token(self) -> str:
        """Get GitHub token from config or environment."""
        if self.git_token:
            return self.git_token
        
        env_token = os.environ.get('GIT_TOKEN')
        if env_token:
            return env_token
        
        raise ValueError("GitHub token not set. Set GIT_TOKEN environment variable or update config.")


# Singleton instance
_config_instance = None


def get_config() -> PipelineConfig:
    """Get or create the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = PipelineConfig()
    return _config_instance


def reset_config():
    """Reset configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None
