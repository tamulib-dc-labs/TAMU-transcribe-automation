"""
Configuration for the transcription pipeline.

ASR is Parakeet-TDT (words, timings, confidence) plus Sortformer (speaker
turns). Both are NeMo models sharing one environment.
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
    module_load_command: str = "ml GCCcore/12.3.0 Python FFmpeg CUDA"  # NeMo needs Python >= 3.10
    
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
    
    #: Cap how many interviews a run queues. 0 = the whole collection.
    max_files: int = 0

    # --- From JSON Mode Settings ---
    from_json: bool = False  # Flag to enable JSON-based processing
    config_repo_name: str = "edge-grant-reviewer"  # Reviewer repo name
    config_json_path: str = "public/config-to-process.json"  # Input config in reviewer repo
    output_config_path: str = "public/config.json"  # Output config to update in reviewer repo
    max_json_files: int = 20  # Maximum number of files to process from JSON (0 = no limit)
    
    # --- Network on compute nodes ---
    #: Grace compute nodes have no direct internet, but HPRC runs an HTTP proxy
    #: reachable from them. Loading the WebProxy module sets http_proxy and
    #: https_proxy, which is what lets the job download its own audio and its
    #: own model weights instead of needing anything staged in advance.
    #: Turning this off breaks both, unless the audio is already local.
    use_web_proxy: bool = True

    # --- Cache Directories ---
    cache_dir: str = "/scratch/group/tamu_libr_dc/cache"
    
    # --- Transcription Settings ---
    #: Words, timings and per-word confidence. Emits word timestamps natively,
    #: so there is no separate forced-alignment stage.
    asr_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    #: Speaker turns. Shares Parakeet's FastConformer frontend and its 16 kHz
    #: input, so both sides of the fusion sit on one measured time base.
    diarization_model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    #: Attach speaker labels. Off = words and timings only.
    diarize: bool = True
    #: Pin one model per GPU so they genuinely overlap. Same device for both
    #: means they contend instead - set parallel_models False in that case.
    words_device: str = "cuda:0"
    turns_device: str = "cuda:1"
    parallel_models: bool = True

    language: Optional[str] = "en"  # tag written to the JSON; models self-detect
    max_line_width: int = 42
    max_line_count: int = 2

    #: The old noisereduce pass, tuned for Whisper. Off by default: both models
    #: are trained on noise-augmented audio and normalise their own input, and
    #: spectral gating can add artefacts that hurt an end-to-end model.
    denoise: bool = False

    # --- Queue Settings ---
    #: Seconds a worker holds a claim before it can be reaped by another job.
    #: Must exceed the slowest single interview.
    lease_seconds: int = 5400
    max_attempts: int = 3
    #: Stop claiming this many minutes into the job, leaving room to finish the
    #: file in hand. Keep it below the Slurm wall clock.
    deadline_minutes: int = 2820
    
    
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
    def venv_bin_path(self) -> str:
        """Path to bin directory in virtual environment."""
        return os.path.join(self.venv_path, "bin")
    
    @property
    def git_repo_path(self) -> str:
        """Path to git repository (one level above working directory)."""
        return os.path.join(os.path.dirname(self.working_dir), self.git_repo_name)
    
    @property
    def local_config_json(self) -> str:
        """Where the reviewer repo's config JSON lands once cloned locally."""
        return os.path.join(self.config_repo_path, self.config_json_path)

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
    def queue_path(self) -> str:
        """Shared work queue. Must be on scratch, not a home directory."""
        return os.path.join(self.data_dir, "queue")

    @property
    def prepared_path(self) -> str:
        """Scratch for decoded audio, removed as soon as the GPU is done."""
        return os.path.join(self.data_dir, "prepared")

    # --- Script Paths ---

    @property
    def transcribe_script_path(self) -> str:
        """Path to the transcription worker."""
        return os.path.join(self.working_dir, "scripts", "transcribe.py")

    @property
    def slurm_job_path(self) -> str:
        """Path to the SLURM job template."""
        return os.path.join(self.working_dir, "config", "run.slurm")

    @property
    def requirements_path(self) -> str:
        """Path to requirements.txt."""
        return os.path.join(self.working_dir, "requirements.txt")

    # --- Credentials ---

    def get_smb_password(self) -> str:
        """Get SMB password from config, environment, or prompt."""
        import getpass

        if self.smb_password:
            return self.smb_password

        env_password = os.environ.get("SMB_PASSWORD")
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

        env_token = os.environ.get("GIT_TOKEN")
        if env_token:
            return env_token

        raise ValueError(
            "GitHub token not set. Set GIT_TOKEN environment variable or update config."
        )


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
