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
    smb_password: str = ""  # put this in config/local_settings.py, not here
    
    # --- Google Sheets Settings ---
    sheet_url: str = "https://docs.google.com/spreadsheets/d/16cHa57n7rJmS744nMH2dY2H4IKLP5fMeHJ0iY8w85EM/edit?usp=sharing"
    max_folders: int = 20
    
    # --- GitHub Settings ---
    git_owner: str = "tamulib-dc-labs"
    git_repo_name: str = "edge-grant-json-and-vtts"
    git_username: str = "JvkChaitanya"
    git_token: str = ""  # put this in config/local_settings.py, not here
    
    #: Cap how many interviews a run queues. 0 = the whole collection.
    max_files: int = 0

    # --- Where the audio comes from ---
    #: "auto" follows from_json below, for compatibility with v2.0 settings.
    #: Set it explicitly to pick: "smb" (the file share), "json" (the reviewer
    #: repo's list), or "local" (audio already staged on scratch).
    source: str = "auto"
    #: Folder to read, when source is "local". Defaults to oral_input_path.
    input_dir: str = ""

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
        """The reviewer repo's config file, in the cloned repo."""
        return os.path.join(self.config_repo_path, self.config_json_path)

    @property
    def resolved_source(self) -> str:
        """Which enumerator the job should use: smb, json or local."""
        if self.source and self.source != "auto":
            return self.source
        return "json" if self.from_json else "smb"

    @property
    def resolved_input_dir(self) -> str:
        """Folder for source=local."""
        return self.input_dir or self.oral_input_path

    @property
    def work_list_path(self) -> str:
        """The interviews this run should do, written out for the SLURM job.

        In from_json mode the list comes from a private repo, so it is read on
        the login node and written here as a plain file the job can read.
        """
        return os.path.join(self.data_dir, "work_list.json")

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


#: Optional file holding your own settings, including secrets. It is
#: gitignored, so anything in it stays on your machine. See
#: config/local_settings.example.py.
LOCAL_SETTINGS_FILE = "local_settings.py"


def apply_local_settings(config: "PipelineConfig") -> list:
    """Overlay config/local_settings.py onto the defaults, if it exists.

    This repository is public, so a token written into config.py would be
    pushed to GitHub and revoked within minutes. local_settings.py is
    gitignored and is the right place for anything private.

    Returns the names of the settings that were overridden.
    """
    path = os.path.join(config.working_dir, "config", LOCAL_SETTINGS_FILE)
    if not os.path.exists(path):
        return []

    namespace: dict = {}
    with open(path, "r", encoding="utf-8") as handle:
        exec(compile(handle.read(), path, "exec"), namespace)  # noqa: S102

    applied = []
    for key, value in namespace.items():
        if key.startswith("_") or callable(value) or isinstance(value, type):
            continue
        if not hasattr(config, key):
            print(f"WARNING: {LOCAL_SETTINGS_FILE} sets unknown option {key!r}; ignoring")
            continue
        setattr(config, key, value)
        applied.append(key)
    return applied


# Singleton instance
_config_instance = None


def get_config() -> PipelineConfig:
    """Get or create the singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = PipelineConfig()
        applied = apply_local_settings(_config_instance)
        if applied:
            # Never print the values - some of them are secrets.
            print(f"Loaded {len(applied)} setting(s) from config/{LOCAL_SETTINGS_FILE}: "
                  f"{', '.join(sorted(applied))}")
    return _config_instance


def reset_config():
    """Reset configuration instance (useful for testing)."""
    global _config_instance
    _config_instance = None
