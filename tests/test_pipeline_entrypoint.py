"""The `run_pipeline.py` entry point and the orchestrator it drives.

These exist because a signature mismatch between the two shipped once and only
showed up on the cluster, at the moment someone ran the program. Nothing here
touches Slurm, the network or a GPU - it checks that the pieces fit together.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO / "scripts" / "run_pipeline.py"


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location("run_pipeline", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config():
    from src.config import get_config, reset_config

    reset_config()
    yield get_config()
    reset_config()


# ------------------------------------------------------- the wiring itself


def test_the_entry_point_can_construct_the_pipeline(entry, config):
    """The bug this file exists for: arguments the constructor never took."""
    from src.pipeline import TranscriptionPipeline

    args = entry.build_parser().parse_args([])
    pipeline = TranscriptionPipeline(skip_upload=args.skip_upload)

    assert pipeline.skip_upload is False


def test_every_cli_flag_maps_to_something_real(entry, config):
    """Each flag must reach either the config or the constructor."""
    import inspect

    from src.pipeline import TranscriptionPipeline

    flags = set(vars(entry.build_parser().parse_args([])))
    constructor = set(inspect.signature(TranscriptionPipeline.__init__).parameters)
    settings = set(vars(config))

    for flag in flags:
        assert (
            flag in constructor
            or flag in settings
            or flag in {"from_json", "no_diarize", "max_files"}  # mapped by hand
        ), f"--{flag.replace('_', '-')} goes nowhere"


def test_from_json_switches_the_mode(entry, config):
    entry.main.__globals__  # noqa: B018 - module is importable
    args = entry.build_parser().parse_args(["--from-json"])
    assert args.from_json is True


def test_no_diarize_and_max_files_reach_the_config(entry, config, monkeypatch):
    """main() applies these to the singleton before the pipeline reads it."""
    from src.pipeline import TranscriptionPipeline

    captured = {}

    def fake_run(self):
        captured["diarize"] = self.config.diarize
        captured["max_files"] = self.config.max_files
        captured["skip_upload"] = self.skip_upload
        return 0

    monkeypatch.setattr(TranscriptionPipeline, "run", fake_run)
    assert entry.main(["--no-diarize", "--max-files", "3", "--skip-upload"]) == 0

    assert captured == {"diarize": False, "max_files": 3, "skip_upload": True}


def test_run_returns_an_exit_code(entry, config, monkeypatch):
    """sys.exit(main()) needs an int, not None."""
    from src.pipeline import TranscriptionPipeline

    monkeypatch.setattr(TranscriptionPipeline, "_load_modules", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_setup_python_environment", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_prepare_directories", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_submit_slurm_job", lambda self: "")

    # No job submitted -> non-zero, so a failed submission is visible to the shell.
    assert TranscriptionPipeline().run() == 1


def test_skip_upload_stops_the_push(config, monkeypatch, tmp_path):
    from src.pipeline import TranscriptionPipeline

    uploaded = []
    monkeypatch.setattr(TranscriptionPipeline, "_load_modules", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_setup_python_environment", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_prepare_directories", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_submit_slurm_job", lambda self: "12345")
    monkeypatch.setattr(TranscriptionPipeline, "_monitor_slurm_job", lambda self, j: None)
    monkeypatch.setattr(
        TranscriptionPipeline, "_upload_to_github", lambda self: uploaded.append(True)
    )

    assert TranscriptionPipeline(skip_upload=True).run() == 0
    assert uploaded == []

    assert TranscriptionPipeline(skip_upload=False).run() == 0
    assert uploaded == [True]


# --------------------------------------------- directories must not be wiped


def test_preparing_directories_does_not_delete_transcripts(config, tmp_path):
    """The output folder is the resume record - clearing it would re-transcribe
    the whole collection on every run."""
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    output = Path(config.oral_output_path) / "json"
    output.mkdir(parents=True)
    existing = output / "iv_001.json"
    existing.write_text("{}", encoding="utf-8")

    TranscriptionPipeline()._prepare_directories()

    assert existing.exists(), "an already-transcribed interview was deleted"


def test_preparing_directories_does_not_delete_staged_audio(config, tmp_path):
    """In --source local mode the input folder holds the audio itself."""
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    audio = Path(config.oral_input_path)
    audio.mkdir(parents=True)
    staged = audio / "iv_001.mp3"
    staged.write_bytes(b"audio")

    TranscriptionPipeline()._prepare_directories()

    assert staged.exists(), "staged source audio was deleted"


def test_preparing_directories_creates_what_is_missing(config, tmp_path):
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    TranscriptionPipeline()._prepare_directories()

    assert Path(config.oral_input_path).is_dir()
    assert Path(config.oral_output_path).is_dir()
    assert Path(config.queue_path).is_dir()


# ------------------------------------------------------ slurm substitution


def test_every_slurm_placeholder_is_filled(config, tmp_path):
    """A leftover {{PLACEHOLDER}} would reach the shell as literal text."""
    import re

    from src.pipeline import TranscriptionPipeline

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    placeholders = set(re.findall(r"\{\{[A-Z_]+\}\}", template))

    config.working_dir = str(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "run.slurm").write_text(template, encoding="utf-8")

    pipeline = TranscriptionPipeline()
    written = {}
    pipeline.command_runner.submit_slurm_job = lambda path: written.setdefault(
        "text", Path(path).read_text(encoding="utf-8")
    ) and "1"

    pipeline._submit_slurm_job()

    leftover = set(re.findall(r"\{\{[A-Z_]+\}\}", written["text"]))
    assert not leftover, f"unfilled placeholders: {sorted(leftover)} of {sorted(placeholders)}"
