"""The `run_pipeline.py` entry point and the orchestrator it drives.

These exist because a signature mismatch between the two shipped once and only
showed up on the cluster, at the moment someone ran the program. Nothing here
touches Slurm, the network or a GPU - it checks that the pieces fit together.
"""

import importlib.util
import re
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
            or flag in {"from_json", "no_diarize", "max_files", "input"}  # mapped by hand
        ), f"--{flag.replace('_', '-')} goes nowhere"


def test_from_json_switches_the_mode(entry, config):
    args = entry.build_parser().parse_args(["--from-json"])
    assert args.from_json is True


@pytest.mark.parametrize(
    "argv,expected",
    [([], "smb"), (["--from-json"], "json"), (["--source", "local"], "local"),
     (["--source", "json"], "json"), (["--source", "smb"], "smb")],
)
def test_the_source_can_be_chosen_from_the_command_line(entry, config, monkeypatch, argv, expected):
    """SETUP.md and TROUBLESHOOTING.md both tell users to run --source local;
    it has to be reachable without bypassing this entry point."""
    from src.pipeline import TranscriptionPipeline

    seen = {}
    monkeypatch.setattr(
        TranscriptionPipeline, "run",
        lambda self: seen.setdefault("source", self.config.resolved_source) and 0 or 0,
    )
    monkeypatch.setattr(TranscriptionPipeline, "_prepare_config_repo", lambda self: True)
    entry.main(argv)

    assert seen["source"] == expected


def test_local_source_tells_the_job_where_to_look(config, tmp_path):
    from src.pipeline import TranscriptionPipeline

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    config.working_dir = str(tmp_path)
    config.source = "local"
    config.input_dir = str(tmp_path / "my_audio")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "run.slurm").write_text(template, encoding="utf-8")

    pipeline = TranscriptionPipeline()
    written = {}
    pipeline.command_runner.submit_slurm_job = lambda p: written.setdefault(
        "text", Path(p).read_text(encoding="utf-8")
    ) and "1"
    pipeline._submit_slurm_job()

    assert "--source local" in written["text"]
    assert str(tmp_path / "my_audio") in written["text"]


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


# ------------------------------------------------------------- from_json mode


REVIEWER_CONFIG = [
    {"name": "02_00113",
     "audio": "https://kaltura/02_00113-a_01-medium.mp4/index.m3u8",
     "url": "https://raw.githubusercontent.com/org/repo/main/json/02_00113.json",
     "vtt": "https://raw.githubusercontent.com/org/repo/main/vtts/02_00113.vtt"},
    {"name": "02_00114",
     "audio": "https://kaltura/02_00114-a_01-medium.mp4/index.m3u8",
     "url": "", "vtt": ""},
]


class FakeConfigRepo:
    """Stands in for the reviewer repository."""

    def __init__(self, entries=None, ok=True):
        self.entries = REVIEWER_CONFIG if entries is None else entries
        self.ok = ok

    def setup_repository(self):
        return self.ok

    def read_config_to_process(self):
        return list(self.entries)


def prepared(config, tmp_path, monkeypatch, repo=None):
    """Run _prepare_config_repo with the reviewer repo stubbed out."""
    import src.pipeline as pipeline_module
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    config.from_json = True
    config.git_token = "fake-token"
    monkeypatch.setattr(
        pipeline_module, "ConfigRepoManager", lambda **kw: repo or FakeConfigRepo()
    )

    pipeline = TranscriptionPipeline()
    return pipeline, pipeline._prepare_config_repo()


def test_from_json_writes_a_work_list_the_job_can_read(config, tmp_path, monkeypatch):
    import json

    pipeline, ok = prepared(config, tmp_path, monkeypatch)

    assert ok is True
    written = json.loads(Path(config.work_list_path).read_text(encoding="utf-8"))
    assert [e["name"] for e in written] == ["02_00113", "02_00114"]
    assert all(e["audio"].startswith("https://kaltura") for e in written)


def test_the_work_list_feeds_straight_into_the_queue(config, tmp_path, monkeypatch):
    """End to end: reviewer config -> work list -> queue references."""
    import json

    from src.asr.sources import URL, enumerate_json

    prepared(config, tmp_path, monkeypatch)
    entries = json.loads(Path(config.work_list_path).read_text(encoding="utf-8"))
    queued = enumerate_json(entries)

    assert len(queued) == 2
    assert all(q["kind"] == URL for q in queued)
    assert queued[0]["id"] == "02_00113"
    # the audio, not the transcript link
    assert queued[0]["url"].startswith("https://kaltura")


def test_max_files_limits_what_is_queued(config, tmp_path, monkeypatch):
    import json

    config.max_files = 1
    prepared(config, tmp_path, monkeypatch)

    assert len(json.loads(Path(config.work_list_path).read_text(encoding="utf-8"))) == 1


def test_output_names_are_predictable_from_the_config(config, tmp_path, monkeypatch):
    """No rename step: the transcript is named after the entry's `name`."""
    from src.asr.sources import enumerate_json, slug

    pipeline, _ = prepared(config, tmp_path, monkeypatch)

    for entry, queued in zip(pipeline._processed_entries, enumerate_json(REVIEWER_CONFIG)):
        assert entry["_folder_name"] == slug(entry["name"]) == queued["id"]


def test_a_failed_clone_stops_the_run(config, tmp_path, monkeypatch):
    pipeline, ok = prepared(config, tmp_path, monkeypatch, repo=FakeConfigRepo(ok=False))
    assert ok is False


def test_an_empty_config_stops_the_run(config, tmp_path, monkeypatch):
    pipeline, ok = prepared(config, tmp_path, monkeypatch, repo=FakeConfigRepo(entries=[]))
    assert ok is False


def test_from_json_run_aborts_before_submitting_when_there_is_no_work(
    config, tmp_path, monkeypatch
):
    import src.pipeline as pipeline_module
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    config.from_json = True
    config.git_token = "fake-token"
    monkeypatch.setattr(
        pipeline_module, "ConfigRepoManager", lambda **kw: FakeConfigRepo(entries=[])
    )
    monkeypatch.setattr(TranscriptionPipeline, "_load_modules", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_setup_python_environment", lambda self: None)

    submitted = []
    monkeypatch.setattr(
        TranscriptionPipeline, "_submit_slurm_job", lambda self: submitted.append(1) or "1"
    )

    assert TranscriptionPipeline().run() == 1
    assert submitted == [], "submitted a job with nothing to do"


def test_the_slurm_job_is_pointed_at_the_work_list(config, tmp_path, monkeypatch):
    import src.pipeline as pipeline_module
    from src.pipeline import TranscriptionPipeline

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    config.working_dir = str(tmp_path)
    config.from_json = True
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "run.slurm").write_text(template, encoding="utf-8")

    pipeline = TranscriptionPipeline()
    written = {}
    pipeline.command_runner.submit_slurm_job = lambda path: written.setdefault(
        "text", Path(path).read_text(encoding="utf-8")
    ) and "1"
    pipeline._submit_slurm_job()

    assert "--source json" in written["text"]
    assert config.work_list_path in written["text"]


# ------------------------------------------- config settings must reach the job


@pytest.mark.parametrize(
    "setting,value,expected",
    [
        ("words_device", "cuda:0", "--words-device cuda:0"),
        ("turns_device", "cuda:0", "--turns-device cuda:0"),
        ("max_attempts", 7, "--max-attempts 7"),
        ("max_line_width", 36, "--max-line-width 36"),
        ("max_line_count", 3, "--max-line-count 3"),
        ("lease_seconds", 9999, "--lease 9999"),
        ("deadline_minutes", 120, "--deadline-minutes 120"),
        ("asr_model", "nvidia/other-asr", "nvidia/other-asr"),
        ("diarization_model", "nvidia/other-diar", "nvidia/other-diar"),
    ],
)
def test_changing_a_setting_actually_reaches_the_job(
    config, tmp_path, setting, value, expected
):
    """A documented setting that never reaches the worker is a silent lie."""
    from src.pipeline import TranscriptionPipeline

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    config.working_dir = str(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "run.slurm").write_text(template, encoding="utf-8")
    setattr(config, setting, value)

    pipeline = TranscriptionPipeline()
    written = {}
    pipeline.command_runner.submit_slurm_job = lambda path: written.setdefault(
        "text", Path(path).read_text(encoding="utf-8")
    ) and "1"
    pipeline._submit_slurm_job()

    assert expected in written["text"], f"config.{setting} never reaches the worker"


def test_every_worker_flag_in_the_template_is_one_the_worker_accepts():
    """A typo in run.slurm would fail only at runtime, on the cluster."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "transcribe", REPO / "scripts" / "transcribe.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["transcribe"] = module
    spec.loader.exec_module(module)

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    # only the lines invoking the worker, not the #SBATCH directives
    invocation = template.split("transcribe.py")[1:]
    used = set()
    for chunk in invocation:
        for line in chunk.splitlines():
            if line.strip().startswith("#"):
                break
            used.update(re.findall(r"--[a-z][a-z-]+", line))

    parser = module.build_parser()
    accepted = set()
    for action in parser._subparsers._group_actions[0].choices.values():
        for opt in action._actions:
            accepted.update(opt.option_strings)

    unknown = used - accepted
    assert not unknown, f"run.slurm passes flags the worker does not accept: {sorted(unknown)}"


# ---------------------------------------------------- local settings overlay


def write_local_settings(tmp_path, body):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "local_settings.py").write_text(body, encoding="utf-8")


def test_local_settings_override_the_defaults(config, tmp_path):
    from src.config import apply_local_settings

    config.working_dir = str(tmp_path)
    write_local_settings(tmp_path, 'git_owner = "my-org"\nmax_files = 5\n')

    applied = apply_local_settings(config)

    assert sorted(applied) == ["git_owner", "max_files"]
    assert config.git_owner == "my-org"
    assert config.max_files == 5


def test_secrets_can_live_in_local_settings(config, tmp_path):
    """The repo is public, so this is the only safe place for a token."""
    from src.config import apply_local_settings

    config.working_dir = str(tmp_path)
    write_local_settings(tmp_path, 'git_token = "ghp_secret"\nsmb_password = "pw"\n')
    apply_local_settings(config)

    assert config.get_git_token() == "ghp_secret"
    assert config.get_smb_password() == "pw"


def test_no_local_settings_file_is_fine(config, tmp_path):
    from src.config import apply_local_settings

    config.working_dir = str(tmp_path)
    assert apply_local_settings(config) == []


def test_an_unknown_option_warns_instead_of_crashing(config, tmp_path, capsys):
    from src.config import apply_local_settings

    config.working_dir = str(tmp_path)
    write_local_settings(tmp_path, 'git_owner = "ok"\nwhisper_model = "large-v3"\n')

    applied = apply_local_settings(config)

    assert applied == ["git_owner"]
    assert "whisper_model" in capsys.readouterr().out


def test_settings_left_out_keep_their_defaults(config, tmp_path):
    from src.config import apply_local_settings

    config.working_dir = str(tmp_path)
    default_lease = config.lease_seconds
    write_local_settings(tmp_path, 'git_owner = "my-org"\n')
    apply_local_settings(config)

    assert config.lease_seconds == default_lease


def test_the_example_file_only_names_real_settings(config):
    """A typo in the example would silently do nothing for whoever copies it."""
    import re

    example = (REPO / "config" / "local_settings.example.py").read_text(encoding="utf-8")
    named = set(re.findall(r"^#?\s*([a-z_]+) = ", example, re.M))

    unknown = {n for n in named if not hasattr(config, n)}
    assert not unknown, f"example names settings that do not exist: {sorted(unknown)}"


def test_local_settings_is_gitignored():
    """Committing this file is exactly what it exists to prevent."""
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8")
    assert "config/local_settings.py" in ignored


# ------------------------------------------ the durable record of finished work


class FakeUploader:
    """Stands in for the transcripts repository."""

    def __init__(self, ok=True):
        self.ok = ok

    def setup_repository(self):
        return self.ok


def with_transcripts_repo(config, tmp_path, monkeypatch, done_ids, ok=True):
    import src.pipeline as pipeline_module
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path / "repo")
    config.git_token = "t"
    repo_json = Path(config.git_repo_path) / "json"
    repo_json.mkdir(parents=True, exist_ok=True)
    for name in done_ids:
        (repo_json / f"{name}.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "GitUploader", lambda **kw: FakeUploader(ok))
    return TranscriptionPipeline()


def test_interviews_already_in_the_transcripts_repo_are_recorded(
    config, tmp_path, monkeypatch
):
    """/scratch is purged; the repo is the record of what is really done."""
    import json

    pipeline = with_transcripts_repo(config, tmp_path, monkeypatch, ["iv_001", "iv_002"])

    assert pipeline._collect_completed() == 2
    written = json.loads(Path(config.completed_list_path).read_text(encoding="utf-8"))
    assert written == ["iv_001", "iv_002"]


def test_an_unreachable_repo_does_not_stop_the_run(config, tmp_path, monkeypatch):
    """Redoing some work beats refusing to run at all."""
    import json

    pipeline = with_transcripts_repo(config, tmp_path, monkeypatch, ["iv_001"], ok=False)

    assert pipeline._collect_completed() == 0
    assert json.loads(Path(config.completed_list_path).read_text(encoding="utf-8")) == []


def test_the_check_can_be_turned_off(config, tmp_path, monkeypatch):
    pipeline = with_transcripts_repo(config, tmp_path, monkeypatch, ["iv_001"])
    config.check_transcripts_repo = False

    assert pipeline._collect_completed() == 0


def test_the_job_is_given_the_completed_list(config, tmp_path):
    from src.pipeline import TranscriptionPipeline

    template = (REPO / "config" / "run.slurm").read_text(encoding="utf-8")
    config.working_dir = str(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "run.slurm").write_text(template, encoding="utf-8")

    pipeline = TranscriptionPipeline()
    written = {}
    pipeline.command_runner.submit_slurm_job = lambda p: written.setdefault(
        "text", Path(p).read_text(encoding="utf-8")
    ) and "1"
    pipeline._submit_slurm_job()

    assert "--skip-list" in written["text"]
    assert config.completed_list_path in written["text"]
