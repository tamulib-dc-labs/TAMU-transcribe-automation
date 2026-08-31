"""What decides whether an interview gets transcribed.

The intended flow, end to end:

    1. read every record from config_json_path
    2. compare against the transcripts repo (edge-grant-json-and-vtts)
    3. transcribe whatever is not there yet
    4. write those transcripts' json/vtt links into output_config_path

Step 2 is the only comparison. output_config_path is written in step 4 and is
never read to decide anything, because it is a record of *links*, not of work
done - a run that uploaded transcripts but died before its last step would
otherwise make the pipeline forget them.

These tests drive the real code: the real ConfigRepoManager reading real files,
the real enumerate_json, the real queue. Only git and the network are stubbed.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

ENTRIES = [
    {"name": "02_00113", "audio": "https://media/02_00113.mp4/index.m3u8"},
    {"name": "02_00114", "audio": "https://media/02_00114.mp4/index.m3u8"},
    {"name": "02_00115", "audio": "https://media/02_00115.mp4/index.m3u8"},
]

#: What config.json would hold if every interview had already been published.
#: Present in every test below, to prove it changes nothing.
ALL_PUBLISHED = [
    {"name": e["name"], "audio": e["audio"],
     "url": f"https://x/json/{e['name']}.json",
     "vtt": f"https://x/vtts/{e['name']}.vtt"}
    for e in ENTRIES
]


@pytest.fixture
def config():
    from src.config import get_config, reset_config

    reset_config()
    yield get_config()
    reset_config()


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_prepare(config, tmp_path, monkeypatch, *, work_list, transcribed,
                published=ALL_PUBLISHED):
    """Run the job's listing steps for real, with git and the network stubbed.

    work_list    what config-to-process.json contains
    published    what config.json already contains - must not affect anything
    transcribed  interview ids already in the transcripts repository

    Returns the ids actually queued for transcription.
    """
    import src.git.config_repo as config_repo_module
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    config.from_json = True
    config.git_token = "fake"

    # Lay the reviewer repo out as if it had just been cloned.
    reviewer = Path(config.config_repo_path)
    for relative, content in ((config.config_json_path, work_list),
                              (config.output_config_path, published)):
        path = reviewer / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")

    # Everything except the clone/pull is the real ConfigRepoManager.
    monkeypatch.setattr(
        config_repo_module.ConfigRepoManager, "setup_repository", lambda self: True
    )
    monkeypatch.setattr(
        TranscriptionPipeline, "_list_transcribed", lambda self: sorted(transcribed)
    )
    # Stop after the queue is built - the GPU work and the upload are not what
    # these tests are about.
    monkeypatch.setattr(TranscriptionPipeline, "_transcribe", lambda self: 0)
    monkeypatch.setattr(TranscriptionPipeline, "_upload_to_github", lambda self: None)
    monkeypatch.setattr(TranscriptionPipeline, "_update_config_repo", lambda self: None)

    fill = load("transcribe", "transcribe.py")

    # _fill_queue shells out to `transcribe.py fill`; run it in-process so the
    # real enumerate and skip-list logic is exercised, without a subprocess.
    def run_in_process(command, *a, **k):
        args = fill.build_parser().parse_args([str(c) for c in command[2:]])
        return type("Done", (), {"returncode": fill._fill(args)})()

    monkeypatch.setattr(subprocess, "run", run_in_process)
    assert TranscriptionPipeline().execute() == 0

    from src.asr.workqueue import PENDING, FileWorkQueue

    return sorted(task.id for task in FileWorkQueue(config.queue_path).tasks(PENDING))


# ------------------------------------------------------------- the main flow


def test_the_transcripts_repo_decides_what_is_left_to_do(config, tmp_path, monkeypatch):
    """Step 2: queue everything in the work list that the repo does not have."""
    queued = run_prepare(
        config, tmp_path, monkeypatch, work_list=ENTRIES, transcribed=["02_00114"]
    )

    assert queued == ["02_00113", "02_00115"]


def test_config_json_does_not_decide_anything(config, tmp_path, monkeypatch):
    """The point of the change.

    config.json lists all three with links. If it were still consulted, nothing
    would be queued. Only the transcripts repo counts.
    """
    queued = run_prepare(config, tmp_path, monkeypatch, work_list=ENTRIES,
                         transcribed=[])

    assert queued == ["02_00113", "02_00114", "02_00115"]


def test_one_file_for_input_and_output_still_queues_the_work(config, tmp_path,
                                                             monkeypatch):
    """Reading and writing the same file is no longer a trap.

    It used to find nothing at all, on the very first run.
    """
    config.config_json_path = "public/config.json"
    config.output_config_path = "public/config.json"

    queued = run_prepare(config, tmp_path, monkeypatch, work_list=ALL_PUBLISHED,
                         transcribed=[], published=ALL_PUBLISHED)

    assert queued == ["02_00113", "02_00114", "02_00115"]


def test_nothing_is_queued_when_the_repo_has_it_all(config, tmp_path, monkeypatch):
    queued = run_prepare(
        config, tmp_path, monkeypatch, work_list=ENTRIES,
        transcribed=["02_00113", "02_00114", "02_00115"],
    )

    assert queued == []


def test_a_transcript_already_on_scratch_also_counts(config, tmp_path, monkeypatch):
    """The local folder is checked too, so a re-run in one session is cheap."""
    config.working_dir = str(tmp_path)
    done = Path(config.oral_output_path) / "json"
    done.mkdir(parents=True)
    (done / "02_00114.json").write_text("{}", encoding="utf-8")

    queued = run_prepare(config, tmp_path, monkeypatch, work_list=ENTRIES,
                         transcribed=[])

    assert queued == ["02_00113", "02_00115"]


def test_the_repo_ids_and_the_queue_ids_are_the_same_strings(config, tmp_path,
                                                             monkeypatch):
    """The comparison is string equality, so both sides must agree.

    A name needing escaping is where they would drift apart: the queue id and
    the output filename are both slug(name), so slug(name) is what the repo
    listing gives back.
    """
    from src.asr.sources import slug

    awkward = [{"name": "02/00116 a", "audio": "https://media/x.mp3"}]
    expected = slug("02/00116 a")

    assert run_prepare(config, tmp_path / "first", monkeypatch,
                       work_list=awkward, transcribed=[]) == [expected]

    # Now the repo holds exactly that file - it has to be recognised.
    assert run_prepare(config, tmp_path / "second", monkeypatch,
                       work_list=awkward, transcribed=[expected]) == []


# ------------------------------------------------- step 4: what gets written


def test_only_transcribed_interviews_get_links_written_back(config, tmp_path,
                                                            monkeypatch):
    """Step 4 records links; it must not invent empty ones.

    The work list names the whole collection now, so writing a row per entry
    would put blank url/vtt fields against interviews never transcribed.
    """
    from src.pipeline import TranscriptionPipeline

    config.working_dir = str(tmp_path)
    config.from_json = True
    config.git_token = "fake"

    pipeline = TranscriptionPipeline()
    pipeline._processed_entries = [
        {"name": e["name"], "audio": e["audio"], "_folder_name": e["name"]}
        for e in ENTRIES
    ]

    class OnlyOneWasTranscribed:
        def get_uploaded_file_urls(self, names):
            return {"02_00114": {"json_url": "https://x/json/02_00114.json",
                                 "vtt_url": "https://x/vtts/02_00114.vtt"}}

    written = {}

    class Recorder:
        def update_config_json(self, entries):
            written["entries"] = entries
            return True

        def commit_and_push(self, message=None):
            return True

    pipeline._uploader = OnlyOneWasTranscribed()
    pipeline._config_repo_manager = Recorder()
    pipeline._update_config_repo()

    assert [e["name"] for e in written["entries"]] == ["02_00114"]
    assert written["entries"][0]["url"].endswith("02_00114.json")
    assert written["entries"][0]["vtt"].endswith("02_00114.vtt")
