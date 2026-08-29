"""Reading the work list from the reviewer repository."""

import json

import pytest

from src.git.config_repo import ConfigRepoManager


def make_manager(tmp_path, config_json_path, output_config_path):
    return ConfigRepoManager(
        repo_folder=str(tmp_path),
        owner="org",
        repo_name="reviewer",
        username="user",
        token="token",
        config_json_path=config_json_path,
        output_config_path=output_config_path,
    )


def write(tmp_path, relative, entries):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


ENTRIES = [
    {"name": "iv_001", "audio": "https://h/iv_001.mp4/index.m3u8"},
    {"name": "iv_002", "audio": "https://h/iv_002.mp4/index.m3u8"},
]


def test_it_reads_the_configured_path_not_a_hardcoded_one(tmp_path, capsys):
    """A renamed input file must actually be read - and named in the log."""
    write(tmp_path, "public/config.json", ENTRIES)
    manager = make_manager(tmp_path, "public/config.json", "public/done.json")

    assert len(manager.read_config_to_process()) == 2
    assert "public/config.json" in capsys.readouterr().out


def test_already_processed_entries_are_skipped(tmp_path):
    write(tmp_path, "public/to-process.json", ENTRIES)
    write(tmp_path, "public/config.json", [{"name": "iv_001"}])
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    remaining = manager.read_config_to_process()

    assert [e["name"] for e in remaining] == ["iv_002"]


def test_using_one_file_for_input_and_output_warns(tmp_path, capsys):
    """It works once, then silently finds nothing - so say so up front."""
    write(tmp_path, "public/config.json", ENTRIES)
    manager = make_manager(tmp_path, "public/config.json", "public/config.json")

    manager.read_config_to_process()

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "find nothing to transcribe" in out


def test_one_file_for_input_and_output_finds_nothing_at_all(tmp_path):
    """The failure the warning is about, and it bites on the very first run.

    Every entry in the output file counts as done. When that file is also the
    input, every entry is skipped and the run reports "all done" having
    transcribed nothing.
    """
    write(tmp_path, "public/config.json", ENTRIES)
    manager = make_manager(tmp_path, "public/config.json", "public/config.json")

    assert manager.read_config_to_process() == []


def test_two_files_is_the_working_arrangement(tmp_path):
    """The same entries, split across an input and an output file, work."""
    write(tmp_path, "public/config-to-process.json", ENTRIES)
    manager = make_manager(
        tmp_path, "public/config-to-process.json", "public/config.json"
    )

    assert len(manager.read_config_to_process()) == 2


def test_a_missing_input_file_is_reported(tmp_path, capsys):
    manager = make_manager(tmp_path, "public/missing.json", "public/config.json")

    assert manager.read_config_to_process() == []
    assert "not found" in capsys.readouterr().out
