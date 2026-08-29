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


def read(tmp_path, relative):
    return json.loads((tmp_path / relative).read_text(encoding="utf-8"))


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


def test_the_output_file_does_not_filter_the_work_list(tmp_path):
    """config.json is written after a run so the reviewer app has links.

    It is not a record of what has been transcribed - the transcripts
    repository is - so it must not decide what gets queued.
    """
    write(tmp_path, "public/to-process.json", ENTRIES)
    write(tmp_path, "public/config.json", [{"name": "iv_001"}])
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    remaining = manager.read_config_to_process()

    assert [e["name"] for e in remaining] == ["iv_001", "iv_002"]


def test_one_file_for_input_and_output_still_returns_the_work(tmp_path):
    """Reading and writing one file is allowed; it is just replaced after."""
    write(tmp_path, "public/config.json", ENTRIES)
    manager = make_manager(tmp_path, "public/config.json", "public/config.json")

    assert len(manager.read_config_to_process()) == 2


def test_one_file_for_input_and_output_says_it_will_be_replaced(tmp_path, capsys):
    write(tmp_path, "public/config.json", ENTRIES)
    manager = make_manager(tmp_path, "public/config.json", "public/config.json")
    manager.read_config_to_process()

    assert "replace it with the transcript links" in capsys.readouterr().out


def test_two_files_also_work(tmp_path):
    write(tmp_path, "public/config-to-process.json", ENTRIES)
    manager = make_manager(
        tmp_path, "public/config-to-process.json", "public/config.json"
    )

    assert len(manager.read_config_to_process()) == 2


def test_a_malformed_work_list_is_reported(tmp_path, capsys):
    path = tmp_path / "public" / "to-process.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    assert manager.read_config_to_process() == []
    assert "Could not parse" in capsys.readouterr().out


def test_a_missing_input_file_is_reported(tmp_path, capsys):
    manager = make_manager(tmp_path, "public/missing.json", "public/config.json")

    assert manager.read_config_to_process() == []
    assert "not found" in capsys.readouterr().out


# ------------------------------------------- writing the reviewer's config


def test_writing_adds_new_entries(tmp_path):
    write(tmp_path, "public/config.json", [{"name": "iv_001", "url": "old"}])
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    assert manager.update_config_json([{"name": "iv_002", "url": "new"}])

    written = read(tmp_path, "public/config.json")
    assert [e["name"] for e in written] == ["iv_001", "iv_002"]


def test_re_transcribing_replaces_the_row_instead_of_duplicating_it(tmp_path):
    """config.json is written every run now, so appending would pile up copies."""
    write(tmp_path, "public/config.json", [
        {"name": "iv_001", "url": "first-pass"},
        {"name": "iv_002", "url": "keep-me"},
    ])
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    manager.update_config_json([{"name": "iv_001", "url": "second-pass"}])

    written = read(tmp_path, "public/config.json")
    assert [e["name"] for e in written] == ["iv_001", "iv_002"]
    assert written[0]["url"] == "second-pass"


def test_the_existing_order_survives_an_update(tmp_path):
    """The reviewer app lists these in file order; it must not reshuffle."""
    write(tmp_path, "public/config.json", [{"name": n} for n in "abcd"])
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    manager.update_config_json([{"name": "c", "url": "u"}, {"name": "e"}])

    assert [e["name"] for e in read(tmp_path, "public/config.json")] == list("abcde")


def test_writing_creates_the_file_when_it_does_not_exist_yet(tmp_path):
    (tmp_path / "public").mkdir(parents=True)
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    assert manager.update_config_json([{"name": "iv_001", "url": "u"}])
    assert read(tmp_path, "public/config.json") == [{"name": "iv_001", "url": "u"}]


def test_a_corrupt_output_file_does_not_lose_the_new_entries(tmp_path):
    path = tmp_path / "public" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    manager = make_manager(tmp_path, "public/to-process.json", "public/config.json")

    assert manager.update_config_json([{"name": "iv_001"}])
    assert read(tmp_path, "public/config.json") == [{"name": "iv_001"}]
