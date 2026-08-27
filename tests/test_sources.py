"""Where audio comes from, and how a worker fetches its own copy.

The queue holds references, not files. These tests cover the reference shapes
and the fetch that turns one into a local path - the network calls themselves
are stubbed.
"""

import json

import pytest

from src.asr import sources
from src.asr.sources import LOCAL, SMB, URL


# ------------------------------------------------------------------ local


def test_enumerate_local_finds_audio(tmp_path):
    for name in ("iv_001.mp3", "iv_002.wav", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")

    found = sources.enumerate_local(tmp_path)

    assert [f["id"] for f in found] == ["iv_001", "iv_002"]
    assert all(f["kind"] == LOCAL for f in found)
    assert all(f["path"].endswith((".mp3", ".wav")) for f in found)


def test_a_local_reference_is_not_a_temporary_copy(tmp_path):
    """The collection itself must survive; only fetched copies are deleted."""
    assert sources.is_temporary({"kind": LOCAL, "path": "x"}) is False
    assert sources.is_temporary({"kind": URL, "url": "x"}) is True
    assert sources.is_temporary({"kind": SMB, "remote_path": "x"}) is True


def test_fetching_a_local_file_returns_it_in_place(tmp_path):
    audio = tmp_path / "iv.mp3"
    audio.write_bytes(b"data")
    dest = tmp_path / "scratch"

    got = sources.fetch({"kind": LOCAL, "path": str(audio)}, dest)

    assert got == audio
    assert not (dest / "iv.mp3").exists()  # not copied


def test_a_missing_local_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sources.fetch({"kind": LOCAL, "path": str(tmp_path / "gone.mp3")}, tmp_path)


# -------------------------------------------------------------------- json


def test_enumerate_json_records_urls_without_downloading():
    entries = [
        {"name": "02_00113", "audio_url": "https://h/a/02_00113-a_01-medium.mp4/index.m3u8"},
        {"name": "02_00114", "audio_url": "https://h/b/plain.mp3"},
    ]
    found = sources.enumerate_json(entries)

    assert [f["kind"] for f in found] == [URL, URL]
    assert found[0]["id"] == "02_00113"
    assert found[0]["filename"] == "02_00113-a_01-medium.mp4"
    assert found[1]["filename"] == "plain.mp3"


def test_enumerate_json_skips_entries_missing_a_url():
    found = sources.enumerate_json([{"name": "x"}, {"audio_url": "https://h/a.mp3"}])
    assert found == []


def test_urls_are_fetched_through_the_proxy(tmp_path, monkeypatch):
    """urllib honours http_proxy, which is what WebProxy sets."""
    import io
    import urllib.request

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda url, timeout=None: _Ctx(io.BytesIO(b"audio-bytes")),
    )
    got = sources.fetch(
        {"kind": URL, "id": "iv", "url": "https://h/a.mp3", "filename": "iv.mp3"},
        tmp_path,
    )

    assert got.read_bytes() == b"audio-bytes"
    assert not list(tmp_path.glob("*.part"))  # nothing half-written left behind


def test_streaming_manifests_go_through_yt_dlp(tmp_path, monkeypatch):
    calls = {}

    def fake_run(command, capture_output=True, text=True):
        calls["cmd"] = command
        # yt-dlp writes the file itself
        idx = command.index("-o")
        __import__("pathlib").Path(command[idx + 1]).write_bytes(b"hls")
        return _Result(0, "")

    monkeypatch.setattr(sources.subprocess, "run", fake_run)
    got = sources.fetch(
        {"kind": URL, "id": "iv", "url": "https://h/a.mp4/index.m3u8",
         "filename": "iv.mp4"},
        tmp_path,
    )

    assert calls["cmd"][0] == "yt-dlp"
    assert got.read_bytes() == b"hls"


def test_a_yt_dlp_failure_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sources.subprocess, "run",
        lambda *a, **k: _Result(1, "HTTP Error 403"),
    )
    with pytest.raises(RuntimeError, match="403"):
        sources.fetch(
            {"kind": URL, "id": "iv", "url": "https://h/a.mp4/index.m3u8"}, tmp_path
        )


# --------------------------------------------------------------------- smb


def test_smb_fetch_needs_a_password_in_the_environment(tmp_path, monkeypatch):
    """Credentials come from the job environment, never from the queue."""
    monkeypatch.delenv("SMB_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="SMB_PASSWORD"):
        sources.fetch(
            {"kind": SMB, "remote_path": "//s/share/iv.mp3", "server": "s"}, tmp_path
        )


def test_the_queue_reference_carries_no_credentials():
    """Anything in the payload lands on shared disk, so it must be safe there."""
    entry = {"kind": SMB, "remote_path": "//s/share/f/iv.mp3", "server": "s",
             "folder": "f", "id": "iv"}
    serialised = json.dumps(entry).lower()

    assert "password" not in serialised
    assert "secret" not in serialised


# ---------------------------------------------------------------- helpers


def test_sheet_folder_names_are_normalised():
    assert sources._folder_name("02_00113-a") == "02_00113"
    assert sources._folder_name("02_00113") == "02_00113"
    assert sources._folder_name("odd-name") == "odd-name"


def test_stream_detection():
    assert sources._is_stream("https://h/a.mp4/index.m3u8")
    assert sources._is_stream("https://h/a.mpd")
    assert not sources._is_stream("https://h/a.mp3")


def test_an_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown source kind"):
        sources.fetch({"kind": "carrier-pigeon"}, tmp_path)


class _Ctx:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.stream.read(n)


class _Result:
    def __init__(self, returncode, stderr):
        self.returncode = returncode
        self.stderr = stderr
