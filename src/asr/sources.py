"""Where audio comes from, and how one worker fetches its own file.

The old pipeline downloaded the whole collection up front, then transcribed it.
That meant the GPU sat idle through every download, storage had to hold the
entire collection at once, and a failure late in the run wasted everything
before it.

Here the queue holds *references*, not files. ``enumerate_*`` lists what exists
without moving any bytes; each worker then fetches only the interview it just
claimed, transcribes it, and deletes the copy. Storage stays bounded by the
number of workers rather than the size of the collection.

Three kinds of reference::

    {"kind": "local", "path": "/scratch/.../iv_001.mp3"}
    {"kind": "url",   "url":  "https://.../iv_001.mp4/index.m3u8"}
    {"kind": "smb",   "remote_path": "//server/share/folder/iv_001.mp3"}

``url`` works from a compute node through HPRC's WebProxy. ``smb`` speaks port
445 directly, which the HTTP proxy does *not* carry - see ``SMB_REACHABILITY``
below.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

LOCAL, URL, SMB = "local", "url", "smb"

#: WebProxy forwards HTTP and HTTPS only. SMB is port 445 and does not go
#: through it, so fetching from the CIFS share on a compute node depends on the
#: node having direct network access to that server. Verify once with an
#: interactive job before relying on it:
#:
#:     srun --pty --partition=gpu --gres=gpu:a100:1 --time=00:30:00 bash
#:     python -c "import smbclient; smbclient.register_session('cifs.library.tamu.edu', username='...', password='...')"
#:
#: If that fails, stage the audio to scratch from a login node once and run
#: with ``--source local`` instead - everything else is unchanged.
SMB_REACHABILITY = "verify from a compute node; WebProxy does not carry port 445"

AUDIO_SUFFIXES = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov")


# --------------------------------------------------------------- enumerate


def enumerate_local(root: str | Path) -> list[dict[str, Any]]:
    """Every audio file already on disk."""
    from .preprocess import find_audio_files

    return [
        {"id": p.stem, "kind": LOCAL, "path": str(p.resolve())}
        for p in find_audio_files(root)
    ]


def enumerate_json(entries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audio referenced by the reviewer repo's config JSON.

    The audio lives in the ``audio`` field. Note that these entries also carry a
    ``url`` field - that is the *transcript* URL the reviewer app links to, not
    the recording - so it must never be used as an audio source.

    Nothing is downloaded here; each entry becomes a reference in the queue.
    """
    out: list[dict[str, Any]] = []
    for entry in entries:
        url = entry.get("audio") or entry.get("audio_url")
        name = entry.get("name") or entry.get("id")
        if not url or not name:
            log.warning(
                "skipping config entry with no 'audio' or no 'name': %r", entry
            )
            continue
        out.append(
            {
                "id": _slug(name),
                "kind": URL,
                "url": url,
                "name": name,
                "filename": _filename_from_url(url, name),
            }
        )
    return out


def enumerate_smb(
    server: str,
    share: str,
    base_path: str,
    username: str,
    password: str,
    folders: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    """List audio on the CIFS share without downloading any of it."""
    import smbclient

    smbclient.register_session(server, username=username, password=password)
    root = f"//{server}/{share}/{base_path}".replace("\\", "/")

    targets = list(folders) if folders is not None else _list_dirs(smbclient, root)
    out: list[dict[str, Any]] = []
    for folder in targets:
        remote_folder = f"{root}/{folder}"
        try:
            names = list(smbclient.scandir(remote_folder))
        except Exception as exc:  # noqa: BLE001 - one bad folder must not stop the scan
            log.warning("cannot list %s: %s", remote_folder, exc)
            continue
        for entry in names:
            if entry.is_dir() or not entry.name.lower().endswith(AUDIO_SUFFIXES):
                continue
            out.append(
                {
                    "id": _slug(Path(entry.name).stem),
                    "kind": SMB,
                    "remote_path": f"{remote_folder}/{entry.name}",
                    "server": server,
                    "folder": folder,
                }
            )
    return out


def folders_from_sheet(sheet_url: str, max_folders: int = 0) -> list[str]:
    """Folder names to process, read from the tracking Google Sheet.

    Exports the sheet as CSV over HTTPS, so this works from a compute node
    through WebProxy.
    """
    import pandas as pd

    sheet_id = sheet_url.split("/d/")[1].split("/")[0]
    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    frame = pd.read_csv(csv_url)

    folders: list[str] = []
    for value in frame.get("as", []):
        name = _folder_name(str(value).strip())
        if name and name not in folders:
            folders.append(name)
        if max_folders and len(folders) >= max_folders:
            break
    return folders


# ------------------------------------------------------------------- fetch


def fetch(payload: dict[str, Any], dest_dir: str | Path) -> Path:
    """Bring one interview's audio to ``dest_dir`` and return its path.

    Raises on failure; the caller decides whether that is retryable.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    kind = payload.get("kind", LOCAL)

    if kind == LOCAL:
        path = Path(payload["path"])
        if not path.exists():
            raise FileNotFoundError(f"source audio missing: {path}")
        return path

    if kind == URL:
        return _fetch_url(payload, dest_dir)

    if kind == SMB:
        return _fetch_smb(payload, dest_dir)

    raise ValueError(f"unknown source kind {kind!r}; expected one of local/url/smb")


def _fetch_url(payload: dict[str, Any], dest_dir: Path) -> Path:
    """Download over HTTP(S). Streaming manifests go through yt-dlp."""
    url = payload["url"]
    destination = dest_dir / payload.get("filename", f"{payload['id']}.mp4")

    if _is_stream(url):
        # HLS/DASH: yt-dlp handles the manifest and honours http_proxy.
        command = ["yt-dlp", "--no-progress", "-o", str(destination), url]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not destination.exists():
            raise RuntimeError(
                f"yt-dlp failed for {url}: {result.stderr.strip()[:400]}"
            )
        return destination

    import urllib.request

    # urllib honours http_proxy / https_proxy, which is what WebProxy sets.
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    os.replace(temporary, destination)
    return destination


def _fetch_smb(payload: dict[str, Any], dest_dir: Path) -> Path:
    """Copy one file off the CIFS share.

    Credentials come from the environment so they never sit in the queue.
    """
    # Check the cheap thing first: a missing password is a config error and
    # should not be reported as a missing dependency.
    server = payload.get("server") or ""
    username = os.environ.get("SMB_USERNAME")
    password = os.environ.get("SMB_PASSWORD")
    if not password:
        raise RuntimeError(
            "SMB_PASSWORD is not set in the job environment; the worker cannot "
            "reach the share"
        )

    try:
        import smbclient
    except ImportError as exc:
        raise RuntimeError(
            "smbprotocol is not installed, so this worker cannot fetch from the "
            "CIFS share. Install it, or enumerate with --source local."
        ) from exc

    smbclient.register_session(server, username=username, password=password)

    remote = payload["remote_path"]
    destination = dest_dir / Path(remote).name
    temporary = destination.with_suffix(destination.suffix + ".part")
    with smbclient.open_file(remote, mode="rb") as source, temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    os.replace(temporary, destination)
    return destination


def is_temporary(payload: dict[str, Any]) -> bool:
    """True when the fetched copy should be deleted after transcription.

    A ``local`` source is the collection itself and must survive; anything
    fetched is a scratch copy.
    """
    return payload.get("kind", LOCAL) != LOCAL


# ----------------------------------------------------------------- helpers


def _list_dirs(smbclient, root: str) -> list[str]:
    try:
        return [e.name for e in smbclient.scandir(root) if e.is_dir()]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"cannot list {root}: {exc}") from exc


def _folder_name(value: str) -> str:
    """``02_00113-a`` and ``02_00113`` both name the same folder."""
    match = re.match(r"(\d+)_(\d+)(?:-[a-zA-Z0-9]+)?", value)
    return f"{match.group(1)}_{match.group(2)}" if match else value


def _filename_from_url(url: str, name: str) -> str:
    match = re.search(r"/([^/]+\.(?:mp4|mp3|m4a|wav))(?:/|$)", url)
    return match.group(1) if match else f"{_slug(name)}.mp4"


def _is_stream(url: str) -> bool:
    return url.endswith((".m3u8", ".mpd")) or "/index.m3u8" in url


def slug(value: str) -> str:
    """Filesystem-safe id. Output files are named with this, so the caller can
    predict a transcript's filename from a config entry's ``name``."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value))[:180]


#: Kept as an alias so internal call sites read the same as before.
_slug = slug


__all__ = [
    "AUDIO_SUFFIXES",
    "LOCAL",
    "SMB",
    "SMB_REACHABILITY",
    "URL",
    "enumerate_json",
    "enumerate_local",
    "enumerate_smb",
    "fetch",
    "folders_from_sheet",
    "is_temporary",
    "slug",
]
