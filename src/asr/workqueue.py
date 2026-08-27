"""A filesystem work queue, for many workers sharing one batch.

Built for a scheduler-managed cluster where you get lots of GPUs but no
long-lived service and no message broker:

* **Many workers, one queue.** Eight GPUs on a node - or eighty across nodes -
  each pull the next file when they finish the last one. Static round-robin
  splitting (worker ``i`` takes every ``n``-th file) loses badly to stragglers
  when interview lengths vary from 20 minutes to three hours; dynamic claiming
  does not.
* **Survives the wall clock.** A job killed at its time limit leaves its
  in-flight tasks leased, not lost. The next job reaps expired leases and
  carries on, so a batch spans as many scheduler allocations as it needs.
* **No server.** State is directories on a shared filesystem.

The claim protocol
------------------

Claiming is arbitrated by an exclusive token created with
``O_CREAT | O_EXCL``, the one primitive that is atomically exclusive on every
filesystem that matters here - POSIX, Lustre, GPFS and NTFS alike. Exactly one
worker can create ``claimed/.<id>.claim``; only that worker then renames the
payload into ``claimed/``.

Ownership metadata lives in the *filename*::

    <id>@@<attempts>@@<worker>@@<lease_expiry>.json

so no write is needed to record a claim. Earlier revisions relied on
read-modify-write, and then on rename exclusivity alone; both double-issued
tasks under eight concurrent workers (measured: 3% and 17% respectively). The
token removes the ambiguity.

Payload files are written once at submit and rewritten only in terminal states,
where nothing races. A transient read failure therefore costs metadata, never
the task: ownership is already settled by the filename.

Atomic rename within a directory is POSIX-guaranteed and holds on Lustre and
GPFS. It is *not* safe over NFSv3 without locking - use a parallel scratch
filesystem, not a home directory.

Layout::

    queue/
      pending/   waiting
      claimed/   leased to a worker, with an expiry
      done/      finished
      failed/    gave up after max_attempts
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger(__name__)

PENDING, CLAIMED, DONE, FAILED = "pending", "claimed", "done", "failed"
_STATES = (PENDING, CLAIMED, DONE, FAILED)

#: Default lease. A worker that dies holds its task only this long.
DEFAULT_LEASE_SECONDS = 3600

#: Field separator in task filenames. Never produced by :func:`_slug`.
SEP = "@@"
_NO_WORKER = "-"

_READ_RETRIES = 4
_READ_BACKOFF = 0.05


@dataclass
class Task:
    id: str
    payload: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    worker: Optional[str] = None
    leased_until: float = 0.0
    error: Optional[str] = None
    history: list[str] = field(default_factory=list)
    #: The filename currently holding this task, so renames need no guessing.
    name: str = ""

    def content(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "payload": self.payload,
                "error": self.error,
                "history": self.history[-20:],
            },
            indent=2,
        )


def worker_id() -> str:
    """Something unique and traceable back to a host, rank and job."""
    parts = [socket.gethostname(), str(os.getpid())]
    for var in ("SLURM_JOB_ID", "SLURM_PROCID", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if value:
            parts.append(f"{var.split('_')[-1].lower()}{value}")
    return _slug("-".join(parts))


class FileWorkQueue:
    """Claim-and-lease queue backed by directories on a shared filesystem."""

    def __init__(self, root: str | Path, max_attempts: int = 3):
        self.root = Path(root)
        self.max_attempts = max_attempts
        for state in _STATES:
            (self.root / state).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- producing

    def submit(self, payloads: Iterable[dict[str, Any]], key: str = "id") -> int:
        """Add tasks, skipping any id already known in any state. Returns added."""
        added = 0
        for payload in payloads:
            task_id = _slug(str(payload.get(key) or uuid.uuid4().hex))
            if self._find(task_id) is not None:
                continue
            task = Task(id=task_id, payload=payload)
            _atomic_write(
                self.root / PENDING / _encode(task_id, 0, _NO_WORKER, 0), task.content()
            )
            added += 1
        return added

    # ------------------------------------------------------------- consuming

    def claim(
        self, worker: Optional[str] = None, lease: float = DEFAULT_LEASE_SECONDS
    ) -> Optional[Task]:
        """Take the next pending task, or ``None`` when nothing is waiting.

        Arbitration is the exclusive token: whoever creates it owns the task
        and moves the payload; everyone else skips to the next candidate.
        """
        worker = _slug(worker or worker_id())
        for candidate in sorted((self.root / PENDING).glob("*.json")):
            task_id, attempts, _, _ = _decode(candidate.name)
            if task_id is None:
                self._quarantine(candidate, "unparseable filename")
                continue

            if not self._take_token(task_id):
                continue  # another worker owns this one

            expiry = time.time() + lease
            name = _encode(task_id, attempts + 1, worker, expiry)
            try:
                os.rename(candidate, self.root / CLAIMED / name)
            except OSError:
                _remove(self._token(task_id))  # it was reaped or finished
                continue

            task = self._load(self.root / CLAIMED / name, task_id)
            task.attempts = attempts + 1
            task.worker = worker
            task.leased_until = expiry
            task.name = name
            task.history.append(f"claimed by {worker} at {_stamp()}")
            return task
        return None

    def heartbeat(self, task: Task, lease: float = DEFAULT_LEASE_SECONDS) -> bool:
        """Extend the lease by renaming. Returns False if the task was reaped."""
        expiry = time.time() + lease
        new_name = _encode(task.id, task.attempts, task.worker or _NO_WORKER, expiry)
        if new_name == task.name:
            return True
        try:
            os.rename(self.root / CLAIMED / task.name, self.root / CLAIMED / new_name)
        except OSError:
            log.warning("heartbeat failed for %s; it may have been reaped", task.id)
            return False
        task.name = new_name
        task.leased_until = expiry
        return True

    def complete(self, task: Task) -> None:
        task.history.append(f"done by {task.worker} at {_stamp()}")
        self._retire(task, DONE)

    def fail(self, task: Task, error: str, retry: bool = True) -> str:
        """Return a task to the queue, or park it in ``failed/`` if spent."""
        task.error = error
        exhausted = task.attempts >= self.max_attempts
        state = PENDING if (retry and not exhausted) else FAILED
        task.history.append(
            f"{'requeued' if state == PENDING else 'failed'} at {_stamp()}: {error[:200]}"
        )
        self._retire(task, state)
        return state

    # ------------------------------------------------------------ maintenance

    def reap(self) -> int:
        """Return expired claims to ``pending``; returns how many were reaped.

        This is what makes a batch survive a job hitting its wall clock: tasks
        in flight when Slurm killed the job come back instead of vanishing.
        """
        now = time.time()
        self._sweep_tokens(now, max_age=60.0)
        reaped = 0
        for path in sorted((self.root / CLAIMED).glob("*.json")):
            task_id, attempts, worker, expiry = _decode(path.name)
            if task_id is None:
                self._quarantine(path, "unparseable filename")
                continue
            if expiry > now:
                continue
            state = FAILED if attempts >= self.max_attempts else PENDING
            try:
                os.rename(path, self.root / state / _encode(task_id, attempts, worker, 0))
            except OSError:
                continue  # another reaper got there first
            _remove(self._token(task_id))
            reaped += 1
            log.warning("reaped %s from %s -> %s", task_id, worker, state)
        return reaped

    def requeue_failed(self) -> int:
        """Move everything in ``failed/`` back to ``pending`` with a fresh count."""
        moved = 0
        for path in sorted((self.root / FAILED).glob("*.json")):
            task_id, _, _, _ = _decode(path.name)
            if task_id is None:
                continue
            try:
                os.rename(path, self.root / PENDING / _encode(task_id, 0, _NO_WORKER, 0))
            except OSError:
                continue
            moved += 1
        return moved

    # ---------------------------------------------------------------- status

    def counts(self) -> dict[str, int]:
        return {s: len(list((self.root / s).glob("*.json"))) for s in _STATES}

    def is_drained(self) -> bool:
        counts = self.counts()
        return counts[PENDING] == 0 and counts[CLAIMED] == 0

    def tasks(self, state: str) -> list[Task]:
        if state not in _STATES:
            raise ValueError(f"unknown state {state!r}")
        out: list[Task] = []
        for path in sorted((self.root / state).glob("*.json")):
            task_id, attempts, worker, expiry = _decode(path.name)
            if task_id is None:
                continue
            task = self._load(path, task_id)
            task.attempts = attempts
            task.worker = None if worker == _NO_WORKER else worker
            task.leased_until = expiry
            task.name = path.name
            out.append(task)
        return out

    # --------------------------------------------------------------- private

    def _token(self, task_id: str) -> Path:
        # Leading dot keeps tokens out of every "*.json" listing.
        return self.root / CLAIMED / f".{task_id}.claim"

    def _take_token(self, task_id: str) -> bool:
        """Create the exclusive claim token. Only one caller can succeed."""
        try:
            fd = os.open(self._token(task_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return False
        try:
            os.write(fd, str(time.time()).encode("ascii"))
        finally:
            os.close(fd)
        return True

    def _sweep_tokens(self, now: float, max_age: float) -> None:
        """Drop tokens with no claimed payload behind them.

        A worker killed between taking the token and renaming the payload would
        otherwise block that task forever.
        """
        for token in self.root.joinpath(CLAIMED).glob(".*.claim"):
            task_id = token.name[1:-len(".claim")]
            if any((self.root / CLAIMED).glob(f"{_escape(task_id)}{SEP}*.json")):
                continue
            try:
                if now - token.stat().st_mtime < max_age:
                    continue
            except OSError:
                continue
            log.warning("clearing orphaned claim token for %s", task_id)
            _remove(token)

    def _retire(self, task: Task, state: str) -> None:
        """Move a claimed task to its next state, then record the metadata."""
        name = _encode(task.id, task.attempts, task.worker or _NO_WORKER, 0)
        destination = self.root / state / name
        try:
            os.rename(self.root / CLAIMED / task.name, destination)
        except OSError as exc:
            log.error("could not retire %s to %s: %s", task.id, state, exc)
            return
        task.name = name
        _remove(self._token(task.id))
        # Terminal directories are not raced over, so rewriting here is safe.
        try:
            _atomic_write(destination, task.content())
        except OSError as exc:
            log.warning("could not update %s metadata: %s", task.id, exc)

    def _load(self, path: Path, task_id: str) -> Task:
        """Read a payload, tolerating a transient shared-filesystem hiccup.

        Ownership is already settled by the filename, so a failed read costs
        metadata, never the task.
        """
        for attempt in range(_READ_RETRIES):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return Task(
                    id=task_id,
                    payload=data.get("payload") or {},
                    error=data.get("error"),
                    history=list(data.get("history") or []),
                )
            except (OSError, ValueError):
                if attempt == _READ_RETRIES - 1:
                    log.warning("payload for %s unreadable; continuing without it", task_id)
                    return Task(id=task_id)
                time.sleep(_READ_BACKOFF * (attempt + 1))
        return Task(id=task_id)

    def _find(self, task_id: str) -> Optional[str]:
        pattern = f"{_escape(task_id)}{SEP}*.json"
        for state in _STATES:
            if any((self.root / state).glob(pattern)):
                return state
        return None

    def _quarantine(self, path: Path, reason: str) -> None:
        log.error("quarantining %s: %s", path.name, reason)
        target = self.root / FAILED / _encode(_slug(path.stem), 0, _NO_WORKER, 0)
        try:
            os.replace(path, target)
        except OSError:
            pass


# ------------------------------------------------------------------ encoding


def _encode(task_id: str, attempts: int, worker: str, expiry: float) -> str:
    fields = [task_id, str(int(attempts)), worker or _NO_WORKER, str(int(expiry))]
    return SEP.join(fields) + ".json"


def _decode(name: str) -> tuple[Optional[str], int, str, float]:
    """``<id>@@<attempts>@@<worker>@@<expiry>.json`` -> its parts."""
    stem = name[:-5] if name.endswith(".json") else name
    parts = stem.rsplit(SEP, 3)
    if len(parts) != 4:
        return None, 0, _NO_WORKER, 0.0
    task_id, attempts, worker, expiry = parts
    try:
        return task_id, int(attempts), worker, float(expiry)
    except ValueError:
        return None, 0, _NO_WORKER, 0.0


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _atomic_write(destination: Path, text: str) -> None:
    """Write via a uniquely-named temp file so no partial file is ever visible.

    The temp name carries a uuid, not just a pid: several threads in one process
    write into these directories concurrently, and a shared temp name would let
    them clobber each other mid-write.
    """
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def _slug(value: str) -> str:
    """Filesystem-safe, and guaranteed not to contain the field separator."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value)
    return safe[:180] or uuid.uuid4().hex


def _escape(value: str) -> str:
    return "".join("[" + c + "]" if c in "*?[" else c for c in value)


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


__all__ = [
    "CLAIMED",
    "DEFAULT_LEASE_SECONDS",
    "DONE",
    "FAILED",
    "FileWorkQueue",
    "PENDING",
    "SEP",
    "Task",
    "worker_id",
]
