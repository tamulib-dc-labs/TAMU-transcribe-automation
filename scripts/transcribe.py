"""Transcription worker. One process, one GPU pair, one shared queue.

Replaces the WhisperX transcription script. Run one per GPU node; run as many
as you have allocation for. They coordinate only through the filesystem, so
there is no broker to stand up and nothing to clean up when a job is killed.

    python scripts/transcribe.py fill   --queue Q --input IN --output OUT
    python scripts/transcribe.py work   --queue Q --output OUT
    python scripts/transcribe.py status --queue Q --failures

Two design choices carry their weight:

**A queue, not a split.** Oral-history interviews run from twenty minutes to
three hours. Splitting the file list N ways leaves most GPUs idle behind the
one worker that drew the long recordings. Claiming the next file when you
finish the last one does not.

**Leases, not assignments.** A worker stops claiming before its wall clock and
exits cleanly; anything still in flight stays leased rather than lost, and the
next job reaps it and carries on. Resuming a batch is just submitting again.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Cache/offline environment must be set before torch or NeMo import, so a
# compute node with HF_HUB_OFFLINE=1 never reaches for the network.
if "HF_HOME" in os.environ:
    os.environ.setdefault("TORCH_HOME", os.environ["HF_HOME"])
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
if os.environ.get("HF_HUB_OFFLINE") == "1":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr.lines import LineConfig  # noqa: E402
from src.asr.output import write_outputs  # noqa: E402
from src.asr.parakeet import DEFAULT_PARAKEET_MODEL, ParakeetConfig  # noqa: E402
from src.asr.preprocess import already_done, prepare_audio  # noqa: E402
from src.asr import sources  # noqa: E402
from src.asr.sortformer import DEFAULT_SORTFORMER_MODEL, SortformerConfig  # noqa: E402
from src.asr.transcriber import Transcriber, TranscriberConfig  # noqa: E402
from src.asr.workqueue import DEFAULT_LEASE_SECONDS, FileWorkQueue, worker_id  # noqa: E402

log = logging.getLogger("transcribe")

#: Leave this much of the wall clock spare so an in-flight file can finish.
DEADLINE_MARGIN_SECONDS = 600


class Worker:
    def __init__(self, args, engine=None):
        self.args = args
        self.queue = FileWorkQueue(args.queue, max_attempts=args.max_attempts)
        self.output = Path(args.output)
        self.scratch = Path(args.scratch or (Path(args.queue).parent / "prepared"))
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.identity = args.worker or worker_id()
        self.engine = engine
        self.deadline = (
            time.time() + args.deadline_minutes * 60 - DEADLINE_MARGIN_SECONDS
            if args.deadline_minutes
            else None
        )
        self._stop = False
        self.processed = 0
        self.failed = 0

    # ------------------------------------------------------------- lifecycle

    def install_signal_handlers(self) -> None:
        """Slurm sends SIGTERM before SIGKILL; wind down instead of dying."""

        def handler(signum, _frame):
            log.warning("signal %s received; finishing the current file", signum)
            self._stop = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def should_stop(self) -> Optional[str]:
        if self._stop:
            return "signal"
        if self.deadline and time.time() > self.deadline:
            return "deadline"
        return None

    def load_engine(self) -> Transcriber:
        if self.engine is None:
            self.engine = build_engine(self.args)
        return self.engine

    # ------------------------------------------------------------------ loop

    def run(self) -> dict:
        self.install_signal_handlers()
        log.info("worker %s starting; queue=%s", self.identity, self.args.queue)

        # Reaping on startup recovers tasks orphaned by a previous job hitting
        # its wall clock. Idempotent, so every worker can do it at once.
        reaped = self.queue.reap()
        if reaped:
            log.info("reaped %d task(s) from a previous allocation", reaped)

        idle_since = None
        while True:
            reason = self.should_stop()
            if reason:
                log.info("stopping (%s) after %d file(s)", reason, self.processed)
                break

            task = self.queue.claim(worker=self.identity, lease=self.args.lease)
            if task is None:
                if self.args.exit_when_empty:
                    log.info("queue drained; exiting")
                    break
                idle_since = idle_since or time.time()
                if self.args.idle_timeout and time.time() - idle_since > self.args.idle_timeout:
                    log.info("idle for %ss; exiting", self.args.idle_timeout)
                    break
                time.sleep(self.args.poll_interval)
                continue

            idle_since = None
            self._process(task)

        return {
            "worker": self.identity,
            "processed": self.processed,
            "failed": self.failed,
            "counts": self.queue.counts(),
        }

    def _process(self, task) -> None:
        log.info("[%s] claimed %s (attempt %d)", self.identity, task.id, task.attempts)
        fetched = None
        prepared = None
        started = time.time()
        try:
            # Fetch only this interview. The queue holds references, so storage
            # is bounded by the number of workers, not the size of the collection.
            fetched = sources.fetch(task.payload, self.scratch)
            source = Path(fetched)
            self.queue.heartbeat(task, lease=self.args.lease)
            log.info("[%s] fetched %s (%.0f MB)", self.identity, source.name,
                     source.stat().st_size / 1e6)

            prepared = prepare_audio(
                source,
                self.scratch,
                denoise=self.args.denoise,
                boost=3.0 if self.args.denoise else 1.0,
            )
            # Preparation on a three-hour interview can outlast a short lease.
            self.queue.heartbeat(task, lease=self.args.lease)

            transcript = self.load_engine().run(prepared)
            if not transcript.segments:
                raise RuntimeError("no speech recognised")

            write_outputs(
                transcript,
                task.id,  # outputs are named after the queued interview id
                self.output,
                language=self.args.language,
                max_line_width=self.args.max_line_width,
                max_line_count=self.args.max_line_count,
            )
            self.queue.complete(task)
            self.processed += 1
            log.info(
                "[%s] done %s in %.0fs (%d words, %d speaker(s))",
                self.identity, task.id, time.time() - started,
                len(transcript.words), len(transcript.meta.get("speakers") or []),
            )
        except Exception as exc:  # noqa: BLE001 - a bad file must not kill the worker
            self.failed += 1
            # A missing or unreadable source will never succeed on retry.
            permanent = isinstance(exc, (FileNotFoundError, IsADirectoryError))
            state = self.queue.fail(
                task, f"{type(exc).__name__}: {exc}", retry=not permanent
            )
            log.error("[%s] %s -> %s: %s", self.identity, task.id, state, exc)
        finally:
            _cleanup(prepared)
            if fetched is not None and sources.is_temporary(task.payload):
                _cleanup(Path(fetched))


def _cleanup(path) -> None:
    if path and Path(path).exists():
        try:
            Path(path).unlink()
        except OSError:
            pass


def build_engine(args) -> Transcriber:
    """Parakeet for words; Sortformer for speaker turns.

    Pinning the two to different GPUs is what makes ``--parallel-models`` a
    real overlap rather than two threads contending for one device.
    """
    return Transcriber(
        TranscriberConfig(
            parakeet=ParakeetConfig(
                model_id=args.parakeet_model, device=args.words_device
            ),
            sortformer=SortformerConfig(
                model_id=args.sortformer_model, device=args.turns_device
            ),
            lines=LineConfig(max_chars=args.max_line_width),
            diarize=not args.no_diarize,
            parallel=args.parallel_models,
        )
    )


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Oral-history transcription worker (Parakeet + Sortformer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    fill = sub.add_parser(
        "fill", help="enumerate a source into the queue (downloads nothing)"
    )
    fill.add_argument("--queue", required=True)
    fill.add_argument("--output", required=True,
                      help="used to skip interviews that already have a transcript")
    fill.add_argument("--source", choices=("local", "smb", "json"), default="local",
                      help="where the audio lives. Workers fetch their own copy")
    fill.add_argument("--input", default=None, help="directory, for --source local")
    fill.add_argument("--config-json", default=None,
                      help="reviewer-repo config file, for --source json")
    fill.add_argument("--max-files", type=int, default=0,
                      help="cap how many interviews are queued (0 = no cap)")
    fill.add_argument("--skip-list", default=None,
                      help="JSON list of interview ids already transcribed "
                           "elsewhere, e.g. in the transcripts repository")

    status = sub.add_parser("status", help="show queue counts")
    status.add_argument("--queue", required=True)
    status.add_argument("--failures", action="store_true", help="list failed tasks")

    requeue = sub.add_parser("requeue", help="move failed tasks back to pending")
    requeue.add_argument("--queue", required=True)

    work = sub.add_parser("work", help="claim and transcribe until drained")
    work.add_argument("--queue", required=True)
    work.add_argument("--output", required=True)
    work.add_argument("--scratch", default=None,
                      help="where prepared WAVs go (default: alongside the queue)")
    work.add_argument("--worker", default=None, help="override the worker identity")

    work.add_argument("--parakeet-model", default=DEFAULT_PARAKEET_MODEL)
    work.add_argument("--sortformer-model", default=DEFAULT_SORTFORMER_MODEL)
    work.add_argument("--words-device", default="cuda:0", help="GPU for Parakeet")
    work.add_argument("--turns-device", default="cuda:1",
                      help="GPU for Sortformer. Different from --words-device is "
                           "what makes --parallel-models a real overlap")
    work.add_argument("--parallel-models", action="store_true", default=True,
                      help="run the two models concurrently (needs two devices)")
    work.add_argument("--sequential-models", dest="parallel_models",
                      action="store_false",
                      help="run them one after the other on a single GPU")
    work.add_argument("--no-diarize", action="store_true",
                      help="skip Sortformer: words and timings only, no speakers")

    work.add_argument("--lease", type=float, default=DEFAULT_LEASE_SECONDS,
                      help="seconds a claim is held before it can be reaped")
    work.add_argument("--max-attempts", type=int, default=3)
    work.add_argument("--deadline-minutes", type=float, default=None,
                      help="stop claiming this many minutes after start; set it "
                           "below the Slurm wall clock")
    work.add_argument("--exit-when-empty", action="store_true", default=True)
    work.add_argument("--stay-alive", dest="exit_when_empty", action="store_false",
                      help="keep polling for new work instead of exiting")
    work.add_argument("--idle-timeout", type=float, default=None)
    work.add_argument("--poll-interval", type=float, default=15.0)

    work.add_argument("--denoise", action="store_true",
                      help="run the noisereduce pass before ASR. Off by default: "
                           "both models are trained on noise-augmented audio and "
                           "normalise internally, and spectral gating can add "
                           "artefacts that hurt an end-to-end model")
    work.add_argument("--language", default="en",
                      help="tag written to the JSON; the model detects its own")
    work.add_argument("--max-line-width", type=int, default=42)
    work.add_argument("--max-line-count", type=int, default=2)
    work.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "fill":
        return _fill(args)
    if args.command == "status":
        return _status(args)
    if args.command == "requeue":
        queue = FileWorkQueue(args.queue)
        print(f"requeued {queue.requeue_failed()} task(s)")
        return 0
    if args.command == "work":
        print(json.dumps(Worker(args).run(), indent=2))
        return 0

    build_parser().print_help()
    return 2


def _fill(args) -> int:
    """Enumerate the source into the queue. Moves no audio.

    Idempotent - already-queued and already-transcribed interviews are skipped -
    so every array task can safely run this at startup and the first one to
    arrive populates the queue for the rest.
    """
    queue = FileWorkQueue(args.queue)
    found = _enumerate(args)

    # The local output folder is a cache on scratch; the skip list carries what
    # the durable record (the transcripts repo) already holds.
    skip = _load_skip_list(args.skip_list)
    pending = [
        p for p in found
        if p["id"] not in skip and not already_done(p["id"], args.output)
    ]
    if args.max_files:
        pending = pending[: args.max_files]

    added = queue.submit(pending)
    print(
        f"found {len(found)} interview(s), {len(found) - len(pending)} already done "
        f"({len(skip)} from the transcripts repo), {added} newly queued"
    )
    print(json.dumps(queue.counts(), indent=2))
    return 0


def _load_skip_list(path) -> set:
    """Interview ids already transcribed, per the durable record."""
    if not path or not Path(path).exists():
        return set()
    try:
        return {str(i) for i in json.loads(Path(path).read_text(encoding="utf-8"))}
    except (OSError, ValueError) as exc:
        log.warning("could not read skip list %s: %s", path, exc)
        return set()


def _enumerate(args) -> list:
    from src.config import get_config

    if args.source == "local":
        if not args.input:
            raise SystemExit("--source local needs --input")
        return sources.enumerate_local(args.input)

    if args.source == "json":
        if not args.config_json:
            raise SystemExit("--source json needs --config-json")
        entries = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        if isinstance(entries, dict):
            entries = entries.get("items") or entries.get("entries") or []
        return sources.enumerate_json(entries)

    config = get_config()
    folders = sources.folders_from_sheet(config.sheet_url, config.max_folders)
    print(f"sheet lists {len(folders)} folder(s) to process")
    return sources.enumerate_smb(
        server=config.smb_server,
        share=config.smb_share,
        base_path=config.smb_base_path,
        username=os.environ.get("SMB_USERNAME", config.smb_username),
        password=os.environ.get("SMB_PASSWORD", ""),
        folders=folders,
    )


def _status(args) -> int:
    queue = FileWorkQueue(args.queue)
    print(json.dumps(queue.counts(), indent=2))
    if args.failures:
        for task in queue.tasks("failed"):
            print(f"  {task.id}: {task.error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
