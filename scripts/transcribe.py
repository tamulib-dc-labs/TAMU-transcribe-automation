"""Transcription worker. One process, one GPU, one list of interviews.

    python scripts/transcribe.py run --output OUT --source json --config-json LIST

It lists the work, skips anything already transcribed, and goes through the
rest one at a time: fetch the recording, transcribe it, write the JSON and VTT,
delete its copy of the audio.

**No queue.** There used to be one, with claims and leases, so four array tasks
could share a collection without colliding. The pipeline is one job with one
worker now, so there was nothing left to arbitrate - only the failure modes. An
interview claimed by a job that then died stayed locked for the length of its
lease, invisible to the next run, which found nothing to do and exited in
milliseconds looking like a success.

What made the queue worth having is kept, because none of it needed a queue:

* **Resumable.** An interview listed in the skip file - built from the
  transcripts repository before this runs - is passed over. Being interrupted
  costs the file in flight and nothing else. The local output folder is not
  consulted: it is a cache on /scratch, and a copy there that never reached
  the repository would otherwise block the interview from ever being redone.
* **A deadline.** The worker stops starting new interviews with enough of the
  wall clock left to finish the one in hand, then exits cleanly.
* **Failures do not stop the run.** A file that cannot be fetched or
  transcribed is logged and counted; the rest still get done.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
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
from src.asr.preprocess import prepare_audio  # noqa: E402
from src.asr import sources  # noqa: E402
from src.asr.sortformer import DEFAULT_SORTFORMER_MODEL, SortformerConfig  # noqa: E402
from src.asr.transcriber import Transcriber, TranscriberConfig  # noqa: E402

log = logging.getLogger("transcribe")

#: Leave this much of the wall clock spare so an in-flight file can finish.
DEADLINE_MARGIN_SECONDS = 600

#: A CUDA context does not recover. Once the driver reports one of these, every
#: later CUDA call in the process fails too, so carrying on to the next
#: interview would only reproduce the same error once per remaining file.
_POISONS_THE_PROCESS = (
    "illegal memory access",
    "device-side assert",
    "CUDA error",
    "CUDA_ERROR",
    "cuDNN error",
)


def _is_unrecoverable(exc: BaseException) -> bool:
    message = f"{type(exc).__name__}: {exc}"
    return any(marker in message for marker in _POISONS_THE_PROCESS)


class Worker:
    """Transcribes a list of interviews, one after another."""

    def __init__(self, args, engine=None):
        self.args = args
        self.engine = engine
        self.output = Path(args.output)
        self.scratch = Path(args.scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.identity = f"{socket.gethostname()}-{os.getpid()}"
        self.deadline = (
            time.time() + args.deadline_minutes * 60 - DEADLINE_MARGIN_SECONDS
            if args.deadline_minutes
            else None
        )
        self._stop = False
        self.processed = 0
        self.failed = 0
        self.skipped = 0
        #: Set when the process itself is no longer usable.
        self.fatal = False

    # ------------------------------------------------------------- lifecycle

    def install_signal_handlers(self) -> None:
        """Slurm sends SIGTERM before SIGKILL; wind down instead of dying."""

        def handler(signum, _frame):
            log.warning("signal %s received; finishing the current interview", signum)
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
        if self.fatal:
            return "the CUDA context is unusable"
        return None

    def load_engine(self) -> Transcriber:
        if self.engine is None:
            self.engine = build_engine(self.args)
        return self.engine

    # ------------------------------------------------------------------ loop

    def run(self) -> dict:
        self.install_signal_handlers()
        log.info("worker %s starting", self.identity)
        # Say which way the models run, every time. Telling parallel from
        # sequential by reading interleaved NeMo output is guesswork, and
        # guessing wrong costs a whole job.
        log.info(
            "models: %s (words=%s, turns=%s)",
            "PARALLEL - two threads" if self.args.parallel_models
            else "SEQUENTIAL - one at a time",
            self.args.words_device,
            self.args.turns_device,
        )

        found = _enumerate(self.args)
        skip = _load_skip_list(self.args.skip_list)

        # Skipping is decided by the transcripts repository alone, via the
        # skip list the job wrote before this ran.
        #
        # The local output folder used to count too, and that is what made an
        # interview impossible to redo: /scratch keeps a JSON that the repo
        # does not have - because the upload failed, or because the transcript
        # was bad and got removed from the repo on purpose - and every later
        # run skipped it on the strength of the stale local copy. One record
        # of what is done, and it is the one that is shared.
        todo = []
        for item in found:
            if item["id"] in skip:
                self.skipped += 1
                continue
            todo.append(item)

        if self.args.max_files:
            todo = todo[: self.args.max_files]

        log.info(
            "%d interview(s) in the work list, %d already in the transcripts "
            "repo, %d to do",
            len(found), self.skipped, len(todo),
        )

        for position, item in enumerate(todo, 1):
            reason = self.should_stop()
            if reason:
                log.warning(
                    "stopping (%s); %d interview(s) not started",
                    reason, len(todo) - position + 1,
                )
                break
            log.info("[%d/%d] %s", position, len(todo), item["id"])
            self._process(item)

        log.info(
            "finished: %d transcribed, %d failed, %d skipped",
            self.processed, self.failed, self.skipped,
        )
        return {
            "worker": self.identity,
            "listed": len(found),
            "skipped": self.skipped,
            "processed": self.processed,
            "failed": self.failed,
        }

    def _process(self, item) -> None:
        fetched = None
        prepared = None
        started = time.time()
        try:
            # Fetch only this interview, so disk use is bounded by one file
            # rather than by the size of the collection.
            fetched = sources.fetch(item, self.scratch)
            source = Path(fetched)
            log.info("fetched %s (%.0f MB)", source.name, source.stat().st_size / 1e6)

            prepared = prepare_audio(
                source,
                self.scratch,
                denoise=self.args.denoise,
                boost=3.0 if self.args.denoise else 1.0,
            )

            transcript = self.load_engine().run(prepared)
            if not transcript.segments:
                raise RuntimeError("no speech recognised")

            write_outputs(
                transcript,
                item["id"],  # outputs are named after the interview id
                self.output,
                language=self.args.language,
                max_line_width=self.args.max_line_width,
                max_line_count=self.args.max_line_count,
            )
            self.processed += 1
            log.info(
                "done %s in %.0fs (%d words, %d speaker(s))",
                item["id"], time.time() - started,
                len(transcript.words), len(transcript.meta.get("speakers") or []),
            )
        except Exception as exc:  # noqa: BLE001 - a bad file must not stop the run
            self.failed += 1
            self.fatal = _is_unrecoverable(exc)
            log.error("FAILED %s: %s", item["id"], exc)
            if self.fatal:
                log.error(
                    "the CUDA context is unusable, so nothing else would "
                    "succeed in this process; re-run the job"
                )
        finally:
            _cleanup(prepared)
            if fetched is not None and sources.is_temporary(item):
                _cleanup(Path(fetched))


def _cleanup(path) -> None:
    if path and Path(path).exists():
        try:
            Path(path).unlink()
        except OSError as exc:
            log.warning("could not remove %s: %s", path, exc)


def build_engine(args) -> Transcriber:
    """Parakeet for words; Sortformer for speaker turns."""
    return Transcriber(
        TranscriberConfig(
            parakeet=ParakeetConfig(
                model_id=args.parakeet_model,
                device=args.words_device,
                long_audio=args.long_audio,
                chunk_seconds=args.chunk_seconds,
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
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="transcribe everything not already done")

    # where the work comes from
    run.add_argument("--output", required=True, help="folder for json/ and vtts/")
    run.add_argument("--scratch", default="/tmp/asr-audio",
                     help="where each recording is fetched to, then deleted")
    run.add_argument("--source", choices=("smb", "json", "local"), default="smb")
    run.add_argument("--input", help="folder to read, with --source local")
    run.add_argument("--config-json", help="work list, with --source json")
    run.add_argument("--skip-list",
                     help="JSON list of ids already transcribed elsewhere")
    run.add_argument("--max-files", type=int, default=0,
                     help="stop after this many, 0 for no limit")

    # models
    run.add_argument("--parakeet-model", default=DEFAULT_PARAKEET_MODEL)
    run.add_argument("--sortformer-model", default=DEFAULT_SORTFORMER_MODEL)
    run.add_argument("--words-device", default="cuda:0", help="GPU for Parakeet")
    run.add_argument("--turns-device", default="cuda:0", help="GPU for Sortformer")
    run.add_argument("--parallel-models", action="store_true", default=False,
                     help="run the two models in two threads; needs two GPUs "
                          "and is not recommended - see src/config.py")
    run.add_argument("--sequential-models", dest="parallel_models",
                     action="store_false", help="one model at a time (default)")
    run.add_argument("--no-diarize", action="store_true",
                     help="skip speaker labels; words and timings only")
    run.add_argument(
        "--long-audio", choices=("chunk", "local", "none"), default="chunk",
        help="how to handle audio longer than one full-attention pass. "
             "chunk: split and transcribe each piece at full accuracy "
             "(default). local: one pass with local attention, faster but "
             "less accurate. none: hand the whole file over",
    )
    run.add_argument(
        "--chunk-seconds", type=float, default=480.0,
        help="seconds per chunk with --long-audio chunk. Lower it if the job "
             "runs out of GPU memory",
    )

    # output shape
    run.add_argument("--language", default="en")
    run.add_argument("--max-line-width", type=int, default=42)
    run.add_argument("--max-line-count", type=int, default=2)
    run.add_argument("--denoise", action="store_true",
                     help="noise reduction before ASR; off by default because "
                          "both models normalise their own input")

    # stopping
    run.add_argument("--deadline-minutes", type=float, default=0,
                     help="stop starting new interviews this far into the job")

    status = sub.add_parser("status", help="what has been transcribed so far")
    status.add_argument("--output", required=True)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)

    if args.command == "run":
        worker = Worker(args)
        print(json.dumps(worker.run(), indent=2))
        # A worker that stopped because its CUDA context died has not done the
        # work, and the steps after it must not treat that as success.
        return 1 if worker.fatal else 0

    if args.command == "status":
        return _status(args)

    build_parser().print_help()
    return 2


def _load_skip_list(path) -> set:
    """Interview ids already transcribed, per the transcripts repository."""
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
    log.info("sheet lists %d folder(s) to process", len(folders))
    return sources.enumerate_smb(
        server=config.smb_server,
        share=config.smb_share,
        base_path=config.smb_base_path,
        username=os.environ.get("SMB_USERNAME", config.smb_username),
        password=os.environ.get("SMB_PASSWORD", ""),
        folders=folders,
    )


def _status(args) -> int:
    """What is on disk. There is no queue state to report any more."""
    output = Path(args.output)
    done = sorted(p.stem for p in (output / "json").glob("*.json"))
    print(json.dumps({"transcribed": len(done), "ids": done}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
