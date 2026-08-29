"""Submit the pipeline to Slurm.

    python scripts/run_pipeline.py               # from the tracking spreadsheet
    python scripts/run_pipeline.py --from-json   # from the reviewer app's list

Run this from a login node. It submits three jobs and exits - nothing is
downloaded, read or transcribed here:

    prepare  ->  transcribe (GPU array)  ->  publish

Each job starts only if the one before it succeeded, so the ordering costs
nothing on the login node. Watch them with `squeue -u $USER`.

Safe to re-run. Interviews that already have a transcript are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.pipeline import TranscriptionPipeline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=("smb", "json", "local"), default=None,
        help="where the audio is. smb: the file share (default). json: the "
             "reviewer repo's list. local: already staged on scratch",
    )
    parser.add_argument(
        "--input", default=None,
        help="folder to read, with --source local",
    )
    parser.add_argument(
        "--from-json", action="store_true",
        help="shorthand for --source json",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="leave the transcripts on disk instead of pushing them to GitHub",
    )
    parser.add_argument(
        "--no-diarize", action="store_true",
        help="skip speaker labels; produce words and timings only",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="stay running until the jobs finish, printing their status. Off "
             "by default - the jobs are chained by Slurm and do not need it",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="only process this many interviews - useful for a first test run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = get_config()
    if args.source:
        config.source = args.source
        config.from_json = args.source == "json"
    if args.from_json:
        config.source = "json"
        config.from_json = True
    if args.input:
        config.input_dir = args.input
    if args.no_diarize:
        config.diarize = False
    if args.max_files is not None:
        config.max_files = args.max_files

    return TranscriptionPipeline(skip_upload=args.skip_upload, wait=args.wait).run()


if __name__ == "__main__":
    sys.exit(main())
