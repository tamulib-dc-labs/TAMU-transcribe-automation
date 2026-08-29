"""Run the whole pipeline: submit the Slurm job, wait, upload the transcripts.

    python scripts/run_pipeline.py               # from the tracking spreadsheet
    python scripts/run_pipeline.py --from-json   # from the reviewer app's list

Run this from a login node. Nothing is downloaded here - the Slurm job lists
the collection, fetches each interview as it works on it, and transcribes it.

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
        "--from-json", action="store_true",
        help="take the file list from the reviewer repo's config JSON "
             "instead of the tracking spreadsheet",
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
        "--max-files", type=int, default=None,
        help="only process this many interviews - useful for a first test run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config = get_config()
    if args.from_json:
        config.from_json = True
    if args.no_diarize:
        config.diarize = False
    if args.max_files is not None:
        config.max_files = args.max_files

    return TranscriptionPipeline(skip_upload=args.skip_upload).run()


if __name__ == "__main__":
    sys.exit(main())
