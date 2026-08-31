"""The command-line options, shared by the two entry points.

scripts/run_pipeline.py parses these on a login node and passes them straight
through to the job; scripts/run_job.py parses the same flags inside the job and
applies them. They live here so the two cannot drift apart - an option that
worked on the login node but was silently dropped on the way to the job would
be invisible until someone checked the output.
"""

from __future__ import annotations

import argparse
import shlex
from typing import Any, List


def build_parser(description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
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
        "--max-files", type=int, default=None,
        help="only process this many interviews - useful for a first test run",
    )
    return parser


def apply_to_config(args: argparse.Namespace, config: Any) -> None:
    """Fold the command-line options into the settings object."""
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


def to_job_args(args: argparse.Namespace) -> List[str]:
    """Rebuild the options as a command line for the job to parse.

    Only what was actually asked for, so the job falls back to config.py for
    everything else.
    """
    out: List[str] = []
    if args.source:
        out += ["--source", args.source]
    if args.input:
        out += ["--input", args.input]
    if args.from_json:
        out.append("--from-json")
    if args.skip_upload:
        out.append("--skip-upload")
    if args.no_diarize:
        out.append("--no-diarize")
    if args.max_files is not None:
        out += ["--max-files", str(args.max_files)]
    return out


def quote(job_args: List[str]) -> str:
    """Render the options for the shell line inside run.slurm."""
    return " ".join(shlex.quote(a) for a in job_args)
