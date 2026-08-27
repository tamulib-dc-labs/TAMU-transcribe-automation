"""Entry point for the full pipeline: download -> transcribe -> upload.

    python scripts/run_pipeline.py              # SMB mode
    python scripts/run_pipeline.py --from-json  # reviewer-repo JSON mode

Runs from a login node. It downloads audio, makes sure the models are cached,
submits the Slurm job, waits for it, then uploads the transcripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.pipeline import TranscriptionPipeline  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-json", action="store_true",
                        help="take the file list from the reviewer repo's config JSON")
    parser.add_argument("--skip-download", action="store_true",
                        help="audio is already in place")
    parser.add_argument("--skip-upload", action="store_true",
                        help="leave transcripts on disk")
    parser.add_argument("--engine", choices=("hybrid", "words-only"), default=None,
                        help="hybrid adds speaker labels (default); words-only skips them")
    args = parser.parse_args(argv)

    config = get_config()
    if args.from_json:
        config.from_json = True
    if args.engine:
        config.diarize = args.engine == "hybrid"

    return TranscriptionPipeline(
        skip_download=args.skip_download,
        skip_upload=args.skip_upload,
    ).run()


if __name__ == "__main__":
    sys.exit(main())
