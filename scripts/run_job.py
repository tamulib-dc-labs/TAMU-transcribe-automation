"""The pipeline itself. Runs inside the Slurm job, never on a login node.

Started by config/run.slurm. Does everything in one process:

    1. list the work   - read the work list, skip what the transcripts repo
                         already has
    2. transcribe      - Parakeet-TDT for words, Sortformer for speaker turns
    3. upload          - push the json and vtt files, then write their links
                         into the reviewer app's config.json

Settings come from src/config.py, which this imports directly. The only things
passed in are the options someone typed on the command line.

You would not normally run this by hand. To start the pipeline, use
scripts/run_pipeline.py from a login node; it submits the job that runs this.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import apply_to_config, build_parser  # noqa: E402
from src.config import get_config  # noqa: E402
from src.pipeline import TranscriptionPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = build_parser(__doc__).parse_args(argv)
    apply_to_config(args, get_config())

    return TranscriptionPipeline(skip_upload=args.skip_upload).execute()


if __name__ == "__main__":
    sys.exit(main())
