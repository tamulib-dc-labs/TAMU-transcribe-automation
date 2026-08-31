"""Start the pipeline. Run this from a login node.

    python scripts/run_pipeline.py               # from the tracking spreadsheet
    python scripts/run_pipeline.py --from-json   # from the reviewer app's list

It submits one Slurm job and exits. That job does everything - lists the work,
downloads each recording as it gets to it, transcribes it, and uploads the
results. Nothing is computed here; `sbatch` is the only thing a login node is
used for.

Safe to re-run. Interviews that already have a transcript are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cli import apply_to_config, build_parser, to_job_args  # noqa: E402
from src.config import get_config  # noqa: E402
from src.pipeline import TranscriptionPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--wait", action="store_true",
        help="stay running until the job finishes, printing its status",
    )
    args = parser.parse_args(argv)

    config = get_config()
    apply_to_config(args, config)

    # The job reads config.py for itself, so only what was given on the command
    # line has to travel with it.
    return TranscriptionPipeline(
        skip_upload=args.skip_upload,
        wait=args.wait,
        job_args=to_job_args(args),
    ).submit()


if __name__ == "__main__":
    sys.exit(main())
