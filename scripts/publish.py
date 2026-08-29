"""Upload the transcripts. Runs as a Slurm job, not on a login node.

Pushes data/oral_output to the transcripts repository and, in --from-json mode,
writes the resulting links back into the reviewer repo's config.json.

Depends on the transcription job, so it starts once the GPUs are done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_config  # noqa: E402
from src.pipeline import TranscriptionPipeline  # noqa: E402
from src.utils.logger import Logger  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    config = get_config()
    pipeline = TranscriptionPipeline()

    outputs = Path(config.oral_output_path) / "json"
    if not outputs.exists() or not any(outputs.glob("*.json")):
        Logger.log_warning("No transcripts to upload.")
        return 0

    pipeline._upload_to_github()

    if config.resolved_source == "json":
        # The reviewer repo and the entries it needs are re-read here, because
        # this runs in a different job from the one that queued the work.
        if pipeline._prepare_config_repo():
            pipeline._update_config_repo()

    return 0


if __name__ == "__main__":
    sys.exit(main())
