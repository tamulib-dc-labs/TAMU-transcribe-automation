"""Fill the work queue. Runs as a Slurm job, not on a login node.

Does everything that has to happen before the GPUs start:

  1. read the work list (from the reviewer repo, in --from-json mode)
  2. list interviews already transcribed in the transcripts repository
  3. enumerate the source into the queue

All of it is network and filesystem work, no GPU. Compute nodes reach GitHub
through WebProxy, and credentials come from config/local_settings.py on the
shared filesystem, so no token is passed through the job environment.

One task, deliberately: cloning the reviewer repo from four array tasks at once
would race on the same directory. The GPU job depends on this one.
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

    pipeline._prepare_directories()

    if config.resolved_source == "json" and not pipeline._prepare_config_repo():
        Logger.log_warning("No work list; the queue will be left as it is.")
        return 0

    pipeline._collect_completed()

    import subprocess

    command = [
        sys.executable, config.transcribe_script_path, "fill",
        "--queue", config.queue_path,
        "--output", config.oral_output_path,
        "--source", config.resolved_source,
        "--skip-list", config.completed_list_path,
    ]
    if config.resolved_source == "json":
        command += ["--config-json", config.work_list_path]
    else:
        if config.resolved_source == "local":
            command += ["--input", config.resolved_input_dir]
        if config.max_files:
            command += ["--max-files", str(config.max_files)]

    return subprocess.run(command).returncode


if __name__ == "__main__":
    sys.exit(main())
