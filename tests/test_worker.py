"""The queue worker.

Under test: queue filling, claim/complete/fail handling, the wall-clock
deadline that makes a batch survive Slurm's 48-hour cap, and recovery of tasks
orphaned by a killed job. Models are stubbed.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.asr.audio import to_wav_bytes
from src.asr.lines import build_lines
from src.asr.types import Segment, Transcript, Word
from src.asr.workqueue import CLAIMED, DONE, FAILED, PENDING, FileWorkQueue

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "transcribe.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("transcribe", WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transcribe"] = module
    spec.loader.exec_module(module)
    return module


class StubEngine:
    def __init__(self, fail_on=(), delay=0.0):
        self.fail_on = set(fail_on)
        self.delay = delay
        self.seen = []

    def run(self, audio_path, reference_text=None):
        name = Path(audio_path).name
        self.seen.append(name)
        if self.delay:
            time.sleep(self.delay)
        if any(bad in name for bad in self.fail_on):
            raise RuntimeError("CUDA out of memory")
        text = "hello from the archive"
        words = [
            Word(text=t, start=0.3 + i * 0.4, end=0.6 + i * 0.4,
                 confidence=0.9, speaker="S01", segment_index=0)
            for i, t in enumerate(text.split())
        ]
        segments = [Segment(0, words[0].start, words[-1].end, text, "S01", words)]
        transcript = Transcript(audio_path=str(audio_path), duration=3.0, segments=segments)
        transcript.lines = build_lines(segments)
        transcript.meta = {"counts": {"segments": 1, "lines": 1, "words": 4}}
        return transcript


@pytest.fixture
def workspace(tmp_path):
    audio = tmp_path / "input"
    audio.mkdir()
    rate = 16_000
    t = np.arange(2 * rate) / rate
    tone = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    for i in range(1, 5):
        (audio / f"iv_{i:03d}.wav").write_bytes(to_wav_bytes(tone, rate))
    return audio, tmp_path / "queue", tmp_path / "output"


def work_args(mod, queue, output, **overrides):
    argv = ["work", "--queue", str(queue), "--output", str(output)]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv.extend([flag, str(value)])
    return mod.build_parser().parse_args(argv)


# --------------------------------------------------------------------- fill


def test_fill_queues_every_audio_file(mod, workspace):
    audio, queue, output = workspace
    args = mod.build_parser().parse_args(
        ["fill", "--queue", str(queue), "--input", str(audio), "--output", str(output)]
    )
    assert mod._fill(args) == 0
    assert FileWorkQueue(queue).counts()[PENDING] == 4


def test_fill_skips_files_that_already_have_a_transcript(mod, workspace):
    audio, queue, output = workspace
    (output / "json").mkdir(parents=True)
    (output / "json" / "iv_001.json").write_text("{}", encoding="utf-8")

    args = mod.build_parser().parse_args(
        ["fill", "--queue", str(queue), "--input", str(audio), "--output", str(output)]
    )
    mod._fill(args)

    assert FileWorkQueue(queue).counts()[PENDING] == 3


def test_fill_is_idempotent(mod, workspace):
    audio, queue, output = workspace
    args = mod.build_parser().parse_args(
        ["fill", "--queue", str(queue), "--input", str(audio), "--output", str(output)]
    )
    mod._fill(args)
    mod._fill(args)

    assert FileWorkQueue(queue).counts()[PENDING] == 4


def test_queued_payload_carries_an_absolute_path(mod, workspace):
    audio, queue, output = workspace
    args = mod.build_parser().parse_args(
        ["fill", "--queue", str(queue), "--input", str(audio), "--output", str(output)]
    )
    mod._fill(args)

    task = FileWorkQueue(queue).tasks(PENDING)[0]
    assert Path(task.payload["path"]).is_absolute()
    assert Path(task.payload["path"]).exists()


# --------------------------------------------------------------------- work


def fill(mod, audio, queue, output):
    args = mod.build_parser().parse_args(
        ["fill", "--queue", str(queue), "--input", str(audio), "--output", str(output)]
    )
    mod._fill(args)


def test_a_worker_drains_the_queue(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    engine = StubEngine()
    summary = mod.Worker(work_args(mod, queue, output), engine=engine).run()

    assert summary["processed"] == 4
    assert summary["counts"][DONE] == 4
    assert len(list((output / "json").glob("*.json"))) == 4


def test_outputs_are_named_after_the_source_audio(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)
    mod.Worker(work_args(mod, queue, output), engine=StubEngine()).run()

    assert sorted(p.stem for p in (output / "json").glob("*.json")) == [
        f"iv_{i:03d}" for i in range(1, 5)
    ]


def test_several_workers_share_one_queue_without_overlap(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    engines = [StubEngine() for _ in range(3)]
    for index, engine in enumerate(engines):
        mod.Worker(
            work_args(mod, queue, output, worker=f"w{index}"), engine=engine
        ).run()

    processed = [name for e in engines for name in e.seen]
    assert len(processed) == 4
    assert len(set(processed)) == 4


def test_a_failing_file_is_retried_then_parked(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    for _ in range(3):
        mod.Worker(
            work_args(mod, queue, output, max_attempts=3),
            engine=StubEngine(fail_on=("iv_002",)),
        ).run()

    counts = FileWorkQueue(queue).counts()
    assert counts[DONE] == 3
    assert counts[FAILED] == 1


def test_a_missing_source_file_is_not_retried(mod, workspace):
    """It will never succeed, so it must not consume three GPU attempts."""
    audio, queue, output = workspace
    fill(mod, audio, queue, output)
    next(audio.glob("*.wav")).unlink()

    summary = mod.Worker(
        work_args(mod, queue, output, max_attempts=3), engine=StubEngine()
    ).run()

    assert summary["processed"] == 3
    assert summary["failed"] == 1  # one attempt, not three
    assert FileWorkQueue(queue).counts()[FAILED] == 1


def test_failures_record_why(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)
    mod.Worker(
        work_args(mod, queue, output, max_attempts=1),
        engine=StubEngine(fail_on=("iv_",)),
    ).run()

    failed = FileWorkQueue(queue).tasks(FAILED)
    assert len(failed) == 4
    assert all("CUDA out of memory" in t.error for t in failed)


def test_prepared_wavs_are_cleaned_up(mod, workspace, tmp_path):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)
    scratch = tmp_path / "prepared"

    mod.Worker(
        work_args(mod, queue, output, scratch=scratch), engine=StubEngine()
    ).run()

    assert list(scratch.glob("*.wav")) == []


# ----------------------------------------------------- wall clock and restart


def test_the_deadline_stops_the_worker_claiming(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    args = work_args(mod, queue, output)
    worker = mod.Worker(args, engine=StubEngine())
    worker.deadline = time.time() - 1  # already past

    summary = worker.run()
    assert summary["processed"] == 0
    assert FileWorkQueue(queue).counts()[PENDING] == 4  # nothing lost


def test_deadline_minutes_is_applied_with_a_margin(mod, workspace):
    audio, queue, output = workspace
    worker = mod.Worker(
        work_args(mod, queue, output, deadline_minutes=60), engine=StubEngine()
    )
    remaining = worker.deadline - time.time()

    assert remaining == pytest.approx(3600 - mod.DEADLINE_MARGIN_SECONDS, abs=5)


def test_a_signal_winds_the_worker_down(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    worker = mod.Worker(work_args(mod, queue, output), engine=StubEngine())
    worker._stop = True

    assert worker.run()["processed"] == 0


def test_work_orphaned_by_a_killed_job_is_recovered(mod, workspace):
    """The 48-hour scenario: job 1 is killed holding a claim, job 2 finishes it."""
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    # Job 1 claims one task with a lease that dies with it, then vanishes.
    q = FileWorkQueue(queue)
    q.claim(worker="job1", lease=-1)
    assert q.counts()[CLAIMED] == 1

    # Job 2 starts: the worker reaps on startup, then drains everything.
    summary = mod.Worker(
        work_args(mod, queue, output, worker="job2"), engine=StubEngine()
    ).run()

    assert summary["processed"] == 4
    assert q.counts()[DONE] == 4


def test_a_live_claim_from_another_worker_is_left_alone(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    q = FileWorkQueue(queue)
    q.claim(worker="other", lease=3600)  # still working on it

    summary = mod.Worker(
        work_args(mod, queue, output), engine=StubEngine()
    ).run()

    assert summary["processed"] == 3  # the fourth belongs to "other"
    assert q.counts()[CLAIMED] == 1


# --------------------------------------------------------------- CLI surface


def test_status_reports_counts(mod, workspace, capsys):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)

    capsys.readouterr()  # drop what `fill` printed
    args = mod.build_parser().parse_args(["status", "--queue", str(queue)])
    mod._status(args)

    assert json.loads(capsys.readouterr().out)[PENDING] == 4


def test_requeue_moves_failed_tasks_back(mod, workspace):
    audio, queue, output = workspace
    fill(mod, audio, queue, output)
    mod.Worker(
        work_args(mod, queue, output, max_attempts=1),
        engine=StubEngine(fail_on=("iv_",)),
    ).run()

    assert mod.main(["requeue", "--queue", str(queue)]) == 0
    assert FileWorkQueue(queue).counts()[PENDING] == 4





def test_hybrid_runs_the_two_models_in_parallel_on_two_gpus(mod, workspace):
    """A Grace node has two A100s, so pin one model to each and overlap them."""
    audio, queue, output = workspace
    args = work_args(mod, queue, output)
    engine = mod.build_engine(args)

    assert engine.config.diarize is True
    assert engine.config.parallel is True
    assert engine.config.parakeet.device == "cuda:0"
    assert engine.config.sortformer.device == "cuda:1"
    assert engine.config.devices_differ is True   # so the overlap is real


def test_pinning_both_models_to_one_gpu_is_not_a_real_overlap(mod, workspace):
    """Two threads on one device contend; the config should say so."""
    audio, queue, output = workspace
    args = work_args(mod, queue, output, turns_device="cuda:0")

    assert mod.build_engine(args).config.devices_differ is False


def test_models_can_be_forced_sequential(mod, workspace):
    audio, queue, output = workspace
    argv = ["work", "--queue", str(queue), "--output", str(output),
            "--sequential-models"]
    args = mod.build_parser().parse_args(argv)

    assert mod.build_engine(args).config.parallel is False


def test_sortformer_v2_1_is_the_default_diarizer(mod, workspace):
    """It shares Parakeet's frame grid and its NeMo environment."""
    audio, queue, output = workspace
    args = work_args(mod, queue, output)

    assert args.sortformer_model.endswith("v2.1")
    assert mod.build_engine(args).config.diarize is True


def test_diarization_can_be_switched_off(mod, workspace):
    audio, queue, output = workspace
    argv = ["work", "--queue", str(queue), "--output", str(output), "--no-diarize"]
    args = mod.build_parser().parse_args(argv)

    assert mod.build_engine(args).config.diarize is False


def test_denoise_is_off_by_default(mod, workspace):
    """The models normalise internally; spectral gating is opt-in."""
    audio, queue, output = workspace
    assert work_args(mod, queue, output).denoise is False


def test_exit_when_empty_is_the_default(mod, workspace):
    audio, queue, output = workspace
    assert work_args(mod, queue, output).exit_when_empty is True


def test_stay_alive_flips_it(mod, workspace):
    audio, queue, output = workspace
    args = mod.build_parser().parse_args(
        ["work", "--queue", str(queue), "--output", str(output), "--stay-alive"]
    )
    assert args.exit_when_empty is False
