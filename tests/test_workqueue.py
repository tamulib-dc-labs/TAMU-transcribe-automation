"""The shared-filesystem work queue.

What has to hold: no task is ever handed to two workers, nothing is lost when a
worker is killed mid-task, and a batch survives being cut off by a scheduler
wall clock and resumed by a later job.
"""

import json
import threading
import time

import pytest

from src.asr.workqueue import (
    CLAIMED,
    DONE,
    FAILED,
    PENDING,
    FileWorkQueue,
    worker_id,
)


@pytest.fixture
def queue(tmp_path):
    return FileWorkQueue(tmp_path / "queue")


def payloads(n, prefix="iv"):
    return [{"id": f"{prefix}_{i:03d}", "path": f"/data/{prefix}_{i:03d}.wav"} for i in range(n)]


# ------------------------------------------------------------------ submitting


def test_submit_creates_pending_tasks(queue):
    assert queue.submit(payloads(5)) == 5
    assert queue.counts()[PENDING] == 5


def test_submitting_the_same_id_twice_is_idempotent(queue):
    queue.submit(payloads(3))
    assert queue.submit(payloads(3)) == 0
    assert queue.counts()[PENDING] == 3


def test_resubmitting_a_finished_task_does_not_redo_it(queue):
    queue.submit(payloads(1))
    queue.complete(queue.claim())

    assert queue.submit(payloads(1)) == 0
    assert queue.counts()[DONE] == 1
    assert queue.counts()[PENDING] == 0


def test_ids_with_awkward_characters_are_slugged(queue):
    queue.submit([{"id": "a/b c:d*.wav"}])
    assert queue.counts()[PENDING] == 1
    assert queue.claim() is not None


# -------------------------------------------------------------------- claiming


def test_claim_returns_a_task_and_moves_it_out_of_pending(queue):
    queue.submit(payloads(2))
    task = queue.claim(worker="w1")

    assert task is not None
    assert task.worker == "w1"
    assert task.attempts == 1
    assert queue.counts() == {PENDING: 1, CLAIMED: 1, DONE: 0, FAILED: 0}


def test_claim_on_an_empty_queue_returns_none(queue):
    assert queue.claim() is None


def test_workers_never_receive_the_same_task_twice(queue):
    queue.submit(payloads(60))
    claimed = []
    errors = []
    lock = threading.Lock()

    def drain(name):
        try:
            while True:
                task = queue.claim(worker=name)
                if task is None:
                    return
                with lock:
                    claimed.append(task.id)
                queue.complete(task)
        except Exception as exc:  # a worker dying here loses tasks silently
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=drain, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(claimed) == 60
    assert len(set(claimed)) == 60  # every task handed out exactly once
    assert queue.counts()[DONE] == 60


def test_a_lease_is_recorded_on_claim(queue):
    queue.submit(payloads(1))
    task = queue.claim(lease=120)
    assert task.leased_until > time.time() + 100


# ------------------------------------------------------------------ finishing


def test_complete_moves_the_task_to_done(queue):
    queue.submit(payloads(1))
    queue.complete(queue.claim())

    assert queue.counts() == {PENDING: 0, CLAIMED: 0, DONE: 1, FAILED: 0}
    assert queue.is_drained()


def test_failure_requeues_until_max_attempts(queue):
    queue = FileWorkQueue(queue.root, max_attempts=3)
    queue.submit(payloads(1))

    assert queue.fail(queue.claim(), "boom") == PENDING
    assert queue.fail(queue.claim(), "boom") == PENDING
    assert queue.fail(queue.claim(), "boom") == FAILED
    assert queue.counts()[FAILED] == 1


def test_failure_records_the_error_and_history(queue):
    queue.submit(payloads(1))
    queue.fail(queue.claim(), "CUDA out of memory", retry=False)

    failed = queue.tasks(FAILED)[0]
    assert "CUDA out of memory" in failed.error
    assert any("failed" in line for line in failed.history)


def test_retry_false_fails_immediately(queue):
    queue.submit(payloads(1))
    assert queue.fail(queue.claim(), "bad file", retry=False) == FAILED


def test_requeue_failed_resets_the_attempt_count(queue):
    queue = FileWorkQueue(queue.root, max_attempts=1)
    queue.submit(payloads(2))
    queue.fail(queue.claim(), "x")
    queue.fail(queue.claim(), "x")
    assert queue.counts()[FAILED] == 2

    assert queue.requeue_failed() == 2
    assert queue.counts()[PENDING] == 2
    assert queue.claim().attempts == 1  # counted from zero again


# --------------------------------------------------------------------- reaping


def test_an_expired_lease_is_returned_to_pending(queue):
    queue.submit(payloads(1))
    queue.claim(lease=-1)  # already expired: stands in for a killed worker

    assert queue.reap() == 1
    assert queue.counts()[PENDING] == 1


def test_a_live_lease_is_not_reaped(queue):
    queue.submit(payloads(1))
    queue.claim(lease=600)

    assert queue.reap() == 0
    assert queue.counts()[CLAIMED] == 1


def test_heartbeat_extends_a_lease_past_the_reaper(queue):
    queue.submit(payloads(1))
    task = queue.claim(lease=-1)
    queue.heartbeat(task, lease=600)

    assert queue.reap() == 0


def test_reaping_respects_max_attempts(queue):
    queue = FileWorkQueue(queue.root, max_attempts=1)
    queue.submit(payloads(1))
    queue.claim(lease=-1)

    queue.reap()
    assert queue.counts()[FAILED] == 1  # not retried forever


def test_a_batch_survives_a_job_being_killed(queue):
    """The wall-clock scenario: job 1 dies mid-flight, job 2 picks it up."""
    queue.submit(payloads(5))

    # Job 1 finishes two and is killed holding three.
    for _ in range(2):
        queue.complete(queue.claim(worker="job1", lease=3600))
    for _ in range(3):
        queue.claim(worker="job1", lease=-1)  # leases die with the job
    assert queue.counts() == {PENDING: 0, CLAIMED: 3, DONE: 2, FAILED: 0}

    # Job 2 starts, reaps, and drains the rest.
    assert queue.reap() == 3
    while True:
        task = queue.claim(worker="job2")
        if task is None:
            break
        queue.complete(task)

    assert queue.counts()[DONE] == 5
    assert queue.is_drained()


# ---------------------------------------------------------------------- status


def test_is_drained_only_when_nothing_is_pending_or_in_flight(queue):
    queue.submit(payloads(2))
    assert not queue.is_drained()

    task = queue.claim()
    assert not queue.is_drained()  # still in flight

    queue.complete(task)
    queue.complete(queue.claim())
    assert queue.is_drained()


def test_tasks_can_be_listed_per_state(queue):
    queue.submit(payloads(3))
    queue.complete(queue.claim())

    assert len(queue.tasks(PENDING)) == 2
    assert len(queue.tasks(DONE)) == 1
    assert queue.tasks(DONE)[0].payload["path"].endswith(".wav")


def test_listing_an_unknown_state_is_rejected(queue):
    with pytest.raises(ValueError, match="unknown state"):
        queue.tasks("elsewhere")


def test_payload_survives_the_round_trip(queue):
    queue.submit([{"id": "x", "path": "/a/b.wav", "extra": {"n": 1}}])
    assert queue.claim().payload["extra"] == {"n": 1}


# ----------------------------------------------------------------- robustness


def test_a_corrupt_payload_still_yields_a_claimable_task(queue):
    """Ownership is in the filename, so an unreadable payload is not fatal.

    The queue hands the task over with an empty payload and lets the worker
    decide - parking a perfectly good interview over a bad metadata read
    would be the queue overstepping.
    """
    queue.submit(payloads(2))
    corrupt = next((queue.root / PENDING).glob("*.json"))
    corrupt.write_text("{not json", encoding="utf-8")

    claimed = []
    while True:
        task = queue.claim()
        if task is None:
            break
        claimed.append(task)
        queue.complete(task)

    assert len(claimed) == 2
    assert any(t.payload == {} for t in claimed)   # the corrupt one
    assert any(t.payload.get("path") for t in claimed)  # the good one


def test_writes_are_atomic_so_no_partial_file_is_ever_claimable(queue):
    queue.submit(payloads(3))
    for path in (queue.root / PENDING).glob("*"):
        assert path.suffix == ".json"  # no .tmp left behind
        json.loads(path.read_text(encoding="utf-8"))


def test_two_queue_objects_share_one_directory(tmp_path):
    a = FileWorkQueue(tmp_path / "q")
    b = FileWorkQueue(tmp_path / "q")
    a.submit(payloads(2))

    assert b.claim() is not None
    assert a.counts()[CLAIMED] == 1


def test_worker_id_is_traceable(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_PROCID", "3")
    identity = worker_id()

    assert "12345" in identity
    assert identity.count("-") >= 2


def test_ownership_is_encoded_in_the_filename_not_the_file():
    """The claim is one rename; nothing is written to establish ownership."""
    from src.asr.workqueue import SEP, _decode, _encode

    name = _encode("iv_001", 2, "node5-pid7", 1234567890)
    assert name.count(SEP) == 3
    assert _decode(name) == ("iv_001", 2, "node5-pid7", 1234567890.0)


def test_a_malformed_filename_is_quarantined_not_claimed(queue):
    (queue.root / PENDING / "garbage.json").write_text("{}", encoding="utf-8")
    assert queue.claim() is None
    assert queue.counts()[FAILED] == 1


def test_an_orphaned_claim_token_is_swept(queue):
    """A worker killed between taking the token and moving the payload."""
    import os
    import time as _time

    queue.submit(payloads(1))
    task_id = queue.tasks(PENDING)[0].id
    token = queue.root / CLAIMED / f".{task_id}.claim"
    token.write_text("stale", encoding="utf-8")
    os.utime(token, (_time.time() - 3600, _time.time() - 3600))

    assert queue.claim() is None      # blocked while the token stands
    queue.reap()                      # reap sweeps orphaned tokens
    assert queue.claim() is not None  # unblocked


def test_a_fresh_token_is_not_swept(queue):
    queue.submit(payloads(1))
    task = queue.claim()
    queue.reap()
    assert queue.counts()[CLAIMED] == 1
    assert task is not None
