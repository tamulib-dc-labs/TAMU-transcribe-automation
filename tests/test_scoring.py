"""Reference-based word accuracy."""

import pytest

from src.asr.scoring import levenshtein_ops, score_transcript
from src.asr.types import STATUS_CORRECT, STATUS_INSERTION, STATUS_SUBSTITUTION
from src.asr.types import Segment, Transcript, Word


def make_transcript(words, confidences=None):
    confidences = confidences or [None] * len(words)
    built = [
        Word(text=w, start=float(i), end=float(i) + 1, confidence=c, segment_index=0)
        for i, (w, c) in enumerate(zip(words, confidences))
    ]
    segment = Segment(
        index=0, start=0.0, end=float(len(words)), text=" ".join(words), words=built
    )
    return Transcript(audio_path="x.wav", duration=float(len(words)), segments=[segment])


def test_perfect_match():
    transcript = make_transcript(["the", "cat", "sat"])
    metrics = score_transcript(transcript, "The cat sat.")

    assert metrics["wer"] == 0.0
    assert metrics["word_accuracy"] == 1.0
    assert metrics["cer"] == 0.0
    assert all(w.status == STATUS_CORRECT for w in transcript.words)


def test_substitution_is_counted():
    transcript = make_transcript(["the", "hat", "sat"])
    metrics = score_transcript(transcript, "the cat sat")

    assert (metrics["substitutions"], metrics["deletions"], metrics["insertions"]) == (
        1,
        0,
        0,
    )
    assert metrics["wer"] == pytest.approx(1 / 3, abs=1e-4)


def test_deletion_is_counted():
    transcript = make_transcript(["the", "sat"])
    metrics = score_transcript(transcript, "the cat sat")

    assert (metrics["substitutions"], metrics["deletions"], metrics["insertions"]) == (
        0,
        1,
        0,
    )
    assert metrics["wer"] == pytest.approx(1 / 3, abs=1e-4)


def test_insertion_is_counted():
    transcript = make_transcript(["the", "big", "cat", "sat"])
    metrics = score_transcript(transcript, "the cat sat")

    assert (metrics["substitutions"], metrics["deletions"], metrics["insertions"]) == (
        0,
        0,
        1,
    )
    assert metrics["wer"] == pytest.approx(1 / 3, abs=1e-4)
    assert metrics["word_accuracy"] == 1.0  # every reference word was recovered


def test_words_are_tagged_with_their_verdict():
    transcript = make_transcript(["the", "hat", "sat", "extra"])
    score_transcript(transcript, "the cat sat")

    statuses = [w.status for w in transcript.words]
    assert statuses == [
        STATUS_CORRECT,
        STATUS_SUBSTITUTION,
        STATUS_CORRECT,
        STATUS_INSERTION,
    ]
    assert transcript.words[1].reference == "cat"
    assert transcript.words[3].reference is None


def test_scoring_ignores_case_and_punctuation():
    transcript = make_transcript(["The,", "CAT!"])
    assert score_transcript(transcript, "the cat")["wer"] == 0.0


def test_confidence_is_split_by_correctness():
    transcript = make_transcript(["the", "hat"], confidences=[0.9, 0.2])
    metrics = score_transcript(transcript, "the cat")

    assert metrics["mean_confidence_correct"] == 0.9
    assert metrics["mean_confidence_incorrect"] == 0.2
    assert metrics["mean_confidence"] == pytest.approx(0.55)


def test_metrics_are_stored_on_the_transcript():
    transcript = make_transcript(["a"])
    metrics = score_transcript(transcript, "a")
    assert transcript.accuracy is metrics


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError, match="no scorable words"):
        score_transcript(make_transcript(["a"]), "  ...  ")


def test_levenshtein_ops_reconstructs_the_reference():
    ops = levenshtein_ops(["a", "b", "c"], ["a", "x", "c", "d"])
    kinds = [op.kind for op in ops]
    assert kinds.count("match") == 2
    assert "substitution" in kinds
    assert "insertion" in kinds
