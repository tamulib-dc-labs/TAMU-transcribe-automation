"""Fusing Parakeet words with Sortformer speaker turns by time overlap."""

import pytest

from src.asr.fusion import (
    SpeakerTurn,
    assign_speakers,
    build_segments,
    fuse,
    turns_from_segments,
)
from src.asr.types import Segment, Word


def w(text, start, end, score=0.9):
    return Word(text=text, start=start, end=end, confidence=score)


def turn(start, end, speaker):
    return SpeakerTurn(start=start, end=end, speaker=speaker)


# ------------------------------------------------------------ turn extraction


def test_turns_are_taken_from_segments_ignoring_their_text():
    segments = [
        Segment(0, 0.0, 2.0, "discarded text", "S01"),
        Segment(1, 2.5, 5.0, "and this", "S02"),
    ]
    turns = turns_from_segments(segments)

    assert [(t.start, t.end, t.speaker) for t in turns] == [
        (0.0, 2.0, "S01"),
        (2.5, 5.0, "S02"),
    ]


def test_segments_without_a_speaker_or_duration_are_not_turns():
    segments = [
        Segment(0, 0.0, 2.0, "no speaker", None),
        Segment(1, 3.0, 3.0, "zero length", "S01"),
        Segment(2, 4.0, 5.0, "good", "S01"),
    ]
    assert len(turns_from_segments(segments)) == 1


# --------------------------------------------------------- speaker assignment


def test_words_take_the_speaker_of_the_turn_they_sit_in():
    words = [w("hello", 0.1, 0.6), w("there", 0.7, 1.2), w("hi", 3.0, 3.4)]
    turns = [turn(0.0, 2.0, "S01"), turn(2.5, 4.0, "S02")]

    assignments, stats = assign_speakers(words, turns)

    assert [x.speaker for x in words] == ["S01", "S01", "S02"]
    assert assignments == [0, 0, 1]
    assert stats.words_assigned == 3
    assert stats.words_unassigned == 0


def test_a_straddling_word_goes_to_the_turn_it_overlaps_most():
    # 0.4s inside S01, 0.6s inside S02
    words = [w("boundary", 1.6, 2.6)]
    turns = [turn(0.0, 2.0, "S01"), turn(2.0, 4.0, "S02")]

    assign_speakers(words, turns)
    assert words[0].speaker == "S02"


def test_a_word_in_a_diarization_gap_takes_the_nearest_turn():
    words = [w("orphan", 2.1, 2.3)]
    turns = [turn(0.0, 2.0, "S01"), turn(3.0, 4.0, "S02")]

    _, stats = assign_speakers(words, turns, max_gap=1.0)

    assert words[0].speaker == "S01"  # 0.1s away vs 0.7s
    assert stats.words_by_nearest == 1
    assert stats.words_assigned == 1


def test_a_word_far_from_every_turn_is_left_unassigned():
    words = [w("stray", 50.0, 50.5)]
    turns = [turn(0.0, 2.0, "S01")]

    assignments, stats = assign_speakers(words, turns, max_gap=1.0)

    assert words[0].speaker is None
    assert assignments == [None]
    assert stats.words_unassigned == 1


def test_no_turns_at_all_leaves_every_word_unassigned():
    words = [w("a", 0.0, 1.0), w("b", 1.0, 2.0)]
    assignments, stats = assign_speakers(words, [])

    assert assignments == [None, None]
    assert stats.words_unassigned == 2
    assert stats.as_dict()["assigned_fraction"] == 0.0


def test_no_words_is_handled():
    assignments, stats = assign_speakers([], [turn(0.0, 1.0, "S01")])
    assert assignments == []
    assert stats.words_total == 0


def test_unordered_turns_are_handled():
    words = [w("late", 3.1, 3.5), w("early", 0.1, 0.5)]
    turns = [turn(3.0, 4.0, "S02"), turn(0.0, 1.0, "S01")]  # out of order

    assign_speakers(sorted(words, key=lambda x: x.start), turns)
    assert {x.text: x.speaker for x in words} == {"early": "S01", "late": "S02"}


def test_stats_report_turns_that_received_no_words():
    words = [w("only", 0.1, 0.5)]
    turns = [turn(0.0, 1.0, "S01"), turn(5.0, 6.0, "S02"), turn(9.0, 9.5, "S03")]

    _, stats = assign_speakers(words, turns, max_gap=0.1)

    assert stats.turns_empty == 2
    assert stats.speakers == ["S01", "S02", "S03"]


def test_mean_overlap_fraction_reflects_assignment_quality():
    words = [w("clean", 0.2, 0.8)]  # fully inside the turn
    _, stats = assign_speakers(words, [turn(0.0, 2.0, "S01")])
    assert stats.mean_overlap_fraction == pytest.approx(1.0)


def test_scale_many_words_and_turns():
    words = [w(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(400)]
    turns = [
        turn(i * 10.0, i * 10.0 + 10.0, f"S{i % 2 + 1:02d}") for i in range(20)
    ]
    _, stats = assign_speakers(words, turns)

    assert stats.words_assigned == 400
    assert stats.speakers == ["S01", "S02"]


# -------------------------------------------------------------- segment build


def test_segments_mirror_the_turn_structure():
    words = [w("hello", 0.1, 0.6), w("there", 0.7, 1.2), w("hi", 3.0, 3.4)]
    turns = [turn(0.0, 2.0, "S01"), turn(2.5, 4.0, "S02")]

    segments, _ = fuse(words, turns)

    assert len(segments) == 2
    assert segments[0].speaker == "S01"
    assert segments[0].text == "hello there"
    assert segments[1].text == "hi"


def test_segment_times_come_from_words_not_turns():
    words = [w("hello", 0.5, 0.9)]
    turns = [turn(0.0, 30.0, "S01")]  # a very loose turn

    segments, _ = fuse(words, turns)

    assert segments[0].start == 0.5
    assert segments[0].end == 0.9


def test_the_same_speaker_returning_makes_a_new_segment():
    words = [w("a", 0.1, 0.5), w("b", 3.1, 3.5), w("c", 6.1, 6.5)]
    turns = [turn(0.0, 1.0, "S01"), turn(3.0, 4.0, "S02"), turn(6.0, 7.0, "S01")]

    segments, _ = fuse(words, turns)

    assert [s.speaker for s in segments] == ["S01", "S02", "S01"]
    assert len(segments) == 3


def test_words_keep_their_confidence_through_fusion():
    words = [w("kept", 0.1, 0.5, score=0.42)]
    segments, _ = fuse(words, [turn(0.0, 1.0, "S01")])

    assert segments[0].words[0].confidence == 0.42


def test_segment_index_is_written_back_onto_words():
    words = [w("a", 0.1, 0.5), w("b", 3.1, 3.5)]
    turns = [turn(0.0, 1.0, "S01"), turn(3.0, 4.0, "S02")]

    segments, _ = fuse(words, turns)

    assert [x.segment_index for x in words] == [0, 1]
    assert [s.index for s in segments] == [0, 1]


def test_unassigned_words_form_their_own_speakerless_segment():
    words = [w("known", 0.1, 0.5), w("stray", 50.0, 50.4)]
    turns = [turn(0.0, 1.0, "S01")]

    segments, _ = fuse(words, turns, max_gap=1.0)

    assert segments[0].speaker == "S01"
    assert segments[1].speaker is None
    assert segments[1].text == "stray"


def test_build_segments_on_empty_input():
    assert build_segments([], [], []) == []


def test_cjk_words_are_rejoined_without_spaces():
    words = [w("你", 0.1, 0.3), w("好", 0.3, 0.5)]
    segments, _ = fuse(words, [turn(0.0, 1.0, "S01")])

    assert segments[0].text == "你好"
