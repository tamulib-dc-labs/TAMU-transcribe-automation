"""Re-flowing words into subtitle lines with word-derived timestamps."""

from src.asr.lines import LineConfig, build_lines
from src.asr.types import Segment, Word


def make_segment(words, start=0.0, step=0.5, speaker="S01", gaps=None):
    gaps = gaps or {}
    built, cursor = [], start
    for index, text in enumerate(words):
        cursor += gaps.get(index, 0.0)
        built.append(
            Word(text=text, start=cursor, end=cursor + step, speaker=speaker, segment_index=0)
        )
        cursor += step
    return Segment(
        index=0,
        start=start,
        end=built[-1].end,
        text=" ".join(words),
        speaker=speaker,
        words=built,
    )


def test_line_times_come_from_its_words_not_the_segment():
    segment = make_segment(["alpha", "beta"], start=10.0)
    segment.start, segment.end = 0.0, 99.0  # deliberately wrong turn boundaries

    lines = build_lines([segment], LineConfig())
    assert lines[0].start == 10.0
    assert lines[0].end == 11.0


def test_breaks_on_max_chars():
    words = ["word"] * 20
    lines = build_lines([make_segment(words)], LineConfig(max_chars=20, max_duration=999))
    assert len(lines) > 1
    assert all(len(line.text) <= 20 for line in lines)


def test_breaks_on_a_long_silence():
    segment = make_segment(["a", "b", "c"], gaps={2: 5.0})
    lines = build_lines([segment], LineConfig(max_gap=0.8, min_duration=0.0))
    assert len(lines) == 2
    assert lines[1].text == "c"


def test_breaks_after_sentence_punctuation():
    segment = make_segment(["This", "is", "a", "sentence.", "Another", "one", "here"])
    lines = build_lines(
        [segment], LineConfig(max_chars=999, max_duration=999, min_chars_for_sentence_break=5)
    )
    assert len(lines) == 2
    assert lines[0].text.endswith("sentence.")


def test_breaks_on_max_duration():
    segment = make_segment(["a", "b", "c", "d"], step=2.0)
    lines = build_lines([segment], LineConfig(max_duration=3.0, max_chars=999))
    assert len(lines) > 1
    assert all(line.duration <= 3.0 + 1e-6 for line in lines)


def test_lines_never_cross_a_speaker_change():
    first = make_segment(["hi"], start=0.0, speaker="S01")
    second = make_segment(["there"], start=0.6, speaker="S02")
    second.index = 1
    for word in second.words:
        word.segment_index = 1

    lines = build_lines([first, second], LineConfig(max_chars=999, max_duration=999))
    assert [line.speaker for line in lines] == ["S01", "S02"]


def test_segment_without_words_falls_back_to_the_turn_span():
    segment = Segment(index=0, start=1.0, end=4.0, text="no words here", speaker="S01")
    lines = build_lines([segment], LineConfig())
    assert (lines[0].start, lines[0].end) == (1.0, 4.0)
    assert lines[0].text == "no words here"


def test_line_confidence_averages_its_words():
    segment = make_segment(["a", "b"])
    segment.words[0].confidence = 0.4
    segment.words[1].confidence = 0.8
    lines = build_lines([segment], LineConfig(max_chars=999, max_duration=999))
    assert lines[0].confidence == 0.6000000000000001 or abs(lines[0].confidence - 0.6) < 1e-9


def test_lines_are_renumbered_contiguously():
    segment = make_segment(["word"] * 12)
    lines = build_lines([segment], LineConfig(max_chars=15))
    assert [line.index for line in lines] == list(range(len(lines)))
