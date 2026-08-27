"""Line-level timestamps.

Diarization already returns one timestamped span per speaker turn, and those are
preserved verbatim as :class:`~asr_pipeline.types.Segment`. But a turn can run
for a minute, which is useless as a subtitle line. This module re-flows the
word stream into readable lines whose start/end come from the *words* rather
than from the turn, so every line boundary is a real acoustic boundary.

Lines never cross a segment (and therefore never cross a speaker change).
"""

from __future__ import annotations

from dataclasses import dataclass

from .textutil import ends_sentence, join_words
from .types import Line, Segment, Word


@dataclass
class LineConfig:
    #: Break before a line grows past this many characters.
    max_chars: int = 42
    #: Break before a line runs longer than this many seconds.
    max_duration: float = 7.0
    #: Break when the silence between two words exceeds this many seconds.
    max_gap: float = 0.8
    #: Break after sentence-final punctuation, once the line is at least this long.
    min_chars_for_sentence_break: int = 12
    #: Never emit a line shorter than this; it is merged into its neighbour.
    min_duration: float = 0.25


def build_lines(segments: list[Segment], config: LineConfig | None = None) -> list[Line]:
    """Re-flow every segment's words into subtitle-sized, word-timed lines."""
    config = config or LineConfig()
    lines: list[Line] = []
    for segment in segments:
        if not segment.words:
            if segment.text:
                lines.append(_line(len(lines), segment, [], fallback=True))
            continue
        for group in _group_words(segment.words, config):
            lines.append(_line(len(lines), segment, group))
    return _merge_slivers(lines, config)


def _group_words(words: list[Word], config: LineConfig) -> list[list[Word]]:
    groups: list[list[Word]] = []
    current: list[Word] = []
    chars = 0

    for word in words:
        if current and _should_break(current, word, chars, config):
            groups.append(current)
            current, chars = [], 0
        current.append(word)
        chars = len(join_words([w.text for w in current]))
    if current:
        groups.append(current)
    return groups


def _should_break(current: list[Word], word: Word, chars: int, config: LineConfig) -> bool:
    previous = current[-1]
    if word.start - previous.end > config.max_gap:
        return True
    if word.end - current[0].start > config.max_duration:
        return True
    if chars + 1 + len(word.text) > config.max_chars:
        return True
    if ends_sentence(previous.text) and chars >= config.min_chars_for_sentence_break:
        return True
    return False


def _line(index: int, segment: Segment, words: list[Word], fallback: bool = False) -> Line:
    if fallback or not words:
        # A segment with text but no word timings: fall back to the turn's span.
        return Line(
            index=index,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=segment.speaker,
            segment_index=segment.index,
            words=[],
        )
    return Line(
        index=index,
        start=words[0].start,
        end=words[-1].end,
        text=join_words([w.text for w in words]),
        speaker=segment.speaker,
        segment_index=segment.index,
        words=list(words),
    )


def _merge_slivers(lines: list[Line], config: LineConfig) -> list[Line]:
    """Fold sub-``min_duration`` lines into an adjacent line of the same segment."""
    if len(lines) < 2:
        return lines

    merged: list[Line] = []
    for line in lines:
        previous = merged[-1] if merged else None
        too_short = line.duration < config.min_duration
        same_segment = previous is not None and previous.segment_index == line.segment_index
        fits = (
            same_segment
            and len(previous.text) + len(line.text) + 1 <= config.max_chars
            and line.end - previous.start <= config.max_duration
        )
        if too_short and fits:
            previous.end = max(previous.end, line.end)
            previous.words.extend(line.words)
            previous.text = join_words([w.text for w in previous.words]) or previous.text
            continue
        merged.append(line)

    for index, line in enumerate(merged):
        line.index = index
    return merged


__all__ = ["LineConfig", "build_lines"]
