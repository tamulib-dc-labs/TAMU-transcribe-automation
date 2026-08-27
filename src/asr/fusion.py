"""Fuse a word-level transcript with speaker turns from a separate model.

The two models have complementary blind spots:

* **Parakeet-TDT** emits word timestamps and per-word confidence natively, but
  has no notion of speakers.
* **Sortformer** emits speaker activity on an 80 ms frame grid, but no words.

Each word is assigned the speaker of the turn it overlaps most in time. That
works because both models measure time against the same 16 kHz waveform through
the same FastConformer frontend, so the two sides share one time base and the
assignment error is bounded by the frame grid rather than by model drift.

Nothing here imports torch, so the fusion logic is testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .textutil import join_words
from .types import Segment, Word


@dataclass(frozen=True)
class SpeakerTurn:
    """One diarized turn: a time span attributed to a speaker."""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class FusionStats:
    words_total: int = 0
    words_assigned: int = 0
    words_unassigned: int = 0
    words_by_nearest: int = 0
    turns_total: int = 0
    turns_empty: int = 0
    mean_overlap_fraction: float = 0.0
    speakers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "words_total": self.words_total,
            "words_assigned": self.words_assigned,
            "words_unassigned": self.words_unassigned,
            "words_assigned_by_proximity": self.words_by_nearest,
            "turns_total": self.turns_total,
            "turns_without_words": self.turns_empty,
            "assigned_fraction": (
                round(self.words_assigned / self.words_total, 4) if self.words_total else 0.0
            ),
            "mean_overlap_fraction": round(self.mean_overlap_fraction, 4),
            "speakers": self.speakers,
        }


def turns_from_segments(segments: list[Segment]) -> list[SpeakerTurn]:
    """Extract diarization turns from a transcript, ignoring its text."""
    return [
        SpeakerTurn(start=s.start, end=s.end, speaker=s.speaker)
        for s in segments
        if s.speaker and s.end > s.start
    ]


def assign_speakers(
    words: list[Word],
    turns: list[SpeakerTurn],
    max_gap: float = 1.0,
) -> tuple[list[Optional[int]], FusionStats]:
    """Attach a speaker to every word; return each word's turn index and stats.

    A word takes the speaker of the turn it overlaps most. A word that overlaps
    no turn - diarization gaps are common at turn edges - falls back to the
    nearest turn within ``max_gap`` seconds, and is left unassigned beyond that
    rather than being guessed at.
    """
    stats = FusionStats(words_total=len(words), turns_total=len(turns))
    if not words:
        return [], stats
    if not turns:
        stats.words_unassigned = len(words)
        return [None] * len(words), stats

    ordered = sorted(range(len(turns)), key=lambda i: (turns[i].start, turns[i].end))
    by_start = [turns[i] for i in ordered]

    assignments: list[Optional[int]] = []
    overlap_fractions: list[float] = []
    used: set[int] = set()
    cursor = 0

    for word in words:
        # Turns ending before this word starts can never match a later word.
        while cursor < len(by_start) and by_start[cursor].end < word.start:
            cursor += 1

        best_index, best_overlap = _best_overlap(word, by_start, cursor)
        if best_overlap > 0:
            original = ordered[best_index]
            assignments.append(original)
            used.add(original)
            word.speaker = by_start[best_index].speaker
            stats.words_assigned += 1
            duration = max(word.duration, 1e-6)
            overlap_fractions.append(min(1.0, best_overlap / duration))
            continue

        nearest_index, distance = _nearest_turn(word, by_start, cursor)
        if nearest_index is not None and distance <= max_gap:
            original = ordered[nearest_index]
            assignments.append(original)
            used.add(original)
            word.speaker = by_start[nearest_index].speaker
            stats.words_assigned += 1
            stats.words_by_nearest += 1
            overlap_fractions.append(0.0)
        else:
            assignments.append(None)
            word.speaker = None
            stats.words_unassigned += 1

    stats.turns_empty = len(turns) - len(used)
    stats.mean_overlap_fraction = (
        sum(overlap_fractions) / len(overlap_fractions) if overlap_fractions else 0.0
    )
    stats.speakers = sorted({t.speaker for t in turns})
    return assignments, stats


def _best_overlap(
    word: Word, turns: list[SpeakerTurn], cursor: int
) -> tuple[int, float]:
    """Index of the turn overlapping ``word`` most, and that overlap in seconds."""
    best_index, best_overlap = -1, 0.0
    index = cursor
    while index < len(turns) and turns[index].start < word.end:
        overlap = min(word.end, turns[index].end) - max(word.start, turns[index].start)
        if overlap > best_overlap:
            best_index, best_overlap = index, overlap
        index += 1
    return best_index, best_overlap


def _nearest_turn(
    word: Word, turns: list[SpeakerTurn], cursor: int
) -> tuple[Optional[int], float]:
    """Closest turn by edge distance, searching just either side of the cursor."""
    best_index, best_distance = None, float("inf")
    for index in range(max(0, cursor - 1), min(len(turns), cursor + 2)):
        turn = turns[index]
        distance = max(0.0, turn.start - word.end, word.start - turn.end)
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index, best_distance


def build_segments(
    words: list[Word],
    turns: list[SpeakerTurn],
    assignments: list[Optional[int]],
) -> list[Segment]:
    """Rebuild segments as the diarizer's turns, populated with the words.

    Consecutive words sharing a turn form one segment. Segment times come from
    the words, not the turn, so boundaries land on real speech.
    """
    segments: list[Segment] = []
    current_turn: Optional[int] = -1
    bucket: list[Word] = []

    def flush() -> None:
        if not bucket:
            return
        index = len(segments)
        speaker = turns[current_turn].speaker if current_turn is not None else None
        for word in bucket:
            word.segment_index = index
            word.speaker = speaker
        segments.append(
            Segment(
                index=index,
                start=bucket[0].start,
                end=bucket[-1].end,
                text=join_words([w.text for w in bucket]),
                speaker=speaker,
                words=list(bucket),
            )
        )

    for word, turn_index in zip(words, assignments):
        if turn_index != current_turn and bucket:
            flush()
            bucket = []
        current_turn = turn_index
        bucket.append(word)
    flush()
    return segments


def fuse(
    words: list[Word],
    turns: list[SpeakerTurn],
    max_gap: float = 1.0,
) -> tuple[list[Segment], FusionStats]:
    """Assign speakers to ``words`` and regroup them into diarized segments."""
    assignments, stats = assign_speakers(words, turns, max_gap=max_gap)
    return build_segments(words, turns, assignments), stats


__all__ = [
    "FusionStats",
    "SpeakerTurn",
    "assign_speakers",
    "build_segments",
    "fuse",
    "turns_from_segments",
]
