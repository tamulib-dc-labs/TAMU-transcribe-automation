"""Core data structures shared by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


# How a word's timestamps were obtained. "measured" means the model produced
# them from the audio (Parakeet's TDT decoder); "interpolated" means they were
# estimated because the model gave none. There is no forced-alignment stage.
SOURCE_ALIGNED = "measured"
SOURCE_INTERPOLATED = "interpolated"

# Per-word verdict when a reference transcript was supplied.
STATUS_CORRECT = "correct"
STATUS_SUBSTITUTION = "substitution"
STATUS_INSERTION = "insertion"


@dataclass
class Word:
    """A single word with its own time span and confidence."""

    text: str
    start: float
    end: float
    #: Acoustic posterior in [0, 1] from the alignment model. ``None`` when the
    #: word was not force-aligned (see :data:`SOURCE_INTERPOLATED`).
    confidence: Optional[float] = None
    source: str = SOURCE_ALIGNED
    speaker: Optional[str] = None
    #: Index of the speaker-turn segment this word came from.
    segment_index: int = -1
    #: ``correct`` / ``substitution`` / ``insertion``, only when scored against
    #: a reference transcript.
    status: Optional[str] = None
    #: The reference word this one was matched to, when scored.
    reference: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Segment:
    """One speaker turn: a start/end span attributed to a speaker."""

    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def confidence(self) -> Optional[float]:
        return _mean_confidence(self.words)


@dataclass
class Line:
    """A re-flowed subtitle line, timed from its constituent words."""

    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    segment_index: int = -1
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def confidence(self) -> Optional[float]:
        return _mean_confidence(self.words)


@dataclass
class Transcript:
    """The complete pipeline output."""

    audio_path: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    raw_text: str = ""
    #: WER / word-accuracy metrics, present only when a reference was supplied.
    accuracy: Optional[dict[str, Any]] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def words(self) -> list[Word]:
        return [w for seg in self.segments for w in seg.words]

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "duration": round(self.duration, 3),
            "text": self.text,
            "meta": self.meta,
            "accuracy": self.accuracy,
            "segments": [
                {
                    "index": s.index,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "speaker": s.speaker,
                    "text": s.text,
                    "confidence": _round_opt(s.confidence),
                    "words": [_word_dict(w) for w in s.words],
                }
                for s in self.segments
            ],
            "lines": [
                {
                    "index": ln.index,
                    "start": round(ln.start, 3),
                    "end": round(ln.end, 3),
                    "speaker": ln.speaker,
                    "segment_index": ln.segment_index,
                    "text": ln.text,
                    "confidence": _round_opt(ln.confidence),
                }
                for ln in self.lines
            ],
        }


def _word_dict(w: Word) -> dict[str, Any]:
    d = {
        "word": w.text,
        "start": round(w.start, 3),
        "end": round(w.end, 3),
        "confidence": _round_opt(w.confidence),
        "source": w.source,
        "speaker": w.speaker,
        "segment_index": w.segment_index,
    }
    if w.status is not None:
        d["status"] = w.status
        d["reference"] = w.reference
    return d


def _round_opt(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), ndigits)


def _mean_confidence(words: list[Word]) -> Optional[float]:
    scores = [w.confidence for w in words if w.confidence is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


__all__ = [
    "Word",
    "Segment",
    "Line",
    "Transcript",
    "SOURCE_ALIGNED",
    "SOURCE_INTERPOLATED",
    "STATUS_CORRECT",
    "STATUS_SUBSTITUTION",
    "STATUS_INSERTION",
    "asdict",
]
