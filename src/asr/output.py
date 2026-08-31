"""The output format the reviewer app already reads.

The previous WhisperX pipeline produced this shape, and the reviewer app
and ``src/git/uploader.py`` were built against it, so it is preserved exactly::

    {"segments": [{"start", "end", "text",
                   "words": [{"word", "start", "end", "score"}]}],
     "language": "en"}

and a VTT written by ``whisperx.utils.get_writer`` with ``max_line_width`` /
``max_line_count`` / ``highlight_words``. This module reproduces both from a
:class:`~asr_pipeline.types.Transcript` so downstream tooling - review UIs,
caption players, QC dashboards - keeps working after the swap.

Additions are strictly additive (``speaker`` on segments and words, ``source``
on words), so consumers that ignore unknown keys are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .lines import LineConfig, build_lines
from .types import SOURCE_ALIGNED, Line, Transcript

DEFAULT_MAX_LINE_WIDTH = 42
DEFAULT_MAX_LINE_COUNT = 2


#: A recording whose first words arrive later than this probably lost its
#: opening - worth saying so in the output rather than leaving it to be noticed
#: by someone reading the transcript.
LEAD_SILENCE_WARN_SECONDS = 30.0


def to_whisperx_dict(
    transcript: Transcript,
    language: str = "en",
    include_extras: bool = True,
) -> dict[str, Any]:
    """Render ``transcript`` in WhisperX's aligned-result shape."""
    segments: list[dict[str, Any]] = []
    for segment in transcript.segments:
        entry: dict[str, Any] = {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "text": segment.text,
            "words": [_word_dict(w, include_extras) for w in segment.words],
        }
        if segment.speaker:
            entry["speaker"] = segment.speaker
        segments.append(entry)

    result: dict[str, Any] = {"segments": segments, "language": language}
    if include_extras:
        result["word_score_buckets"] = calculate_word_score_buckets(result)
        result["alignment_stats"] = alignment_stats(transcript)
        result["speakers"] = transcript.meta.get("speakers") or []
        result["asr_model"] = (transcript.meta.get("transcription") or {}).get("model")
        # Enough to explain a bad transcript without re-running the job: which
        # audio, how long, how much of it the model actually covered, and any
        # warnings the backends raised. A file whose first minute is missing
        # shows up here as a first_word_at far from zero.
        result["run"] = _run_summary(transcript)
    return result


def _run_summary(transcript: Transcript) -> dict[str, Any]:
    """Diagnostics that make a suspicious transcript readable on its own."""
    meta = transcript.meta or {}
    words = transcript.words
    asr = meta.get("transcription") or {}

    summary: dict[str, Any] = {
        "audio": transcript.audio_path,
        "audio_seconds": round(transcript.duration, 2),
        "first_word_at": round(words[0].start, 2) if words else None,
        "last_word_at": round(words[-1].end, 2) if words else None,
        "diarization": (meta.get("diarization") or {}).get("status"),
        "stage_seconds": meta.get("stage_seconds"),
    }

    # Silence at the head of a recording is normal; a minute of it is not, and
    # it is the signature of the model dropping the opening.
    lead = summary["first_word_at"]
    if lead is not None and lead > LEAD_SILENCE_WARN_SECONDS:
        summary["warning"] = (
            f"no words in the first {lead:.0f}s of {transcript.duration:.0f}s"
        )

    warnings = list(asr.get("warnings") or [])
    if warnings:
        summary["asr_warnings"] = warnings
    return summary


def _word_dict(word, include_extras: bool) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "word": word.text,
        "start": round(word.start, 3),
        "end": round(word.end, 3),
        # WhisperX calls the alignment posterior "score"; same quantity here.
        "score": None if word.confidence is None else round(word.confidence, 4),
    }
    if include_extras:
        if word.speaker:
            entry["speaker"] = word.speaker
        # "measured" = the model produced these timings; "interpolated" = estimated.
        entry["source"] = word.source
        if word.status is not None:
            entry["status"] = word.status
    return entry


def calculate_word_score_buckets(result: dict[str, Any]) -> dict[str, float]:
    """Percentile thresholds over word scores (ported from the WhisperX pipeline).

    Returns the 75th / 50th / 25th percentile as Good / Neutral / Bad. Because
    the thresholds are percentiles of the file's own scores, they stay
    meaningful even though a different aligner produces a different score
    distribution than WhisperX did.
    """
    scores = sorted(
        word["score"]
        for segment in result.get("segments", [])
        for word in segment.get("words", [])
        if word.get("score") is not None
    )
    if not scores:
        return {"Good": 0.9, "Neutral": 0.7, "Bad": 0.5}

    n = len(scores)
    return {
        "Good": round(scores[min(int(n * 0.75), n - 1)], 3),
        "Neutral": round(scores[min(int(n * 0.50), n - 1)], 3),
        "Bad": round(scores[min(int(n * 0.25), n - 1)], 3),
    }


def alignment_stats(transcript: Transcript) -> dict[str, Any]:
    """WhisperX-pipeline-compatible alignment statistics."""
    total = len(transcript.segments)
    aligned = sum(
        1
        for segment in transcript.segments
        if any(w.source == SOURCE_ALIGNED for w in segment.words)
    )
    words = sum(len(segment.words) for segment in transcript.segments)
    measured = sum(
        1 for segment in transcript.segments for w in segment.words if w.source == SOURCE_ALIGNED
    )
    return {
        "total_segments": total,
        "successful_alignments": aligned,
        "failed_alignments": total - aligned,
        "success_rate": round(aligned / total * 100, 1) if total else 0.0,
        "total_words": words,
        "words_with_measured_timings": measured,
    }


# ------------------------------------------------------------------- subtitles


def to_vtt(
    transcript: Transcript,
    max_line_width: int = DEFAULT_MAX_LINE_WIDTH,
    max_line_count: int = DEFAULT_MAX_LINE_COUNT,
    highlight_words: bool = False,
    show_speaker: bool = True,
) -> str:
    """WebVTT with WhisperX's line-shaping options.

    ``max_line_width`` caps each rendered line, ``max_line_count`` caps lines
    per cue, and ``highlight_words`` emits one cue per word with the active
    word underlined (the karaoke style WhisperX produces).
    """
    lines = _lines_for(transcript, max_line_width)
    out = ["WEBVTT", ""]
    for cue in _group_into_cues(lines, max_line_count):
        if highlight_words and any(line.words for line in cue):
            out.extend(_highlighted_cues(cue, show_speaker))
        else:
            out.append(f"{_stamp(cue[0].start)} --> {_stamp(cue[-1].end)}")
            out.append(_cue_body(cue, show_speaker))
            out.append("")
    return "\n".join(out)


def to_srt(
    transcript: Transcript,
    max_line_width: int = DEFAULT_MAX_LINE_WIDTH,
    max_line_count: int = DEFAULT_MAX_LINE_COUNT,
    show_speaker: bool = True,
) -> str:
    lines = _lines_for(transcript, max_line_width)
    blocks: list[str] = []
    for index, cue in enumerate(_group_into_cues(lines, max_line_count), start=1):
        blocks.append(
            f"{index}\n"
            f"{_stamp(cue[0].start, ',')} --> {_stamp(cue[-1].end, ',')}\n"
            f"{_cue_body(cue, show_speaker)}\n"
        )
    return "\n".join(blocks)


def _lines_for(transcript: Transcript, max_line_width: int) -> list[Line]:
    """Reuse the transcript's lines, or rebuild them at the requested width."""
    if transcript.lines and all(len(ln.text) <= max_line_width for ln in transcript.lines):
        return transcript.lines
    return build_lines(transcript.segments, LineConfig(max_chars=max_line_width))


def _group_into_cues(lines: list[Line], max_line_count: int) -> list[list[Line]]:
    """Pack consecutive lines into cues, never crossing a speaker turn."""
    cues: list[list[Line]] = []
    current: list[Line] = []
    for line in lines:
        crosses_turn = current and current[-1].segment_index != line.segment_index
        if current and (crosses_turn or len(current) >= max(1, max_line_count)):
            cues.append(current)
            current = []
        current.append(line)
    if current:
        cues.append(current)
    return cues


def _cue_body(cue: list[Line], show_speaker: bool) -> str:
    prefix = f"<v {cue[0].speaker}>" if show_speaker and cue[0].speaker else ""
    return prefix + "\n".join(_escape(line.text) for line in cue)


def _highlighted_cues(cue: list[Line], show_speaker: bool) -> list[str]:
    """One cue per word, that word underlined - WhisperX's highlight style."""
    words = [w for line in cue for w in line.words]
    prefix = f"<v {cue[0].speaker}>" if show_speaker and cue[0].speaker else ""
    out: list[str] = []
    for index, word in enumerate(words):
        rendered = " ".join(
            f"<u>{_escape(w.text)}</u>" if i == index else _escape(w.text)
            for i, w in enumerate(words)
        )
        out.append(f"{_stamp(word.start)} --> {_stamp(word.end)}")
        out.append(prefix + rendered)
        out.append("")
    return out


def _stamp(seconds: float, decimal: str = ".") -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ------------------------------------------------------------------- writing


def write_outputs(
    transcript: Transcript,
    audio_path: str | Path,
    output_dir: str | Path,
    language: str = "en",
    max_line_width: int = DEFAULT_MAX_LINE_WIDTH,
    max_line_count: int = DEFAULT_MAX_LINE_COUNT,
    highlight_words: bool = False,
    json_subdir: str = "json",
    vtt_subdir: str = "vtts",
) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and ``<stem>.vtt`` in the WhisperX layout.

    Mirrors the ``output_dir/json`` + ``output_dir/vtts`` convention so an
    existing uploader keeps finding its files.
    """
    output_dir = Path(output_dir)
    stem = Path(audio_path).stem

    json_dir = output_dir / json_subdir
    vtt_dir = output_dir / vtt_subdir
    json_dir.mkdir(parents=True, exist_ok=True)
    vtt_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(to_whisperx_dict(transcript, language), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    vtt_path = vtt_dir / f"{stem}.vtt"
    vtt_path.write_text(
        to_vtt(transcript, max_line_width, max_line_count, highlight_words),
        encoding="utf-8",
    )
    return json_path, vtt_path


__all__ = [
    "alignment_stats",
    "calculate_word_score_buckets",
    "to_srt",
    "to_vtt",
    "to_whisperx_dict",
    "write_outputs",
]
