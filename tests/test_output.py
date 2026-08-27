"""WhisperX-shaped output, so existing consumers survive the ASR swap."""

import json

import pytest

from src.asr.output import (
    alignment_stats,
    calculate_word_score_buckets,
    to_vtt,
    to_whisperx_dict,
    write_outputs,
)
from src.asr.lines import LineConfig, build_lines
from src.asr.types import SOURCE_INTERPOLATED, Segment, Transcript, Word


def word(text, start, end, score=0.9, speaker="S01", segment_index=0, source=None):
    return Word(
        text=text,
        start=start,
        end=end,
        confidence=score,
        source=source or ("measured" if score is not None else SOURCE_INTERPOLATED),
        speaker=speaker,
        segment_index=segment_index,
    )


def transcript_with(*segments):
    result = Transcript(audio_path="clip.wav", duration=30.0, segments=list(segments))
    result.lines = build_lines(list(segments), LineConfig(max_chars=42))
    result.meta = {"speakers": ["S01"], "transcription": {"model": "parakeet-tdt-0.6b-v3"}}
    return result


@pytest.fixture
def simple():
    words = [word("hello", 0.0, 0.5, 0.95), word("world", 0.6, 1.2, 0.80)]
    segment = Segment(0, 0.0, 1.2, "hello world", "S01", words)
    return transcript_with(segment)


# ------------------------------------------------------------------ json shape


def test_json_matches_the_whisperx_segment_shape(simple):
    result = to_whisperx_dict(simple, language="en")

    assert result["language"] == "en"
    segment = result["segments"][0]
    assert set(segment) >= {"start", "end", "text", "words"}
    assert segment["start"] == 0.0
    assert segment["end"] == 1.2
    assert segment["text"] == "hello world"


def test_words_carry_the_whisperx_keys(simple):
    entry = to_whisperx_dict(simple)["segments"][0]["words"][0]

    assert entry["word"] == "hello"
    assert entry["start"] == 0.0
    assert entry["end"] == 0.5
    assert entry["score"] == 0.95  # WhisperX calls the posterior "score"


def test_diarization_is_added_without_breaking_the_shape(simple):
    result = to_whisperx_dict(simple)

    assert result["segments"][0]["speaker"] == "S01"
    assert result["segments"][0]["words"][0]["speaker"] == "S01"
    assert result["speakers"] == ["S01"]


def test_interpolated_words_report_a_null_score(simple):
    simple.segments[0].words[1].confidence = None
    simple.segments[0].words[1].source = SOURCE_INTERPOLATED
    entry = to_whisperx_dict(simple)["segments"][0]["words"][1]

    assert entry["score"] is None
    assert entry["source"] == SOURCE_INTERPOLATED


def test_extras_can_be_suppressed_for_a_strict_consumer(simple):
    result = to_whisperx_dict(simple, include_extras=False)

    assert set(result) == {"segments", "language"}
    assert set(result["segments"][0]["words"][0]) == {"word", "start", "end", "score"}


def test_output_is_json_serialisable(simple):
    json.dumps(to_whisperx_dict(simple))  # must not raise


# -------------------------------------------------------------- score buckets


def test_score_buckets_are_percentiles_of_the_files_own_scores():
    scores = [i / 100 for i in range(1, 101)]
    result = {"segments": [{"words": [{"score": s} for s in scores]}]}

    buckets = calculate_word_score_buckets(result)

    assert buckets["Bad"] == pytest.approx(0.26, abs=0.02)
    assert buckets["Neutral"] == pytest.approx(0.51, abs=0.02)
    assert buckets["Good"] == pytest.approx(0.76, abs=0.02)
    assert buckets["Bad"] < buckets["Neutral"] < buckets["Good"]


def test_score_buckets_fall_back_when_nothing_was_aligned():
    result = {"segments": [{"words": [{"score": None}]}]}
    assert calculate_word_score_buckets(result) == {"Good": 0.9, "Neutral": 0.7, "Bad": 0.5}


def test_score_buckets_ignore_unscored_words():
    result = {"segments": [{"words": [{"score": 0.5}, {"score": None}, {"word": "x"}]}]}
    assert calculate_word_score_buckets(result)["Good"] == 0.5


# ------------------------------------------------------------ alignment stats


def test_alignment_stats_report_the_whisperx_keys(simple):
    stats = alignment_stats(simple)

    assert set(stats) >= {
        "total_segments",
        "successful_alignments",
        "failed_alignments",
        "success_rate",
        "total_words",
    }
    assert stats["total_segments"] == 1
    assert stats["successful_alignments"] == 1
    assert stats["success_rate"] == 100.0
    assert stats["total_words"] == 2


def test_a_fully_interpolated_segment_counts_as_a_failed_alignment():
    words = [word("a", 0.0, 1.0, None), word("b", 1.0, 2.0, None)]
    stats = alignment_stats(transcript_with(Segment(0, 0.0, 2.0, "a b", "S01", words)))

    assert stats["successful_alignments"] == 0
    assert stats["failed_alignments"] == 1
    assert stats["success_rate"] == 0.0
    assert stats["words_with_measured_timings"] == 0


# -------------------------------------------------------------------- subtitles


def long_transcript():
    texts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    words = [word(t, i * 1.0, i * 1.0 + 0.9) for i, t in enumerate(texts)]
    segment = Segment(0, 0.0, 8.0, " ".join(texts), "S01", words)
    return transcript_with(segment)


def test_vtt_respects_max_line_width():
    vtt = to_vtt(long_transcript(), max_line_width=12, max_line_count=1)

    body = [
        ln
        for ln in vtt.splitlines()
        if ln and "-->" not in ln and ln != "WEBVTT"
    ]
    for line in body:
        assert len(line.replace("<v S01>", "")) <= 12


def test_max_line_count_packs_lines_into_one_cue():
    one = to_vtt(long_transcript(), max_line_width=12, max_line_count=1)
    two = to_vtt(long_transcript(), max_line_width=12, max_line_count=2)

    assert two.count("-->") < one.count("-->")
    assert two.count("-->") == pytest.approx(one.count("-->") / 2, abs=1)


def test_cue_spans_the_whole_group():
    vtt = to_vtt(long_transcript(), max_line_width=12, max_line_count=2)
    first_cue = [ln for ln in vtt.splitlines() if "-->" in ln][0]

    start, end = first_cue.split(" --> ")
    assert start == "00:00:00.000"
    assert end > start


def test_cues_never_span_two_speakers():
    first = [word("hi", 0.0, 0.4, speaker="S01")]
    second = [word("there", 0.5, 1.0, speaker="S02", segment_index=1)]
    segments = [
        Segment(0, 0.0, 0.4, "hi", "S01", first),
        Segment(1, 0.5, 1.0, "there", "S02", second),
    ]
    vtt = to_vtt(transcript_with(*segments), max_line_width=42, max_line_count=4)

    assert vtt.count("-->") == 2
    assert "<v S01>" in vtt and "<v S02>" in vtt


def test_highlight_words_emits_one_cue_per_word():
    vtt = to_vtt(long_transcript(), max_line_width=42, max_line_count=2, highlight_words=True)

    assert vtt.count("<u>") == 8  # one highlighted word per cue
    assert "<u>alpha</u>" in vtt


def test_vtt_starts_with_the_header():
    assert to_vtt(long_transcript()).startswith("WEBVTT")


# --------------------------------------------------------------------- writing


def test_write_outputs_uses_the_whisperx_directory_layout(simple, tmp_path):
    json_path, vtt_path = write_outputs(simple, "clip.mp3", tmp_path)

    assert json_path == tmp_path / "json" / "clip.json"
    assert vtt_path == tmp_path / "vtts" / "clip.vtt"
    assert json_path.exists() and vtt_path.exists()


def test_written_json_round_trips(simple, tmp_path):
    json_path, _ = write_outputs(simple, "clip.mp3", tmp_path, language="es")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    assert data["language"] == "es"
    assert data["segments"][0]["words"][0]["word"] == "hello"
    assert "word_score_buckets" in data
    assert "alignment_stats" in data
