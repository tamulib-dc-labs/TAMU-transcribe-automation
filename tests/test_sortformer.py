"""The Sortformer diarizer.

NeMo's documented output is 'begin_seconds, end_seconds, speaker_index' but not
its container type, so the parser accepts every plausible shape - a version
bump changing tuples to dicts would otherwise produce zero turns silently, and
a transcript with no speakers looks like a diarization failure rather than a
parsing bug.
"""

import pytest

from src.asr.fusion import SpeakerTurn
from src.asr.sortformer import (
    DEFAULT_SORTFORMER_MODEL,
    FRAME_SECONDS,
    LONG_CONTEXT,
    PRESETS,
    SortformerConfig,
    SortformerDiarizer,
    _parse_segment,
    _speaker_label,
)


class StubModel:
    """Stands in for SortformerEncLabelModel."""

    def __init__(self, segments):
        self.segments = segments
        self.sortformer_modules = type("M", (), {"_check_streaming_parameters": lambda s: None})()
        self.calls = []

    def diarize(self, audio, batch_size=1, **kwargs):
        self.calls.append((audio, batch_size))
        return [self.segments]


def make_diarizer(segments, **config_kwargs):
    diarizer = SortformerDiarizer(SortformerConfig(**config_kwargs))
    diarizer._model = StubModel(segments)  # marks it loaded
    return diarizer


# ------------------------------------------------------------------ defaults


def test_defaults_to_v2_1():
    assert DEFAULT_SORTFORMER_MODEL.endswith("v2.1")
    assert SortformerConfig().model_id == DEFAULT_SORTFORMER_MODEL


def test_default_preset_is_the_long_context_one():
    """Archived interviews are not latency-sensitive; this is the accurate
    and the cheapest setting, at RTF 0.002."""
    assert SortformerConfig().preset == "long-context"
    assert PRESETS["long-context"] == LONG_CONTEXT
    assert LONG_CONTEXT["chunk_len"] == 340


def test_frame_grid_is_eighty_milliseconds():
    """This is what bounds fusion error against Parakeet's word times."""
    assert FRAME_SECONDS == 0.08


def test_an_unknown_preset_is_rejected():
    diarizer = SortformerDiarizer(SortformerConfig(preset="warp-speed"))
    with pytest.raises(ValueError, match="unknown preset"):
        diarizer._apply_preset(StubModel([]))


# ------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "entry",
    [
        (0.48, 3.25, 0),
        [0.48, 3.25, 0],
        "0.48 3.25 speaker_0",
        "0.48,3.25,0",
        {"start": 0.48, "end": 3.25, "speaker": 0},
        {"begin": 0.48, "end": 3.25, "speaker_index": 0},
    ],
    ids=["tuple", "list", "string", "csv", "dict", "dict-alt"],
)
def test_every_plausible_segment_shape_parses(entry):
    start, end, speaker = _parse_segment(entry)
    assert start == pytest.approx(0.48)
    assert end == pytest.approx(3.25)
    assert _speaker_label(speaker, SortformerConfig()) == "S01"


@pytest.mark.parametrize("entry", [None, 42, "", "only two", {"start": 1.0}, ()])
def test_unparseable_segments_return_none(entry):
    assert _parse_segment(entry) is None


def test_speaker_indices_become_the_existing_label_format():
    config = SortformerConfig()
    assert _speaker_label(0, config) == "S01"
    assert _speaker_label(3, config) == "S04"
    assert _speaker_label("speaker_1", config) == "S02"


# ------------------------------------------------------------------ diarize


def test_turns_come_back_ready_for_fusion():
    diarizer = make_diarizer([(0.5, 3.2, 0), (3.4, 7.0, 1)])
    turns, meta = diarizer.turns("iv.wav", duration=10.0)

    assert all(isinstance(t, SpeakerTurn) for t in turns)
    assert [(t.start, t.end, t.speaker) for t in turns] == [
        (0.5, 3.2, "S01"),
        (3.4, 7.0, "S02"),
    ]
    assert meta["turns"] == 2
    assert meta["speakers"] == ["S01", "S02"]


def test_metadata_records_the_frame_grid_and_model():
    diarizer = make_diarizer([(0.0, 1.0, 0)])
    _, meta = diarizer.turns("iv.wav", duration=100.0)

    assert meta["backend"] == "sortformer"
    assert meta["frame_seconds"] == FRAME_SECONDS
    assert meta["model"].endswith("v2.1")
    assert meta["rtf"] >= 0


def test_turns_are_sorted_by_time():
    diarizer = make_diarizer([(5.0, 6.0, 1), (0.5, 1.0, 0), (2.0, 3.0, 0)])
    turns, _ = diarizer.turns("iv.wav", duration=10.0)

    assert [t.start for t in turns] == [0.5, 2.0, 5.0]


def test_turns_are_clamped_to_the_audio_duration():
    diarizer = make_diarizer([(0.0, 999.0, 0)])
    turns, _ = diarizer.turns("iv.wav", duration=12.0)

    assert turns[0].end == 12.0


def test_negative_starts_are_clamped():
    diarizer = make_diarizer([(-3.0, 2.0, 0)])
    turns, _ = diarizer.turns("iv.wav", duration=10.0)
    assert turns[0].start == 0.0


def test_a_garbled_segment_is_skipped_not_fatal():
    diarizer = make_diarizer([(0.5, 1.5, 0), "nonsense", (2.0, 3.0, 1)])
    turns, meta = diarizer.turns("iv.wav", duration=10.0)

    assert meta["turns"] == 2  # the good ones survive


def test_short_turns_can_be_dropped():
    diarizer = make_diarizer(
        [(0.0, 0.05, 0), (1.0, 4.0, 1)], min_turn_seconds=0.2
    )
    turns, _ = diarizer.turns("iv.wav", duration=10.0)

    assert len(turns) == 1
    assert turns[0].speaker == "S02"


def test_empty_diarization_yields_no_turns():
    turns, meta = make_diarizer([]).turns("iv.wav", duration=10.0)
    assert turns == []
    assert meta["turns"] == 0

def test_the_diarizer_signature_takes_a_path_and_duration():
    """The transcriber calls it positionally; keep the contract pinned."""
    import inspect

    params = list(inspect.signature(SortformerDiarizer.turns).parameters)
    assert params[:3] == ["self", "audio_path", "duration"]
