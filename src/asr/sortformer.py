"""NVIDIA Sortformer diarization: speaker turns on the same time base as the words.

Diarization is a *measurement* here, not a prediction, and that is the whole
point. A language-model diarizer writes its timestamps as text - it emits the
characters ``[0.48]`` the way it emits any other token, so those times are
predicted and nothing constrains them to the audio. Fusing predicted times
against Parakeet's frame-derived word times gives unbounded misalignment.

Sortformer emits speaker activity on a fixed **80 ms frame grid**, derived from
the same 16 kHz waveform by the same FastConformer encoder family Parakeet uses.
Both sides are measured against the audio, so fusion error is bounded by the
frame grid - about +/- 40 ms - rather than by model drift.

It is also NeMo, like Parakeet, so the two share one environment. That is what
collapsed the split-worker design back into a single process.

Limits worth remembering: at most four speakers (it degrades sharply beyond
that), English-primary, and trained on conversational rather than archival
audio - noisy tape is the case to check on real material.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .fusion import SpeakerTurn

log = logging.getLogger(__name__)

#: v2.1 is the later release, with better robustness on meeting-style speech.
DEFAULT_SORTFORMER_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"

#: The model's fixed output resolution. Every boundary lands on this grid.
FRAME_SECONDS = 0.08

#: Long-context preset from the model card: highest accuracy, RTF 0.002.
#: All values are counts of 80 ms frames.
LONG_CONTEXT = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_update_period": 300,
    "spkcache_len": 188,
}

#: Lower latency presets, kept for reference. Batch work wants LONG_CONTEXT.
PRESETS = {
    "long-context": LONG_CONTEXT,
    "high-latency": {
        "chunk_len": 124, "chunk_right_context": 1, "fifo_len": 124,
        "spkcache_update_period": 124, "spkcache_len": 188,
    },
    "low-latency": {
        "chunk_len": 6, "chunk_right_context": 7, "fifo_len": 188,
        "spkcache_update_period": 144, "spkcache_len": 188,
    },
}


@dataclass
class SortformerConfig:
    model_id: str = DEFAULT_SORTFORMER_MODEL
    device: str = "auto"
    #: Streaming preset. Archived interviews are not latency-sensitive, so the
    #: long-context setting is right: it is both the most accurate and the
    #: cheapest per second of audio.
    preset: str = "long-context"
    batch_size: int = 1
    #: Speakers are emitted as S01, S02, ... to match the existing output schema.
    speaker_prefix: str = "S"
    #: Drop turns shorter than this; they are usually backchannel artefacts
    #: that would fragment the transcript for no benefit.
    min_turn_seconds: float = 0.0


class SortformerDiarizer:
    """Lazily-loaded NeMo Sortformer, returning turns ready for fusion."""

    def __init__(self, config: Optional[SortformerConfig] = None):
        self.config = config or SortformerConfig()
        self._model = None

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        if self._model is not None:
            return
        from nemo.collections.asr.models import SortformerEncLabelModel

        log.info("loading Sortformer %s", self.config.model_id)
        model = SortformerEncLabelModel.from_pretrained(self.config.model_id)
        model.eval()

        device = _resolve_device(self.config.device)
        if device is not None:
            model = model.to(device)

        self._apply_preset(model)
        self._model = model

    def _apply_preset(self, model) -> None:
        preset = PRESETS.get(self.config.preset)
        if preset is None:
            raise ValueError(
                f"unknown preset {self.config.preset!r}; expected one of {sorted(PRESETS)}"
            )
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            log.warning("model exposes no sortformer_modules; using its defaults")
            return
        for key, value in preset.items():
            setattr(modules, key, value)
        check = getattr(modules, "_check_streaming_parameters", None)
        if callable(check):
            check()
        log.info("sortformer preset '%s' applied", self.config.preset)

    # ------------------------------------------------------------ diarization

    def turns(
        self, audio_path: str | Path, duration: Optional[float] = None
    ) -> tuple[list[SpeakerTurn], dict[str, Any]]:
        """Diarize one file, returning turns ready for fusion."""
        self.load()
        started = time.time()

        predictions = self._model.diarize(
            audio=[str(audio_path)], batch_size=self.config.batch_size
        )
        raw = predictions[0] if predictions else []
        turns = self._to_turns(raw, duration)

        elapsed = time.time() - started
        meta = {
            "role": "diarization",
            "model": self.config.model_id,
            "backend": "sortformer",
            "preset": self.config.preset,
            "frame_seconds": FRAME_SECONDS,
            "turns": len(turns),
            "speakers": sorted({t.speaker for t in turns}),
            "elapsed_sec": round(elapsed, 3),
        }
        if duration:
            meta["rtf"] = round(elapsed / duration, 5)
        return turns, meta

    def _to_turns(self, raw: Any, duration: Optional[float]) -> list[SpeakerTurn]:
        limit = duration if duration and duration > 0 else float("inf")
        turns: list[SpeakerTurn] = []
        for entry in raw or []:
            parsed = _parse_segment(entry)
            if parsed is None:
                log.warning("unrecognised diarization segment: %r", entry)
                continue
            start, end, speaker = parsed
            start = max(0.0, min(start, limit))
            end = max(start, min(end, limit))
            if end - start < self.config.min_turn_seconds:
                continue
            turns.append(
                SpeakerTurn(
                    start=start, end=end, speaker=_speaker_label(speaker, self.config)
                )
            )
        turns.sort(key=lambda t: (t.start, t.end))
        return turns


# ------------------------------------------------------------------- parsing


def _parse_segment(entry: Any) -> Optional[tuple[float, float, Any]]:
    """Read one segment, whichever shape NeMo hands back.

    The model card documents 'begin_seconds, end_seconds, speaker_index' but
    not the container. Accepting all three plausible shapes costs a few lines
    and avoids a version bump silently producing zero turns.
    """
    if isinstance(entry, dict):
        start = entry.get("start", entry.get("begin", entry.get("start_time")))
        end = entry.get("end", entry.get("end_time"))
        speaker = entry.get("speaker", entry.get("speaker_index", entry.get("label")))
        if start is None or end is None:
            return None
        return float(start), float(end), speaker

    if isinstance(entry, (tuple, list)) and len(entry) >= 3:
        try:
            return float(entry[0]), float(entry[1]), entry[2]
        except (TypeError, ValueError):
            return None

    if isinstance(entry, str):
        # e.g. "0.48 3.25 speaker_0" or "0.48,3.25,1"
        fields = [f for f in re.split(r"[\s,]+", entry.strip()) if f]
        if len(fields) < 3:
            return None
        try:
            return float(fields[0]), float(fields[1]), fields[2]
        except ValueError:
            return None

    return None


def _speaker_label(speaker: Any, config: SortformerConfig) -> str:
    """Normalise to S01, S02, ... so the output schema is unchanged."""
    index = _speaker_index(speaker)
    if index is None:
        return str(speaker)
    return f"{config.speaker_prefix}{index + 1:02d}"


def _speaker_index(speaker: Any) -> Optional[int]:
    if isinstance(speaker, bool):
        return None
    if isinstance(speaker, int):
        return speaker
    if isinstance(speaker, float) and speaker.is_integer():
        return int(speaker)
    if isinstance(speaker, str):
        digits = re.findall(r"\d+", speaker)
        if digits:
            return int(digits[-1])
    return None


def _resolve_device(name: str):
    try:
        import torch
    except ImportError:
        return None
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(name)
    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        log.warning("cuda requested but unavailable; running %s on cpu", "Sortformer")
        return torch.device("cpu")

    # torch.cuda.is_available() is True with a single GPU, so cuda:1 gets this
    # far and then fails at .to(device). Falling back is much better than the
    # alternative: the caller treats a load failure as "no speakers" and the
    # run finishes with labels missing from every transcript.
    count = torch.cuda.device_count()
    if device.index is not None and device.index >= count:
        log.warning(
            "%s asked for %s but the job was given %d GPU(s); using cuda:0. "
            "Both models now share one device and will contend rather than "
            "overlap - check --gres in config/run.slurm.",
            "Sortformer", name, count,
        )
        return torch.device("cuda:0")

    return device


__all__ = [
    "DEFAULT_SORTFORMER_MODEL",
    "FRAME_SECONDS",
    "LONG_CONTEXT",
    "PRESETS",
    "SortformerConfig",
    "SortformerDiarizer",
]
