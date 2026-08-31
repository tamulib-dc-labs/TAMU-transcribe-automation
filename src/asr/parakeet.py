"""NVIDIA Parakeet-TDT backend: word timestamps and confidence in one pass.

Parakeet emits word-level timestamps natively, so no forced
alignment stage is needed behind it. Per-word confidence comes from the
transducer decoder when ``ConfidenceConfig(preserve_word_confidence=True)`` is
enabled - a different quantity from a forced-alignment posterior, but the same
role: how strongly the model backs each word.

NeMo is imported lazily. It is a heavy dependency and conflicts with the
Transformers pin MOSS needs, which is why the hybrid pipeline can drive MOSS
over HTTP instead of importing both into one process. See
:mod:`asr_pipeline.hybrid`.

Known upstream issues this backend defends against:

* `NVIDIA-NeMo/Speech#15143 <https://github.com/NVIDIA-NeMo/Speech/issues/15143>`_
  - ``len(words) != len(word_confidence)`` on parakeet-tdt-0.6b-v3, seen when
  chunking long files. Confidences are dropped rather than mispaired, and the
  mismatch is reported in the result metadata.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .types import SOURCE_ALIGNED, Word

log = logging.getLogger(__name__)

DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

#: Beyond this many seconds, switch the encoder to local attention. Full
#: attention handles ~24 minutes on an A100 80GB; local attention reaches ~3
#: hours at some cost in accuracy.
LONG_FORM_THRESHOLD = 20 * 60


@dataclass
class ParakeetConfig:
    model_id: str = DEFAULT_PARAKEET_MODEL
    device: str = "auto"
    #: Switch to local attention for files longer than :data:`LONG_FORM_THRESHOLD`.
    auto_long_form: bool = True
    att_context_size: tuple[int, int] = (256, 256)
    #: Ask the decoder for per-word confidence. Costs a little decode time.
    word_confidence: bool = True
    batch_size: int = 1


@dataclass
class WordTranscription:
    """An ASR result that already carries word-level timings."""

    words: list[Word]
    text: str = ""
    language: Optional[str] = None
    #: The model's own segments, if it produced any.
    segments: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    elapsed_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": "parakeet",
            "elapsed_sec": round(self.elapsed_sec, 3),
            "words": len(self.words),
            "native_segments": len(self.segments),
            "language": self.language,
            "warnings": self.warnings,
        }


class ParakeetBackend:
    """Lazily-loaded NeMo Parakeet-TDT transcriber."""

    def __init__(self, config: Optional[ParakeetConfig] = None):
        self.config = config or ParakeetConfig()
        self._model = None
        self._long_form_enabled = False

    # ---------------------------------------------------------------- loading

    def load(self) -> None:
        if self._model is not None:
            return
        import nemo.collections.asr as nemo_asr  # heavy; imported on first use

        log.info("loading Parakeet %s", self.config.model_id)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model_id)
        model.eval()

        device = _resolve_device(self.config.device)
        if device is not None:
            model = model.to(device)

        self._model = model
        if self.config.word_confidence:
            self._enable_word_confidence()

    def _enable_word_confidence(self) -> None:
        """Turn on per-word confidence, degrading gracefully if unavailable."""
        try:
            from omegaconf import open_dict
            from nemo.collections.asr.parts.utils.asr_confidence_utils import (
                ConfidenceConfig,
            )

            decoding_cfg = self._model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.confidence_cfg = ConfidenceConfig(
                    preserve_frame_confidence=True,
                    preserve_token_confidence=True,
                    preserve_word_confidence=True,
                )
            self._model.change_decoding_strategy(decoding_cfg)
            log.info("per-word confidence enabled")
        except Exception as exc:  # noqa: BLE001 - optional, never fatal
            log.warning(
                "could not enable per-word confidence (%s); words will have "
                "timings but no score",
                exc,
            )

    def _enable_long_form(self) -> None:
        if self._long_form_enabled:
            return
        try:
            self._model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=list(self.config.att_context_size),
            )
            self._long_form_enabled = True
            log.info("switched encoder to local attention for long-form audio")
        except Exception as exc:  # noqa: BLE001
            log.warning("could not switch to local attention: %s", exc)

    # ----------------------------------------------------------- transcription

    def transcribe_words(
        self, audio_path: str | Path, duration: Optional[float] = None
    ) -> WordTranscription:
        """Transcribe and return words that already carry their own timings."""
        self.load()
        started = time.time()

        if (
            self.config.auto_long_form
            and duration is not None
            and duration > LONG_FORM_THRESHOLD
        ):
            self._enable_long_form()

        outputs = self._model.transcribe(
            [str(audio_path)], timestamps=True, batch_size=self.config.batch_size
        )
        hypothesis = outputs[0]
        return self._to_transcription(hypothesis, started)

    def _to_transcription(self, hypothesis, started: float) -> WordTranscription:
        warnings: list[str] = []
        timestamps = getattr(hypothesis, "timestamp", None) or {}
        raw_words = timestamps.get("word") or []
        segments = list(timestamps.get("segment") or [])

        confidences = _word_confidences(hypothesis, len(raw_words), warnings)
        words: list[Word] = []
        for index, entry in enumerate(raw_words):
            text = _word_text(entry)
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=float(entry.get("start", 0.0)),
                    end=float(entry.get("end", 0.0)),
                    confidence=confidences[index] if confidences else None,
                    source=SOURCE_ALIGNED,
                    segment_index=-1,
                )
            )

        if not raw_words:
            warnings.append("model returned no word timestamps")

        return WordTranscription(
            words=words,
            text=(getattr(hypothesis, "text", "") or "").strip(),
            language=getattr(hypothesis, "langs", None) or None,
            segments=segments,
            model=self.config.model_id,
            elapsed_sec=time.time() - started,
            warnings=warnings,
        )


def _word_text(entry: dict[str, Any]) -> str:
    for key in ("word", "segment", "char", "text"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _word_confidences(
    hypothesis, expected: int, warnings: list[str]
) -> Optional[list[float]]:
    """Per-word confidence, or ``None`` if absent or the wrong length.

    Guards NVIDIA-NeMo/Speech#15143: a length mismatch silently mispairs every
    score after the first divergence, which is worse than having no scores.
    """
    scores = getattr(hypothesis, "word_confidence", None)
    if not scores:
        return None
    scores = list(scores)
    if len(scores) != expected:
        warnings.append(
            f"word_confidence length {len(scores)} != word count {expected}; "
            "scores discarded (see NVIDIA-NeMo/Speech#15143)"
        )
        log.warning(warnings[-1])
        return None
    try:
        return [float(s) for s in scores]
    except (TypeError, ValueError):
        warnings.append("word_confidence was not numeric; scores discarded")
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
        log.warning("cuda requested but unavailable; running %s on cpu", "Parakeet")
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
            "Parakeet", name, count,
        )
        return torch.device("cuda:0")

    return device


def segments_to_turnless_words(transcription: WordTranscription) -> list[Word]:
    """The words, with any stale speaker/segment attribution cleared."""
    for word in transcription.words:
        word.speaker = None
        word.segment_index = -1
    return transcription.words


__all__ = [
    "DEFAULT_PARAKEET_MODEL",
    "LONG_FORM_THRESHOLD",
    "ParakeetBackend",
    "ParakeetConfig",
    "WordTranscription",
    "segments_to_turnless_words",
]
