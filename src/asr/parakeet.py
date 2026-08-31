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

#: Longest file this will hand to the model in one piece.
#:
#: NVIDIA quote 24 minutes with full attention *on an A100 80GB*. Grace's A100s
#: are 40GB, so the real ceiling here is roughly half that. 8 minutes leaves
#: room for the decoder's own working set on top of the encoder activations.
FULL_ATTENTION_SECONDS = 8 * 60

#: Overlap between consecutive chunks. Long enough that a word spanning a
#: boundary is whole in at least one chunk; the duplicate half is dropped at
#: the overlap midpoint.
CHUNK_OVERLAP_SECONDS = 15.0

#: Kept for the local-attention strategy: past this, switch the encoder rather
#: than chunk. Reaches ~3 hours but costs accuracy, which is why it is no
#: longer the default.
LONG_FORM_THRESHOLD = 20 * 60


@dataclass
class ParakeetConfig:
    model_id: str = DEFAULT_PARAKEET_MODEL
    device: str = "auto"
    #: How to handle audio longer than one full-attention pass.
    #:
    #: "chunk"  - split into FULL_ATTENTION_SECONDS pieces and transcribe each
    #:            with full attention, then stitch. Best accuracy; the model
    #:            never sees a degraded attention window.
    #: "local"  - switch the encoder to local attention and transcribe in one
    #:            pass. Faster, reaches ~3 hours, costs accuracy.
    #: "none"   - hand the whole file over regardless. Will run out of memory
    #:            on anything long.
    long_audio: str = "chunk"
    #: Seconds per chunk under "chunk". Lower it if the job runs out of memory.
    chunk_seconds: float = FULL_ATTENTION_SECONDS
    #: Overlap between chunks, so a word on a boundary survives in one of them.
    chunk_overlap_seconds: float = CHUNK_OVERLAP_SECONDS
    #: Local-attention window, used only when long_audio is "local".
    att_context_size: tuple[int, int] = (256, 256)
    #: Ask the decoder for per-word confidence. Costs a little decode time.
    word_confidence: bool = True
    #: How a word's confidence is computed. "max_prob" is the decoder's own
    #: probability; "entropy" is NeMo's default and is a normalised Tsallis
    #: entropy, which does not read like a probability.
    confidence_method: str = "max_prob"
    #: How token confidences collapse into one word score. NeMo defaults to
    #: "min", which scores a word by its worst token; "prod" is the joint
    #: probability of the word's tokens.
    confidence_aggregation: str = "prod"
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
        """Turn on per-word confidence, degrading gracefully if unavailable.

        The method and aggregation are set explicitly, because NeMo's defaults
        are wrong for this use.

        NeMo defaults to ``name="entropy"`` (Tsallis, alpha=0.33, exp-normed)
        aggregated with ``"min"``. That is a normalised entropy, not a
        probability, and ``min`` scores every word by its single worst token.
        Together they produced a median word score of 0.05 and a maximum of
        0.85 on real interview audio - no word ever looked confident, and the
        reviewer UI's colour buckets were meaningless.

        ``max_prob`` with ``prod`` gives the decoder's own probability for the
        word: the product of its tokens' probabilities, which is a quantity
        that reads the way a confidence should and is comparable with the
        WhisperX scores this replaced.
        """
        try:
            from omegaconf import open_dict
            from nemo.collections.asr.parts.utils.asr_confidence_utils import (
                ConfidenceConfig,
                ConfidenceMethodConfig,
            )

            decoding_cfg = self._model.cfg.decoding
            with open_dict(decoding_cfg):
                decoding_cfg.confidence_cfg = ConfidenceConfig(
                    preserve_frame_confidence=True,
                    preserve_token_confidence=True,
                    preserve_word_confidence=True,
                    aggregation=self.config.confidence_aggregation,
                    method_cfg=ConfidenceMethodConfig(
                        name=self.config.confidence_method
                    ),
                )
            self._model.change_decoding_strategy(decoding_cfg)
            log.info(
                "per-word confidence enabled (%s, aggregation=%s)",
                self.config.confidence_method,
                self.config.confidence_aggregation,
            )
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

        strategy = self.config.long_audio
        too_long = duration is not None and duration > self.config.chunk_seconds

        if strategy == "chunk" and too_long:
            return self._transcribe_chunked(audio_path, duration, started)

        if strategy == "local" and duration is not None and duration > LONG_FORM_THRESHOLD:
            self._enable_long_form()

        outputs = self._model.transcribe(
            [str(audio_path)], timestamps=True, batch_size=self.config.batch_size
        )
        return self._to_transcription(outputs[0], started)

    def _transcribe_chunked(
        self, audio_path: str | Path, duration: float, started: float
    ) -> WordTranscription:
        """Transcribe a long file as overlapping full-attention chunks.

        Each chunk is short enough for the encoder's full attention window, so
        no accuracy is given up to a degraded one. Chunks overlap, and where
        two cover the same instant the later chunk's words are dropped up to
        the middle of the overlap - one clean cut, no duplicated or missing
        words at the seam.

        Timestamps are shifted by each chunk's own offset, so the result is on
        the same measured time base as an unchunked run, which is what lets the
        speaker turns still line up.
        """
        import tempfile

        from .audio import TARGET_SR, load_audio, slice_audio, to_wav_bytes

        samples, rate = load_audio(audio_path, TARGET_SR)
        step = max(1.0, self.config.chunk_seconds - self.config.chunk_overlap_seconds)
        starts = []
        cursor = 0.0
        while cursor < duration:
            starts.append(cursor)
            cursor += step

        log.info(
            "%s: %.0fs is longer than one %.0fs full-attention pass; "
            "transcribing as %d overlapping chunk(s)",
            Path(audio_path).name, duration, self.config.chunk_seconds, len(starts),
        )

        words: list[Word] = []
        texts: list[str] = []
        warnings: list[str] = []

        with tempfile.TemporaryDirectory(prefix="parakeet-chunks-") as workdir:
            for index, offset in enumerate(starts):
                chunk, chunk_start = slice_audio(
                    samples, rate, offset, min(offset + self.config.chunk_seconds, duration)
                )
                if chunk.size == 0:
                    continue

                path = Path(workdir) / f"chunk{index:03d}.wav"
                path.write_bytes(to_wav_bytes(chunk, rate))

                outputs = self._model.transcribe(
                    [str(path)], timestamps=True, batch_size=self.config.batch_size
                )
                part = self._to_transcription(outputs[0], started)
                warnings.extend(part.warnings)

                # Cut both sides of every overlap at its midpoint, so each
                # instant of audio is claimed by exactly one chunk. Trimming
                # only the later chunk's head would leave the first half of
                # the overlap covered twice - which shows up as duplicated
                # words and timestamps that go backwards at every seam.
                half = self.config.chunk_overlap_seconds / 2.0
                first, last = index == 0, index == len(starts) - 1
                lower = 0.0 if first else half
                upper = float("inf") if last else self.config.chunk_seconds - half

                kept = []
                for word in part.words:
                    if word.start < lower or word.start >= upper:
                        continue
                    word.start += chunk_start
                    word.end += chunk_start
                    kept.append(word)

                words.extend(kept)
                if kept:
                    texts.append(" ".join(w.text for w in kept))

        log.info("chunked transcription produced %d words", len(words))
        return WordTranscription(
            words=words,
            text=" ".join(texts).strip(),
            model=self.config.model_id,
            elapsed_sec=time.time() - started,
            warnings=sorted(set(warnings)),
        )

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
