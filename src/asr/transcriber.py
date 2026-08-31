"""Transcription: Parakeet for words, Sortformer for speaker turns, fused by time.

Each model does only what it is best at::

    audio
      |
      +--[A]-- Parakeet-TDT ----> words: text + start/end + confidence
      |                                            |
      +--[B]-- Sortformer -------> speaker turns   |
                                        |          |
                                        v          v
                                    [fuse by time overlap]
                                              |
                                     diarized segments
                                              |
                                      line reflow -> JSON + VTT

Why this pairing: both models derive time from the same 16 kHz waveform through
the same FastConformer frontend, so the fusion joins two *measured* time bases
and alignment error is bounded by Sortformer's 80 ms frame grid. A diarizer that
writes its timestamps as generated text would give predicted times instead, and
nothing constrains those to the audio.

Both are NeMo, so they share one environment and one process. They run
sequentially by default: ``parallel=True`` with two GPUs was tried on Grace and
fails two ways - concurrent NeMo imports deadlock on the import lock, and
PyTorch's thread-local current device makes a model pinned to cuda:1 build
tensors on cuda:0, which raises an illegal memory access and poisons the CUDA
context for the rest of the process. The overlap saved seconds against a
download measured in minutes, so it was not a trade worth making.

If diarization fails the run continues without speaker labels rather than
failing outright - a transcript with word timings and no speakers is still
useful, and the degradation is recorded in the result metadata.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .audio import TARGET_SR, duration_of, load_audio
from .fusion import SpeakerTurn, fuse
from .lines import LineConfig, build_lines
from .parakeet import ParakeetBackend, ParakeetConfig, WordTranscription
from .scoring import score_transcript
from .sortformer import SortformerConfig, SortformerDiarizer
from .types import Transcript, Word

log = logging.getLogger(__name__)


@dataclass
class TranscriberConfig:
    parakeet: ParakeetConfig = field(default_factory=ParakeetConfig)
    sortformer: SortformerConfig = field(default_factory=SortformerConfig)
    lines: LineConfig = field(default_factory=LineConfig)

    #: Attach speaker labels at all. Off = Parakeet only, no diarization.
    diarize: bool = True
    #: Run the two models concurrently. Only a real gain when they sit on
    #: different devices - see ``parakeet.device`` and ``sortformer.device``.
    parallel: bool = True
    #: A word overlapping no turn takes the nearest turn within this many
    #: seconds; beyond it the word is left without a speaker.
    max_gap: float = 1.0
    sample_rate: int = TARGET_SR

    @property
    def devices_differ(self) -> bool:
        """True when the two models are pinned to different accelerators."""
        return (
            self.parakeet.device != self.sortformer.device
            and "auto" not in (self.parakeet.device, self.sortformer.device)
        )


class Transcriber:
    """Holds both models loaded so a batch pays the load cost once."""

    def __init__(
        self,
        config: Optional[TranscriberConfig] = None,
        parakeet_backend=None,
        diarizer=None,
    ):
        self.config = config or TranscriberConfig()
        self._parakeet = parakeet_backend or ParakeetBackend(self.config.parakeet)
        self._diarizer = diarizer if diarizer is not None else self._build_diarizer()

        if (
            self.config.parallel
            and self._diarizer is not None
            and not self.config.devices_differ
        ):
            log.warning(
                "parallel=True but both models share device %r; they will contend "
                "for it rather than overlap. Pin them to different GPUs, or pass "
                "--sequential-models.",
                self.config.parakeet.device,
            )

    def _build_diarizer(self):
        return SortformerDiarizer(self.config.sortformer) if self.config.diarize else None

    # ------------------------------------------------------------------ run

    def run(
        self, audio_path: str | Path, reference_text: Optional[str] = None
    ) -> Transcript:
        audio_path = Path(audio_path).expanduser()
        timings: dict[str, float] = {}

        with _timed(timings, "ingest"):
            samples, sample_rate = load_audio(audio_path, self.config.sample_rate)
        duration = duration_of(samples, sample_rate)
        del samples  # both models read the file themselves; free the array early
        log.info("%s: %.1fs of audio", audio_path.name, duration)

        with _timed(timings, "models"):
            transcription, turns, diar_meta = self._run_models(audio_path, duration)

        words = _clamp_words(transcription.words, duration)

        with _timed(timings, "fuse"):
            segments, fusion_stats = fuse(words, turns, max_gap=self.config.max_gap)
        log.info(
            "fused %d words into %d segments across %d speaker(s)",
            len(words), len(segments), len(fusion_stats.speakers),
        )

        with _timed(timings, "reflow"):
            lines = build_lines(segments, self.config.lines)

        transcript = Transcript(
            audio_path=str(audio_path),
            duration=duration,
            segments=segments,
            lines=lines,
            raw_text=transcription.text,
            meta={
                "transcription": transcription.as_dict(),
                "diarization": diar_meta,
                "fusion": fusion_stats.as_dict(),
                "speakers": fusion_stats.speakers,
                "sample_rate": sample_rate,
                "counts": {
                    "segments": len(segments),
                    "lines": len(lines),
                    "words": len(words),
                },
                "stage_seconds": {k: round(v, 3) for k, v in timings.items()},
            },
        )

        if reference_text:
            with _timed(timings, "score"):
                score_transcript(transcript, reference_text)
            transcript.meta["stage_seconds"]["score"] = round(timings["score"], 3)

        return transcript

    # ------------------------------------------------------------ execution

    def _run_models(
        self, audio_path: Path, duration: float
    ) -> tuple[WordTranscription, list[SpeakerTurn], dict[str, Any]]:
        if self._diarizer is None:
            return (
                self._parakeet.transcribe_words(audio_path, duration),
                [],
                {"role": "diarization", "status": "disabled", "turns": 0},
            )

        if not self.config.parallel:
            transcription = self._parakeet.transcribe_words(audio_path, duration)
            turns, meta = self._diarize(audio_path, duration)
            return transcription, turns, meta

        # Both release the GIL inside their forward pass, so on two devices
        # these genuinely overlap.
        with ThreadPoolExecutor(max_workers=2) as pool:
            words_future = pool.submit(
                self._parakeet.transcribe_words, audio_path, duration
            )
            turns_future = pool.submit(self._diarize, audio_path, duration)
            transcription = words_future.result()
            turns, meta = turns_future.result()
        return transcription, turns, meta

    def _diarize(
        self, audio_path: Path, duration: float
    ) -> tuple[list[SpeakerTurn], dict[str, Any]]:
        """Diarization must never sink a run - the transcript is the point."""
        try:
            return self._diarizer.turns(audio_path, duration)
        except Exception as exc:  # noqa: BLE001
            log.error("diarization failed, continuing without speakers: %s", exc)
            return [], {
                "role": "diarization",
                "status": "failed",
                "error": str(exc),
                "turns": 0,
            }


def transcribe(
    audio_path: str | Path,
    config: Optional[TranscriberConfig] = None,
    reference_text: Optional[str] = None,
) -> Transcript:
    """One-shot convenience wrapper around :class:`Transcriber`."""
    return Transcriber(config).run(audio_path, reference_text)


def _clamp_words(words: list[Word], duration: float) -> list[Word]:
    """Keep word spans inside the file and non-decreasing."""
    limit = duration if duration > 0 else float("inf")
    cursor = 0.0
    for word in words:
        word.start = min(max(word.start, cursor), limit)
        word.end = min(max(word.end, word.start), limit)
        cursor = word.start
    return words


class _timed:
    """Context manager recording wall-clock seconds into ``store[key]``."""

    def __init__(self, store: dict[str, float], key: str):
        self.store = store
        self.key = key

    def __enter__(self) -> "_timed":
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self.store[self.key] = time.perf_counter() - self.started
        return False


__all__ = ["Transcriber", "TranscriberConfig", "transcribe"]
