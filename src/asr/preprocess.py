"""Audio preparation: decode, optionally denoise, write a 16 kHz mono WAV.

This is the CPU-bound stage that runs immediately before the GPU stage in each
worker. Denoising a ninety-minute interview costs a few minutes of one core;
if that ever becomes the bottleneck, run more workers rather than adding
machinery inside one.

Decoding goes through :mod:`asr_pipeline.audio`, which reads video containers
with PyAV, so no ffmpeg subprocess is needed - one less thing to install in a
container and one less process spawn per file.

Denoising is the stationary spectral-gating pass the WhisperX pipeline used,
kept for continuity. It is genuinely expensive (roughly real-time on one core
for a long interview), which is exactly why it belongs in a parallel stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import TARGET_SR, load_audio, to_wav_bytes

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mkv", ".mov")

#: Denoise in windows of this many seconds, as the original pipeline did.
DENOISE_CHUNK_SECONDS = 30


def prepare_audio(
    source: str | Path,
    dest_dir: str | Path,
    denoise: bool = True,
    boost: float = 3.0,
    sample_rate: int = TARGET_SR,
) -> Path:
    """Decode ``source`` to a 16 kHz mono WAV in ``dest_dir``; return its path."""
    source = Path(source)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    samples, rate = load_audio(source, sample_rate)
    if denoise:
        samples = reduce_noise(samples, rate)
    if boost and boost != 1.0:
        samples = apply_boost(samples, boost)

    destination = dest_dir / f"{source.stem}.prepared.wav"
    destination.write_bytes(to_wav_bytes(samples, rate))
    return destination


def reduce_noise(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Stationary spectral-gating denoise, chunked to bound memory.

    Returns the input unchanged if ``noisereduce`` is not installed, rather
    than failing the file.
    """
    try:
        import noisereduce as nr
    except ImportError:
        log.warning("noisereduce not installed; skipping the denoise pass")
        return samples

    if samples.size == 0:
        return samples

    chunk = max(1, DENOISE_CHUNK_SECONDS * sample_rate)
    out = np.zeros_like(samples)
    for start in range(0, samples.size, chunk):
        end = min(start + chunk, samples.size)
        out[start:end] = nr.reduce_noise(
            y=samples[start:end],
            sr=sample_rate,
            stationary=True,
            n_std_thresh_stationary=2.5,
            prop_decrease=0.8,
            freq_mask_smooth_hz=1000,
            time_mask_smooth_ms=100,
            n_fft=2048,
            hop_length=512,
            clip_noise_stationary=True,
        )
    return out


def apply_boost(samples: np.ndarray, factor: float) -> np.ndarray:
    """Scale by ``factor``, then peak-normalise only if it would clip."""
    boosted = samples * factor
    peak = float(np.abs(boosted).max()) if boosted.size else 0.0
    if peak > 1.0:
        boosted = boosted / peak
    return boosted.astype(np.float32)


def find_audio_files(
    root: str | Path,
    exclude: Optional[str | Path] = None,
    extensions: tuple[str, ...] = AUDIO_EXTENSIONS,
) -> list[Path]:
    """Every audio/video file under ``root``, skipping anything under ``exclude``."""
    root = Path(root)
    excluded = Path(exclude).resolve() if exclude else None

    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.name.endswith(".prepared.wav"):
            continue  # our own intermediate output
        if excluded and _is_within(path.resolve(), excluded):
            continue
        found.append(path)
    return sorted(found)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


__all__ = [
    "AUDIO_EXTENSIONS",
    "apply_boost",
    "find_audio_files",
    "prepare_audio",
    "reduce_noise",
]
