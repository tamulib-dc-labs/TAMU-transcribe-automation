"""Audio ingestion: decode anything to mono float32 at a target sample rate.

Backends are tried in order of quality/availability so the pipeline runs on a
bare machine (no ffmpeg) as long as one of them is present:

``soundfile`` (wav/flac/ogg/mp3) -> ``av`` (mp4/mkv/m4a/webm) -> ``torchaudio``
-> stdlib ``wave`` (PCM wav only).
"""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path
from typing import Optional

import numpy as np

TARGET_SR = 16_000

VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv", ".m4a"}


class AudioLoadError(RuntimeError):
    """Raised when no decoder backend could read the file."""


def load_audio(path: str | Path, sample_rate: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Decode ``path`` to a mono float32 array at ``sample_rate``.

    Returns ``(samples, sample_rate)``. Samples are in ``[-1, 1]``.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    errors: list[str] = []
    loaders = (
        (_load_with_av, _load_with_soundfile)
        if path.suffix.lower() in VIDEO_SUFFIXES
        else (_load_with_soundfile, _load_with_av)
    )
    for loader in (*loaders, _load_with_torchaudio, _load_with_wave):
        try:
            samples, sr = loader(path)
        except ImportError as exc:
            errors.append(f"{loader.__name__}: not installed ({exc})")
            continue
        except Exception as exc:  # noqa: BLE001 - report and try the next backend
            errors.append(f"{loader.__name__}: {exc}")
            continue
        samples = to_mono(samples)
        return resample(samples, sr, sample_rate), sample_rate

    raise AudioLoadError(
        f"Could not decode {path}. Tried:\n  " + "\n  ".join(errors) +
        "\nInstall decoders with: pip install 'asr-pipeline[audio]'"
    )


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32), int(sr)


def _load_with_av(path: Path) -> tuple[np.ndarray, int]:
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream in container")
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=TARGET_SR
        )
        for frame in container.decode(stream):
            for out in _as_list(resampler.resample(frame)):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in _as_list(resampler.resample(None)):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("no decodable audio samples")
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return pcm, TARGET_SR


def _as_list(frames) -> list:
    if frames is None:
        return []
    return frames if isinstance(frames, list) else [frames]


def _load_with_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    import torchaudio  # type: ignore

    wav, sr = torchaudio.load(str(path))
    return wav.numpy().astype(np.float32), int(sr)


def _load_with_wave(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"unsupported PCM sample width: {width}")
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if width == 1:
        data = (data - 128.0) / 128.0
    else:
        data /= float(np.iinfo(dtype).max)
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, sr


def to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        return samples
    # Accept both (n, channels) and (channels, n).
    axis = 1 if samples.shape[0] > samples.shape[1] else 0
    return samples.mean(axis=axis, dtype=np.float32)


def resample(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono audio, preferring a real resampler when one is installed."""
    if sr_in == sr_out or samples.size == 0:
        return np.ascontiguousarray(samples, dtype=np.float32)

    try:
        import soxr  # type: ignore

        return np.asarray(soxr.resample(samples, sr_in, sr_out), dtype=np.float32)
    except ImportError:
        pass
    try:
        from scipy.signal import resample_poly  # type: ignore

        g = math.gcd(int(sr_in), int(sr_out))
        return np.asarray(
            resample_poly(samples, sr_out // g, sr_in // g), dtype=np.float32
        )
    except ImportError:
        pass
    return _resample_sinc(samples, sr_in, sr_out)


def _resample_sinc(samples: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Anti-aliased fallback resampler: windowed-sinc low-pass + interpolation.

    Lower quality than soxr but dependency-free, and good enough that word
    alignment is not measurably affected for speech.
    """
    if sr_out < sr_in:  # decimating -> filter first to avoid aliasing
        cutoff = 0.95 * (sr_out / 2) / (sr_in / 2)  # normalised to Nyquist
        taps = _lowpass_taps(cutoff, num_taps=101)
        samples = np.convolve(samples, taps, mode="same").astype(np.float32)

    n_out = int(round(samples.size * sr_out / sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_in = np.arange(samples.size, dtype=np.float64)
    x_out = np.arange(n_out, dtype=np.float64) * (sr_in / sr_out)
    return np.interp(x_out, x_in, samples).astype(np.float32)


def _lowpass_taps(cutoff: float, num_taps: int = 101) -> np.ndarray:
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2
    taps = cutoff * np.sinc(cutoff * n) * np.hamming(num_taps)
    return (taps / taps.sum()).astype(np.float32)


def slice_audio(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    pad: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return ``(chunk, chunk_start_seconds)`` for ``[start-pad, end+pad]``."""
    lo = max(0, int(round((start - pad) * sample_rate)))
    hi = min(samples.size, int(math.ceil((end + pad) * sample_rate)))
    if hi <= lo:
        return np.zeros(0, dtype=np.float32), max(0.0, start)
    return samples[lo:hi], lo / sample_rate


def to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 audio as 16-bit PCM WAV (for HTTP upload)."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def duration_of(samples: np.ndarray, sample_rate: int) -> float:
    return float(samples.size) / float(sample_rate) if sample_rate else 0.0


__all__ = [
    "TARGET_SR",
    "AudioLoadError",
    "load_audio",
    "resample",
    "to_mono",
    "slice_audio",
    "to_wav_bytes",
    "duration_of",
]
