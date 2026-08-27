"""Audio ingestion, resampling and slicing."""

import numpy as np
import pytest

from src.asr.audio import (
    duration_of,
    load_audio,
    resample,
    slice_audio,
    to_mono,
    to_wav_bytes,
)


def sine(seconds: float, freq: float, sample_rate: int) -> np.ndarray:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def write_wav(path, samples, sample_rate):
    path.write_bytes(to_wav_bytes(samples, sample_rate))
    return path


def test_wav_round_trip_preserves_duration_and_shape(tmp_path):
    original = sine(0.5, 440.0, 16_000)
    path = write_wav(tmp_path / "tone.wav", original, 16_000)

    loaded, sample_rate = load_audio(path, 16_000)
    assert sample_rate == 16_000
    assert loaded.dtype == np.float32
    assert loaded.ndim == 1
    assert duration_of(loaded, sample_rate) == pytest.approx(0.5, abs=0.01)
    assert np.abs(loaded - original).max() < 1e-3


def test_resampling_to_the_target_rate(tmp_path):
    path = write_wav(tmp_path / "tone.wav", sine(1.0, 300.0, 44_100), 44_100)
    loaded, sample_rate = load_audio(path, 16_000)

    assert sample_rate == 16_000
    assert loaded.size == pytest.approx(16_000, rel=0.01)
    assert duration_of(loaded, sample_rate) == pytest.approx(1.0, abs=0.01)


def test_resample_is_a_no_op_at_the_same_rate():
    samples = sine(0.1, 440.0, 16_000)
    assert resample(samples, 16_000, 16_000) is not None
    assert np.array_equal(resample(samples, 16_000, 16_000), samples)


def test_resample_preserves_signal_energy_roughly():
    samples = sine(0.5, 200.0, 48_000)
    out = resample(samples, 48_000, 16_000)
    assert out.size == pytest.approx(8_000, rel=0.01)
    assert float(np.sqrt(np.mean(out**2))) == pytest.approx(
        float(np.sqrt(np.mean(samples**2))), rel=0.15
    )


def test_stereo_is_mixed_down():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    assert to_mono(stereo).shape == (100,)
    assert np.allclose(to_mono(stereo), 0.5)


def test_slice_audio_returns_the_chunk_and_its_offset():
    samples = np.arange(16_000, dtype=np.float32)
    chunk, start = slice_audio(samples, 16_000, 0.25, 0.5)
    assert start == pytest.approx(0.25)
    assert chunk.size == 4_000


def test_slice_audio_pads_and_clamps_at_the_edges():
    samples = np.arange(16_000, dtype=np.float32)
    chunk, start = slice_audio(samples, 16_000, 0.0, 0.1, pad=0.5)
    assert start == 0.0
    assert chunk.size == pytest.approx(16_000 * 0.6, abs=1)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_audio("does-not-exist.wav")
