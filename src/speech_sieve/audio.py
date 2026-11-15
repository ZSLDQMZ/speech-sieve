"""Safe cross-platform audio decoding and sample-geometry helpers."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


class AudioLoadError(ValueError):
    """Structured error raised when audio cannot be loaded or fails specification checks."""


def _pcm_bytes_to_float32(raw: bytes, sampwidth: int, channels: int) -> np.ndarray:
    """Decode little-endian integer PCM bytes (1/2/3/4 bytes/sample) to float32 [-1, 1]."""
    if sampwidth == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        arr = signed.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioLoadError(f"unsupported_pcm_sample_width:{sampwidth}")
    return arr.reshape(-1, channels)


def _read_raw_audio(path: Path) -> tuple[np.ndarray, int]:
    """Read audio via soundfile -> stdlib wave -> scipy.io.wavfile fallback chain."""
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return np.asarray(data, dtype=np.float32), int(sr)
    except ImportError:
        pass
    except Exception as exc:
        raise AudioLoadError(f"audio_decode_failed:{type(exc).__name__}") from exc

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wf:
                sr = wf.getframerate()
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(wf.getnframes())
            return _pcm_bytes_to_float32(raw, sampwidth, channels), int(sr)
        except wave.Error:
            pass

        try:
            from scipy.io import wavfile  # type: ignore

            sr, data = wavfile.read(str(path))
            arr = np.asarray(data)
            if arr.ndim == 1:
                arr = arr[:, None]
            if np.issubdtype(arr.dtype, np.integer):
                scale = float(np.iinfo(arr.dtype).max) + 1.0
                arr = arr.astype(np.float32) / scale
            else:
                arr = arr.astype(np.float32)
            return arr, int(sr)
        except ImportError:
            pass
        except Exception as exc:
            raise AudioLoadError(f"audio_decode_failed:{type(exc).__name__}") from exc

    raise AudioLoadError(
        "unsupported_audio_backend: install optional 'speech-sieve[soundfile]' for non-PCM formats"
    )


def _linear_resample(signal: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lightweight deterministic NumPy linear resampler used only when allow_resample=True."""
    if src_sr == dst_sr or len(signal) == 0:
        return signal
    dst_len = max(1, round(len(signal) * dst_sr / src_sr))
    src_pos = np.linspace(0.0, len(signal) - 1, num=dst_len, dtype=np.float64)
    return np.interp(src_pos, np.arange(len(signal), dtype=np.float64), signal).astype(np.float32)


def load_audio(path: str | Path, audio_cfg: dict[str, Any]) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Decode audio and apply explicit channel/sample-rate/finite/duration policies."""
    file_path = Path(path)
    if not file_path.exists():
        raise AudioLoadError("audio_file_not_found")

    data, orig_sr = _read_raw_audio(file_path)
    if data.ndim != 2 or len(data) == 0:
        raise AudioLoadError("empty_audio")
    if not np.isfinite(data).all():
        raise AudioLoadError("nonfinite_audio")

    orig_channels = int(data.shape[1])
    target_sr = int(audio_cfg.get("target_sample_rate", 16000))
    require_mono = bool(audio_cfg.get("require_mono", True))
    allow_downmix = bool(audio_cfg.get("allow_downmix", False))
    allow_resample = bool(audio_cfg.get("allow_resample", False))
    max_duration_s = float(audio_cfg.get("max_duration_s", 1800.0))

    downmixed = False
    if orig_channels > 1:
        if allow_downmix:
            mono = data.mean(axis=1, dtype=np.float32)
            downmixed = True
        elif require_mono:
            raise AudioLoadError("unsupported_audio_specification")
        else:
            mono = data[:, 0].astype(np.float32, copy=False)
    else:
        mono = data[:, 0].astype(np.float32, copy=False)

    resampled = False
    sr = orig_sr
    if orig_sr != target_sr:
        if allow_resample:
            mono = _linear_resample(mono, orig_sr, target_sr)
            sr = target_sr
            resampled = True
        elif audio_cfg.get("strict_sample_rate", True):
            raise AudioLoadError("unsupported_audio_specification")

    duration_s = len(mono) / sr
    if duration_s > max_duration_s:
        raise AudioLoadError("duration_cap")

    meta = {
        "original_sample_rate": orig_sr,
        "sample_rate": sr,
        "original_channels": orig_channels,
        "downmixed": downmixed,
        "resampled": resampled,
        "duration_s": float(duration_s),
    }
    return np.ascontiguousarray(mono, dtype=np.float32), sr, meta


def span(start: int, end: int, n: int, sr: int) -> dict[str, Any]:
    """Build a bounded sample/second interval dictionary inside [0, n]."""
    if not (0 <= start <= end <= n):
        raise ValueError("interval_outside_original_audio")
    return {
        "start_sample": int(start),
        "end_sample": int(end),
        "sample_rate": int(sr),
        "start_s": float(start / sr),
        "end_s": float(end / sr),
    }


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index runs where boolean mask is True."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return []
    edges = np.diff(np.r_[False, arr, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def windows(n: int, width: int, hop: int) -> list[int]:
    """Return sliding window start sample indices covering [0, n)."""
    if n <= width:
        return [0]
    result = list(range(0, n - width + 1, hop))
    if result[-1] != n - width:
        result.append(n - width)
    return result
