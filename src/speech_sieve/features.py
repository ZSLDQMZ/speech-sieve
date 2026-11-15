"""Pure-NumPy signal, level, rhythm, and spectral profiling operators."""

from __future__ import annotations

from typing import Any

import numpy as np

from .alignment import count_text_tokens
from .audio import runs


def _db(power: float) -> float | None:
    return float(10.0 * np.log10(power)) if power > 0 else None


def analyse_basic(wave: np.ndarray, sr: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute finite waveform amplitude stats and dynamic-width frame mean-square energy."""
    fcfg = cfg["features"]
    acfg = cfg["audio"]
    n = len(wave)
    frame_samples = max(1, round(float(fcfg["frame_ms"]) * sr / 1000.0))
    near_thresh = float(acfg.get("near_full_scale_threshold", 0.999))

    w64 = wave.astype(np.float64, copy=False)
    energy = np.array(
        [np.mean(w64[i : i + frame_samples] ** 2) for i in range(0, n, frame_samples)],
        dtype=np.float32,
    )
    near = np.abs(wave) >= near_thresh
    lengths = [b - a for a, b in runs(near)]
    all_zero = bool(np.all(wave == 0))
    record = {
        "all_zero": all_zero,
        "peak": float(np.max(np.abs(wave))) if n else None,
        "rms": float(np.sqrt(np.mean(w64**2))) if n else None,
        "near_full_scale_fraction": float(near.mean()) if n else None,
        "max_near_full_scale_samples": max(lengths, default=0),
        "flags": ["all_zero"] if n and all_zero else [],
    }
    return record, {"energy_frames": energy, "frame_samples": frame_samples}


def analyse_speech_level(
    energy: np.ndarray,
    n: int,
    sr: int,
    frame_samples: int,
    intervals: list[dict[str, Any]] | None,
    rms: float | None,
    peak: float | None,
    near_full_scale_fraction: float | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute digital speech-RMS levels and windowed level variation."""
    fcfg = cfg["features"]
    starts = np.arange(len(energy), dtype=np.int64) * frame_samples
    ends = np.minimum(n, starts + frame_samples)
    widths = ends - starts
    expected_frames = (n + frame_samples - 1) // frame_samples
    if len(energy) != expected_frames:
        raise ValueError("energy_frame_count_mismatch")
    if not (np.isfinite(energy).all() and np.all(energy >= 0)):
        raise ValueError("invalid_energy_frame")

    power = float(np.dot(energy, widths) / n)
    expected_power = (rms or 0.0) ** 2
    rel = abs(power - expected_power) / max(expected_power, 1e-30)
    if rel >= 1e-4:
        raise ValueError("energy_reconstruction_mismatch")

    whole_db = _db(power)
    window_s = float(fcfg.get("speech_level_window_s", 3.0))
    min_voice_s = float(fcfg.get("speech_level_min_voice_s", 1.5))
    min_windows = int(fcfg.get("speech_level_min_windows", 3))

    if intervals is None:
        return {
            "duration_s": float(n / sr),
            "vad_voice_s": None,
            "whole_rms_dbfs": whole_db,
            "speech_rms_dbfs": None,
            "peak": peak,
            "near_full_scale_fraction": near_full_scale_fraction,
            "state": "not_evaluated_vad_disabled",
            "short_voice_under1s": None,
            "windows": [],
            "macro_dynamic_eligible": False,
            "window_p90_minus_p10_db": None,
            "min_required_windows": min_windows,
            "window_seconds": window_s,
            "metric_note": "digital_signal_level_dbfs_not_lufs_or_snr",
        }

    weights = np.zeros(len(energy), dtype=np.float64)
    for iv in intervals:
        weights += np.maximum(0, np.minimum(ends, iv["end_sample"]) - np.maximum(starts, iv["start_sample"]))
    total_iv_s = sum(iv["end_s"] - iv["start_s"] for iv in intervals)
    if not (np.all(weights <= widths) and abs(weights.sum() / sr - total_iv_s) < 1e-6):
        raise ValueError("vad_weight_inconsistency")

    voice_s = float(weights.sum() / sr)
    speech_db = _db(float(np.dot(energy, weights) / weights.sum())) if weights.sum() > 0 else None
    win_samples = max(1, round(window_s * sr))
    windows_out: list[dict[str, Any]] = []
    for a in range(0, n, win_samples):
        b = min(n, a + win_samples)
        ww = np.zeros(len(energy), dtype=np.float64)
        for iv in intervals:
            lo = max(a, iv["start_sample"])
            hi = min(b, iv["end_sample"])
            if hi > lo:
                ww += np.maximum(0, np.minimum(ends, hi) - np.maximum(starts, lo))
        if ww.sum() / sr >= min_voice_s:
            lvl = _db(float(np.dot(energy, ww) / ww.sum()))
            if lvl is not None:
                windows_out.append(
                    {
                        "start_s": float(a / sr),
                        "end_s": float(b / sr),
                        "voice_s": float(ww.sum() / sr),
                        "speech_rms_dbfs": lvl,
                    }
                )

    levels = [w["speech_rms_dbfs"] for w in windows_out]
    eligible = len(levels) >= min_windows
    out: dict[str, Any] = {
        "duration_s": float(n / sr),
        "vad_voice_s": voice_s,
        "whole_rms_dbfs": whole_db,
        "speech_rms_dbfs": speech_db,
        "peak": peak,
        "near_full_scale_fraction": near_full_scale_fraction,
        "state": "measurable" if weights.sum() > 0 else "no_vad_speech",
        "short_voice_under1s": voice_s < 1.0,
        "windows": windows_out,
        "macro_dynamic_eligible": eligible,
        "window_p90_minus_p10_db": (
            float(np.percentile(levels, 90) - np.percentile(levels, 10)) if eligible else None
        ),
        "min_required_windows": min_windows,
        "window_seconds": window_s,
        "metric_note": "digital_signal_level_dbfs_not_lufs_or_snr",
    }
    if speech_db is not None and whole_db is not None:
        out["speech_minus_whole_db"] = float(speech_db - whole_db)
    return out


def analyse_rhythm(
    energy: np.ndarray,
    n: int,
    sr: int,
    frame_samples: int,
    intervals: list[dict[str, Any]] | None,
    reference_text: str | None,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Profile relative energy dips within active extent using dynamic frame duration."""
    if intervals is None:
        return None, "vad_disabled"
    if not intervals:
        return None, "no_vad_speech"

    fcfg = cfg["features"]
    token_unit = cfg.get("alignment", {}).get("token_unit", "auto")
    smoothed = np.convolve(np.pad(energy, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
    if len(smoothed) != len(energy):
        raise ValueError("smoothing_length_mismatch")

    starts = np.arange(len(energy), dtype=np.float64) * frame_samples
    ends = np.minimum(float(n), starts + frame_samples)
    centers = (starts + ends) / (2.0 * sr)
    frame_s = float(frame_samples / sr)
    half_frame_s = 0.5 * frame_s

    first = float(intervals[0]["start_s"])
    last = float(intervals[-1]["end_s"])
    span_s = last - first
    if span_s <= 0:
        return None, "nonpositive_extent_duration"

    voiced = np.zeros(len(energy), dtype=bool)
    for iv in intervals:
        voiced |= (centers >= iv["start_s"]) & (centers < iv["end_s"])
    if not voiced.any():
        return None, "no_voiced_frames"

    ref = float(np.percentile(smoothed[voiced], 75))
    if ref <= 0:
        return None, "nonpositive_reference_power"

    inside = (centers >= first) & (centers < last)
    env = smoothed[inside]
    times = centers[inside]
    if env.size == 0:
        return None, "empty_extent_frames"

    dip_lo, dip_hi = fcfg.get("rhythm_dip_range_s", [0.12, 1.2])
    min_burst_s = float(fcfg.get("rhythm_min_burst_s", 0.12))
    depths = [int(d) for d in fcfg.get("rhythm_depths_db", [12, 18, 24])]
    tokens = count_text_tokens(reference_text or "", unit=token_unit)

    record: dict[str, Any] = {
        "active_extent_s": [first, last],
        "extent_duration_s": float(span_s),
        "reference_token_count": int(tokens),
        "tokens_per_extent_s": float(tokens / span_s),
        "relative_reference_power_p75": ref,
        "frame_duration_s": frame_s,
        "settings": {},
    }

    for depth in depths:
        high = env >= ref * (10.0 ** (-depth / 10.0))
        dips = [(a, z) for a, z in runs(~high) if a > 0 and z < len(high)]
        short = [(a, z) for a, z in dips if dip_lo - 1e-8 <= (z - a) * frame_s <= dip_hi + 1e-8]
        burst_runs = [(a, z) for a, z in runs(high) if (z - a) * frame_s >= min_burst_s - 1e-8]
        bursts = [(z - a) * frame_s for a, z in burst_runs]
        onsets = [float(times[a] - half_frame_s) for a, _ in burst_runs]
        cycles = np.diff(onsets) if len(onsets) >= 2 else np.empty(0, dtype=np.float64)
        mean_cycle = float(np.mean(cycles)) if len(cycles) > 0 else 0.0
        record["settings"][str(depth)] = {
            "short_dip_count": len(short),
            "short_dips_per10s": float(len(short) * 10.0 / span_s),
            "short_dip_fraction": float(sum((z - a) * frame_s for a, z in short) / span_s),
            "burst_count_at_least_min": len(bursts),
            "burst_median_s": float(np.median(bursts)) if bursts else None,
            "onset_interval_cv": (
                float(np.std(cycles) / mean_cycle) if len(cycles) >= 3 and mean_cycle > 0 else None
            ),
            "short_dip_intervals_s": [
                [max(first, float(times[a] - half_frame_s)), min(last, float(times[z - 1] + half_frame_s))]
                for a, z in short
            ],
        }
    return record, None


def analyse_spectral(
    wave: np.ndarray,
    sr: int,
    intervals: list[dict[str, Any]] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute continuous spectral descriptors with configurable NFFT and Nyquist clamping."""
    fcfg = cfg["features"]
    frame_samples = max(1, round(float(fcfg.get("spectral_frame_ms", 25.0)) * sr / 1000.0))
    hop_samples = max(1, round(float(fcfg.get("hop_ms", 10.0)) * sr / 1000.0))
    nfft = int(fcfg.get("nfft", 2048))
    if nfft < frame_samples:
        nfft = 1 << (frame_samples - 1).bit_length()
    batch_frames = int(fcfg.get("spectral_batch_frames", 2048))
    min_vad_frames = int(fcfg.get("min_spectral_voice_frames", 50))

    if len(wave) < frame_samples:
        return {
            "run_status": "not_evaluated",
            "reason": "shorter_than_spectral_frame",
            "nfft": nfft,
            "freq_step_hz": float(sr / nfft),
            "joint_frame_fraction": None,
            "high_frequency_navigation": [],
            "voice": None,
            "nonvoice": None,
            "full_signal": None,
            "voice_to_nonvoice_median_power_gap_db": None,
            "duration_s": float(len(wave) / sr),
        }

    bands = fcfg["spectral_bands"]
    nyquist_safe = sr * 0.49
    noise_lo = float(bands["noise_low_hz"])
    noise_hi = min(float(bands["noise_high_hz"]), nyquist_safe)
    below_hz = min(float(bands["below_hz"]), nyquist_safe)
    above_hz_cfg = float(bands["above_hz"])
    above_evaluatable = above_hz_cfg < (sr * 0.5)
    peak_lo = float(bands["peak_low_hz"])
    peak_hi = min(float(bands["peak_high_hz"]), nyquist_safe)

    freq = np.fft.rfftfreq(nfft, 1.0 / sr)
    window = np.hanning(frame_samples).astype(np.float32)
    n = 1 + (len(wave) - frame_samples) // hop_samples
    times = (np.arange(n) * hop_samples + frame_samples / 2.0) / sr

    if intervals is None:
        active = np.ones(n, dtype=bool)
    else:
        active = np.zeros(n, dtype=bool)
        for iv in intervals:
            active |= (times >= iv["start_s"]) & (times < iv["end_s"])

    noise_mask = (freq >= noise_lo) & (freq <= noise_hi)
    peak_mask = (freq >= peak_lo) & (freq <= peak_hi)
    centers = freq[1:-1]
    triple_mask = (centers >= peak_lo) & (centers <= peak_hi)
    below_mask = freq < below_hz
    above_mask = freq >= above_hz_cfg

    metric_names = [
        "power",
        "flatness_band",
        "rolloff95_hz",
        "below_band_fraction",
        "above_band_fraction",
        "narrow3bin_fraction",
    ]
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in metric_names}
    groups = ("full_signal",) if intervals is None else ("voice", "nonvoice")
    mean_spec = {label: np.zeros(len(freq), dtype=np.float64) for label in groups}
    counts = {label: 0 for label in groups}

    for first in range(0, n, batch_frames):
        last = min(n, first + batch_frames)
        index = np.arange(first, last)[:, None] * hop_samples + np.arange(frame_samples)[None, :]
        x = wave[index]
        power = np.mean(x.astype(np.float64) ** 2, axis=1)
        spec = np.abs(np.fft.rfft(x * window, n=nfft, axis=1)) ** 2
        total = spec.sum(axis=1)
        valid = total > 1e-20
        safe = np.maximum(total, 1e-20)

        if noise_mask.any():
            noiseband = spec[:, noise_mask]
            flat = np.exp(np.mean(np.log(np.maximum(noiseband, 1e-20)), axis=1)) / np.maximum(
                noiseband.mean(axis=1), 1e-20
            )
            flat[~valid] = np.nan
        else:
            flat = np.full(len(x), np.nan, dtype=np.float64)

        cumulative = np.cumsum(spec, axis=1)
        roll = freq[(cumulative >= 0.95 * total[:, None]).argmax(axis=1)]
        roll[~valid] = np.nan

        if triple_mask.any():
            triples = spec[:, :-2] + spec[:, 1:-1] + spec[:, 2:]
            narrow = triples[:, triple_mask].max(axis=1) / safe
            narrow[~valid] = np.nan
        else:
            narrow = np.full(len(x), np.nan, dtype=np.float64)

        below_frac = spec[:, below_mask].sum(axis=1) / safe if below_mask.any() else np.zeros(len(x))
        above_frac = (
            spec[:, above_mask].sum(axis=1) / safe
            if above_evaluatable and above_mask.any()
            else np.full(len(x), np.nan, dtype=np.float64)
        )

        for key, vals in (
            ("power", power),
            ("flatness_band", flat),
            ("rolloff95_hz", roll),
            ("below_band_fraction", below_frac),
            ("above_band_fraction", above_frac),
            ("narrow3bin_fraction", narrow),
        ):
            arrays[key].append(vals)

        if intervals is None:
            mean_spec["full_signal"] += spec.sum(axis=0)
            counts["full_signal"] += len(spec)
        else:
            for label, mask in (("voice", active[first:last]), ("nonvoice", ~active[first:last])):
                if mask.any():
                    mean_spec[label] += spec[mask].sum(axis=0)
                    counts[label] += int(mask.sum())

    joined = {k: np.concatenate(v) for k, v in arrays.items()}

    def _summarize(mask: np.ndarray, label: str) -> dict[str, Any]:
        summary: dict[str, Any] = {"frames": int(mask.sum())}
        for k, vals in joined.items():
            sub = vals[mask]
            sub = sub[np.isfinite(sub)]
            summary[k] = (
                {
                    "p10": float(np.percentile(sub, 10)),
                    "median": float(np.median(sub)),
                    "p90": float(np.percentile(sub, 90)),
                }
                if len(sub)
                else None
            )
        z = mean_spec[label]
        idx = np.where(peak_mask)[0]
        summary["mean_spectrum_peak_hz"] = (
            float(freq[idx[np.argmax(z[idx])]]) if counts[label] and idx.size and z.sum() > 1e-20 else None
        )
        return summary

    if intervals is None:
        return {
            "run_status": "described_no_vad",
            "nfft": nfft,
            "freq_step_hz": float(sr / nfft),
            "nyquist_hz": float(sr / 2.0),
            "clamped_bands_hz": {"noise_high_hz": noise_hi, "peak_high_hz": peak_hi, "above_hz_evaluated": above_evaluatable},
            "joint_frame_fraction": None,
            "high_frequency_navigation": [],
            "voice": None,
            "nonvoice": None,
            "full_signal": _summarize(active, "full_signal"),
            "voice_to_nonvoice_median_power_gap_db": None,
            "duration_s": float(len(wave) / sr),
        }

    descriptions = {label: _summarize(mask, label) for label, mask in (("voice", active), ("nonvoice", ~active))}
    vp = joined["power"][active]
    nvp = joined["power"][~active]
    gap = (
        float(10.0 * np.log10(np.median(vp) / np.median(nvp)))
        if len(vp) >= min_vad_frames and len(nvp) >= min_vad_frames and np.median(vp) > 0 and np.median(nvp) > 1e-15
        else None
    )

    if above_evaluatable:
        above_vals = np.nan_to_num(joined["above_band_fraction"], nan=-1.0)
        flat_vals = np.nan_to_num(joined["flatness_band"], nan=-1.0)
        joint = active & (above_vals >= 0.2) & (flat_vals >= 0.05)
    else:
        joint = np.zeros(n, dtype=bool)

    ix = np.flatnonzero(joint)
    segments: list[list[float]] = []
    if len(ix):
        breaks = np.where(np.diff(ix) > 1)[0]
        seg_starts = np.r_[ix[0], ix[breaks + 1]]
        seg_ends = np.r_[ix[breaks], ix[-1]]
        segments = sorted(
            [[max(0.0, float(times[s] - 0.005)), min(len(wave) / sr, float(times[e] + 0.005))] for s, e in zip(seg_starts, seg_ends)],
            key=lambda seg: -(seg[1] - seg[0]),
        )[:10]

    status = (
        "described"
        if int(active.sum()) >= min_vad_frames
        else "short_voice"
        if active.any()
        else "no_vad_voice"
    )
    return {
        "run_status": status,
        "nfft": nfft,
        "freq_step_hz": float(sr / nfft),
        "nyquist_hz": float(sr / 2.0),
        "clamped_bands_hz": {"noise_high_hz": noise_hi, "peak_high_hz": peak_hi, "above_hz_evaluated": above_evaluatable},
        "joint_frame_fraction": float(joint.sum() / active.sum()) if active.any() else None,
        "high_frequency_navigation": segments,
        "voice": descriptions["voice"],
        "nonvoice": descriptions["nonvoice"],
        "full_signal": None,
        "voice_to_nonvoice_median_power_gap_db": gap,
        "duration_s": float(len(wave) / sr),
    }
