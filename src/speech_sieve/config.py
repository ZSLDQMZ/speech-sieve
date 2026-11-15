"""Preset loading, schema validation, and cross-platform atomic JSON/JSONL helpers."""

from __future__ import annotations

import copy
import errno
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class ConfigError(ValueError):
    """Raised when preset configuration violates schema or physical invariants."""


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "preset_name": "default",
    "mode": "scripted_reading",
    "audio": {
        "target_sample_rate": 16000,
        "require_mono": True,
        "allow_resample": False,
        "allow_downmix": False,
        "strict_sample_rate": True,
        "max_duration_s": 1800.0,
        "near_full_scale_threshold": 0.999,
    },
    "features": {
        "frame_ms": 20.0,
        "hop_ms": 10.0,
        "spectral_frame_ms": 25.0,
        "nfft": 2048,
        "spectral_batch_frames": 2048,
        "min_spectral_voice_frames": 50,
        "spectral_bands": {
            "noise_low_hz": 300.0,
            "noise_high_hz": 7900.0,
            "below_hz": 300.0,
            "above_hz": 3000.0,
            "peak_low_hz": 60.0,
            "peak_high_hz": 1000.0,
        },
        "rhythm_depths_db": [12, 18, 24],
        "rhythm_dip_range_s": [0.12, 1.2],
        "rhythm_min_burst_s": 0.12,
        "speech_level_window_s": 3.0,
        "speech_level_min_voice_s": 1.5,
        "speech_level_min_windows": 3,
    },
    "repeat": {
        "enabled": True,
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "nfft": 512,
        "mel_bands": 40,
        "mfcc_dimensions": 13,
        "mel_low_hz": 60.0,
        "mel_high_hz": 7600.0,
        "feature_batch_frames": 4096,
        "energy_absolute_dbfs": -65.0,
        "energy_relative_db": -35.0,
        "coarse_downsample": 4,
        "coarse_stride_s": 0.12,
        "seed_lengths_s": [0.48, 0.96, 1.92],
        "target_duration_ratios": [0.8, 1.0, 1.25],
        "descriptor_bins": 4,
        "min_active_fraction": 0.65,
        "min_temporal_variation": 0.025,
        "coarse_retrieval_distance_cap": 0.45,
        "coarse_block_rows": 128,
        "neighbors_per_seed": 2,
        "nonoverlap_gap_s": 0.08,
        "max_dtw_pairs": 80,
        "target_boundary_margin_s": 0.24,
        "matched_duration_ratio": [0.67, 1.5],
        "dtw_mean_distance_cap": 0.22,
        "good_frame_distance_cap": 0.35,
        "min_good_path_fraction": 0.75,
        "max_coarse_descriptors_per_scale": 6000,
        "per_file_budget_s": 30.0,
        "max_duration_s": 1800.0,
    },
    "alignment": {
        "enabled": True,
        "token_unit": "auto",
        "coverage_margins_s": [0.2, 0.5],
        "window_s": 5.0,
        "window_step_s": 2.0,
        "min_window_voice_s": 2.0,
        "cross_coverages": [0.6, 0.8, 0.9],
        "cross_default_coverage": 0.8,
        "cross_near_zero_fraction": 0.2,
        "cross_min_voice_s": 0.2,
        "cross_outside_hull_margin_s": 0.5,
    },
    "topic_prefix": {
        "enabled": False,
        "strip_question_prefixes": [],
        "explicit_cues": [],
        "fillers": "",
        "punctuation_split": r"([\n,!?.;:，。！？；：])",
        "min_lead_len": 5,
        "min_fuzzy_len": 5,
        "fuzzy_ratio_min": 0.65,
        "near_start_max_tokens": 12,
        "filler_max_tokens": 6,
    },
    "models": {
        "device": "auto",
        "vad": {
            "enabled": True,
            "backend": "silero",
            "weight_path": None,
            "threshold_on": 0.5,
            "threshold_off": 0.35,
            "min_silence_ms": 100.0,
            "min_speech_ms": 32.0,
            "batch_frame_samples": 512,
            "batch_size": 32,
        },
        "aligner": {
            "enabled": True,
            "backend": "qwen",
            "weight_path": None,
            "language": None,
            "dtype": "auto",
            "attn_implementation": "sdpa",
            "min_duration_s": 0.0125,
            "max_duration_s": 300.0,
            "tiers": [[9.0, 8], [30.0, 4], [300.0, 1]],
        },
        "overlap": {
            "enabled": True,
            "backend": "pyannote",
            "weight_path": None,
            "window_s": 10.0,
            "hop_s": 5.0,
            "grid_step_ms": 10.0,
            "probability_threshold": 0.5,
            "batch_size": 16,
            "row_chunk_size": 64,
        },
    },
    "gates": {
        "audio_integrity": {
            "enabled": True,
            "severity": "alert",
            "maximum_clipping_fraction": 0.001,
            "minimum_clipped_run_samples": 4,
        },
        "signal_level_low": {"enabled": True, "severity": "alert", "maximum_dbfs": -35.0},
        "signal_level_variance": {"enabled": True, "severity": "alert", "minimum_db": 12.0},
        "alignment_coverage_gap": {
            "enabled": True,
            "severity": "alert",
            "margin_s": 0.5,
            "minimum_unexplained_voice_s": 10.0,
            "minimum_unexplained_fraction": 0.2,
        },
        "acoustic_repetition": {
            "enabled": True,
            "severity": "alert",
            "coverage": 0.8,
            "outside_hull_margin_s": 0.5,
        },
        "rhythm_cadence_profile": {
            "enabled": False,
            "severity": "review_recommended",
            "energy_depth_db": 18,
            "minimum_extent_s": 5.0,
            "minimum_dip_count": 4,
            "minimum_dips_per_10s": 4.0,
            "minimum_dip_fraction": 0.1,
            "maximum_tokens_per_extent_s": 3.0,
        },
        "spectral_tilt_profile": {
            "enabled": False,
            "severity": "review_recommended",
            "maximum_rolloff95_median_hz": 500.0,
            "maximum_speech_rms_dbfs": -25.0,
            "minimum_voice_frames": 50,
        },
        "overlap_risk": {"enabled": True, "severity": "review_recommended", "minimum_max_probability": 0.8},
        "edge_truncation_risk": {"enabled": True, "severity": "review_recommended", "minimum_seconds": 5.0},
        "excessive_silence": {
            "enabled": True,
            "severity": "alert",
            "maximum_vad_voice_s": 1.0,
            "maximum_vad_voice_fraction": 0.05,
        },
        "non_speech_dominant": {
            "enabled": True,
            "severity": "alert",
            "minimum_vad_voice_s": 20.0,
            "maximum_tokens_per_vad_s": 0.5,
        },
        "prompt_echo_prefix": {"enabled": False, "severity": "review_recommended"},
        "advisory_cer": {"enabled": False, "severity": "advisory"},
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_preset(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON preset file and merge with validated defaults."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"preset_not_found:{p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("preset_must_be_mapping")
    cfg = _deep_merge(DEFAULT_CONFIG, raw)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate structural and numerical consistency of a speech_sieve configuration."""
    if cfg.get("mode") not in ("scripted_reading", "spontaneous"):
        raise ConfigError(f"invalid_mode:{cfg.get('mode')}")

    audio = cfg.get("audio", {})
    sr = int(audio.get("target_sample_rate", 0))
    if sr <= 0:
        raise ConfigError("audio.target_sample_rate_must_be_positive")
    if float(audio.get("max_duration_s", 0)) <= 0:
        raise ConfigError("audio.max_duration_s_must_be_positive")

    feat = cfg.get("features", {})
    if float(feat.get("frame_ms", 0)) <= 0 or float(feat.get("hop_ms", 0)) <= 0:
        raise ConfigError("features.frame_ms_and_hop_ms_must_be_positive")
    spec_samples = round(float(feat.get("spectral_frame_ms", 25.0)) * sr / 1000.0)
    nfft = int(feat.get("nfft", 0))
    if nfft < max(16, spec_samples):
        raise ConfigError(f"features.nfft_too_small:{nfft}<{spec_samples}")
    bands = feat.get("spectral_bands", {})
    if float(bands.get("noise_low_hz", 0)) >= float(bands.get("noise_high_hz", 0)):
        raise ConfigError("features.spectral_bands.noise_range_invalid")
    if float(bands.get("peak_low_hz", 0)) >= float(bands.get("peak_high_hz", 0)):
        raise ConfigError("features.spectral_bands.peak_range_invalid")

    rep = cfg.get("repeat", {})
    if float(rep.get("hop_ms", 0)) <= 0 or float(rep.get("frame_ms", 0)) <= 0:
        raise ConfigError("repeat.frame_ms_and_hop_ms_must_be_positive")
    ratios = rep.get("matched_duration_ratio", [])
    if not (isinstance(ratios, list) and len(ratios) == 2 and 0 < ratios[0] <= ratios[1]):
        raise ConfigError("repeat.matched_duration_ratio_invalid")

    align = cfg.get("alignment", {})
    if align.get("token_unit") not in ("auto", "char", "whitespace"):
        raise ConfigError(f"invalid_token_unit:{align.get('token_unit')}")
    cov_list = align.get("cross_coverages", [])
    if align.get("cross_default_coverage") not in cov_list:
        raise ConfigError("alignment.cross_default_coverage_must_be_in_cross_coverages")

    models = cfg.get("models", {})
    if models.get("device") not in ("auto", "cpu", "cuda", "mps"):
        raise ConfigError(f"invalid_device:{models.get('device')}")


def scalar_json(obj: Any) -> Any:
    """Convert NumPy scalars cleanly for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def fsync_dir(path: Path) -> str:
    """Synchronize directory entry safely across POSIX and Windows NTFS."""
    if os.name == "nt":
        return "fsync_directory_skipped_windows"
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError as error:
        if error.errno in (errno.EACCES, errno.EPERM, errno.EINVAL, errno.ENOTSUP, errno.ENOSYS):
            return "fsync_directory_unsupported"
        raise
    try:
        os.fsync(fd)
        return "fsync_directory_ok"
    except OSError as error:
        if error.errno in (errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP), errno.ENOSYS):
            return "fsync_directory_unsupported"
        raise
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str) -> str:
    """Write text atomically via temp file + os.fsync + os.replace + safe dir sync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return fsync_dir(path.parent)


def write_json_atomic(path: Path, obj: Any) -> str:
    """Write formatted JSON atomically."""
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=scalar_json) + "\n"
    return write_text_atomic(path, payload)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> str:
    """Write JSONL records atomically."""
    lines = [json.dumps(r, ensure_ascii=False, allow_nan=False, default=scalar_json) for r in records]
    payload = ("\n".join(lines) + "\n") if lines else ""
    return write_text_atomic(path, payload)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSON lines from a file."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
