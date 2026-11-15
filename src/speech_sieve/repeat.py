"""Pure-NumPy multi-scale MFCC + subsequence DTW self-comparison repetition operator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _norm_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def _dct_ortho_matrix(n_mel: int, n_mfcc: int) -> np.ndarray:
    """Exact orthonormal Type-II DCT projection rows [1..n_mfcc] without SciPy."""
    n = np.arange(n_mel, dtype=np.float64)[None, :]
    k = np.arange(1, n_mfcc + 1, dtype=np.float64)[:, None]
    mat = np.sqrt(2.0 / n_mel) * np.cos(np.pi * k * (n + 0.5) / n_mel)
    return mat.astype(np.float32)


def _compute_mfcc(audio: np.ndarray, sr: int, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int, int]:
    frame_samples = max(1, round(float(cfg["frame_ms"]) * sr / 1000.0))
    hop_samples = max(1, round(float(cfg["hop_ms"]) * sr / 1000.0))
    if len(audio) < frame_samples:
        return (
            np.empty((0, int(cfg["mfcc_dimensions"])), dtype=np.float32),
            np.zeros(0, dtype=bool),
            frame_samples,
            hop_samples,
        )

    nfft = int(cfg["nfft"])
    if nfft < frame_samples:
        nfft = 1 << (frame_samples - 1).bit_length()

    mel_bands = int(cfg["mel_bands"])
    mfcc_dims = int(cfg["mfcc_dimensions"])
    mel_lo = float(cfg.get("mel_low_hz", 60.0))
    mel_hi = min(float(cfg.get("mel_high_hz", 7600.0)), sr * 0.49)
    if mel_lo >= mel_hi:
        raise ValueError("invalid_mel_frequency_range")

    framed = np.lib.stride_tricks.sliding_window_view(audio, frame_samples)[::hop_samples]
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    points = hz(np.linspace(mel(mel_lo), mel(mel_hi), mel_bands + 2))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    filt = np.array(
        [
            np.maximum(0.0, np.minimum((freqs - a) / max(b - a, 1e-9), (z - freqs) / max(z - b, 1e-9)))
            for a, b, z in zip(points[:-2], points[1:-1], points[2:])
        ],
        dtype=np.float32,
    )
    win = np.hanning(frame_samples).astype(np.float32)
    dct_mat = _dct_ortho_matrix(mel_bands, mfcc_dims)
    batch = int(cfg.get("feature_batch_frames", 4096))

    cepstra: list[np.ndarray] = []
    levels: list[np.ndarray] = []
    for a in range(0, len(framed), batch):
        block = framed[a : a + batch]
        levels.append(20.0 * np.log10(np.maximum(np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1)), 1e-10)))
        power = np.abs(np.fft.rfft(block * win, n=nfft, axis=1)) ** 2
        log_mel = np.log(np.maximum(power @ filt.T, 1e-10)).astype(np.float32)
        cepstra.append(log_mel @ dct_mat.T)

    x = np.concatenate(cepstra, axis=0)
    db = np.concatenate(levels, axis=0)
    active = db >= max(float(cfg["energy_absolute_dbfs"]), float(db.max()) + float(cfg["energy_relative_db"]))
    source = x[active] if active.any() else x
    x = x - source.mean(axis=0)
    return _norm_rows(x).astype(np.float32), active, frame_samples, hop_samples


def _frame_span(start: int, end: int, n: int, sr: int, frame_samples: int, hop_samples: int) -> dict[str, Any]:
    a = start * hop_samples
    b = min(n, (end - 1) * hop_samples + frame_samples)
    if not (0 <= a < b <= n):
        raise ValueError("repeat_span_out_of_bounds")
    return {
        "start_sample": int(a),
        "end_sample": int(b),
        "start_s": float(a / sr),
        "end_s": float(b / sr),
    }


def _retrieve_candidates(
    x: np.ndarray,
    active: np.ndarray,
    hop_sec: float,
    cfg: dict[str, Any],
) -> tuple[list[tuple[float, int, int, int, int]], dict[str, Any]]:
    ds = int(cfg["coarse_downsample"])
    count = len(x) // ds
    if count <= 0:
        return [], {"raw_seed_pairs": 0, "scale_stats": [], "limitations": []}

    z = _norm_rows(x[: count * ds].reshape(count, ds, -1).mean(axis=1))
    activity = active[: count * ds].reshape(count, ds).mean(axis=1)
    dt = ds * hop_sec
    stride = max(1, round(float(cfg["coarse_stride_s"]) / dt))

    seeds: list[tuple[float, int, int, int, int]] = []
    limitations: list[str] = []
    scale_stats: list[dict[str, Any]] = []
    min_active = float(cfg["min_active_fraction"])
    min_var = float(cfg["min_temporal_variation"])
    bins = int(cfg["descriptor_bins"])
    nonoverlap_frames = math.ceil(float(cfg["nonoverlap_gap_s"]) / hop_sec)

    def _descriptors(length: int) -> tuple[np.ndarray | None, np.ndarray]:
        desc: list[np.ndarray] = []
        starts: list[int] = []
        for a in range(0, max(0, len(z) - length + 1), stride):
            part = z[a : a + length]
            if activity[a : a + length].mean() < min_active:
                continue
            if float(np.mean(1.0 - part @ _norm_rows(part.mean(axis=0)))) < min_var:
                continue
            desc.append(np.concatenate([p.mean(axis=0) for p in np.array_split(part, bins)]))
            starts.append(a * ds)
        return (_norm_rows(np.asarray(desc, dtype=np.float32)) if desc else None), np.asarray(starts, dtype=np.int64)

    max_desc = int(cfg["max_coarse_descriptors_per_scale"])
    block_rows = int(cfg["coarse_block_rows"])
    k_cap = int(cfg["neighbors_per_seed"])
    dist_cap = float(cfg["coarse_retrieval_distance_cap"])
    max_pairs = int(cfg["max_dtw_pairs"])

    for seconds in cfg["seed_lengths_s"]:
        length = max(bins, round(float(seconds) / dt))
        if len(z) // stride > max_desc:
            limitations.append(f"descriptor_capacity_{seconds}")
            continue
        query, starts = _descriptors(length)
        if query is None:
            scale_stats.append({"seed_s": seconds, "query_windows": 0, "target_windows": 0})
            continue
        targets: list[np.ndarray] = []
        where_list: list[int] = []
        widths_list: list[int] = []
        for ratio in cfg["target_duration_ratios"]:
            width = max(bins, round(length * float(ratio)))
            desc, pos = _descriptors(width)
            if desc is not None:
                targets.append(desc)
                where_list.extend(pos.tolist())
                widths_list.extend([width * ds] * len(pos))
        if not targets:
            scale_stats.append({"seed_s": seconds, "query_windows": len(starts), "target_windows": 0})
            continue

        target = np.concatenate(targets, axis=0)
        where = np.asarray(where_list, dtype=np.int64)
        widths = np.asarray(widths_list, dtype=np.int64)
        scale_stats.append({"seed_s": seconds, "query_windows": int(len(starts)), "target_windows": int(len(where))})

        for a in range(0, len(starts), block_rows):
            stop = min(len(starts), a + block_rows)
            distances = np.maximum(0.0, 1.0 - query[a:stop] @ target.T)
            valid = where[None, :] >= starts[a:stop, None] + length * ds + nonoverlap_frames
            distances[~valid] = np.inf
            k = min(k_cap, len(where))
            neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
            for row, ids in enumerate(neighbors):
                for j in ids:
                    val = float(distances[row, j])
                    if val <= dist_cap:
                        seeds.append((val, int(starts[a + row]), int(where[j]), length * ds, int(widths[j])))

    seeds.sort()
    unique: list[tuple[float, int, int, int, int]] = []
    for seed in seeds:
        if any(
            abs(seed[1] - p[1]) < min(seed[3], p[3]) * 0.5 and abs(seed[2] - p[2]) < min(seed[4], p[4]) * 0.5
            for p in unique
        ):
            continue
        unique.append(seed)
        if len(unique) >= max_pairs:
            if len(seeds) > len(unique):
                limitations.append("dtw_candidate_cap")
            break
    return unique, {"raw_seed_pairs": len(seeds), "scale_stats": scale_stats, "limitations": limitations}


def subsequence_dtw(query: np.ndarray, target: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Bounded local-slope subsequence DTW returning exact target frame endpoints."""
    n, m = len(query), len(target)
    if n == 0 or m == 0:
        return None
    costs = np.clip(1.0 - query @ target.T, 0.0, 2.0)
    d = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    d[0, :] = 0.0
    length = np.zeros((n + 1, m + 1), dtype=np.int32)
    start = np.zeros((n + 1, m + 1), dtype=np.int32)
    start[0, :] = np.arange(m + 1, dtype=np.int32)
    back = np.zeros((n + 1, m + 1), dtype=np.uint8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c_ij = float(costs[i - 1, j - 1])
            best = d[i - 1, j - 1] + c_ij
            step = 1
            pi, pj, extra = i - 1, j - 1, 1
            if j >= 2:
                val = d[i - 1, j - 2] + float(costs[i - 1, j - 2]) + c_ij
                if val < best:
                    best, step, pi, pj, extra = val, 2, i - 1, j - 2, 2
            if i >= 2:
                val = d[i - 2, j - 1] + float(costs[i - 2, j - 1]) + c_ij
                if val < best:
                    best, step, pi, pj, extra = val, 3, i - 2, j - 1, 2
            d[i, j] = best
            back[i, j] = step
            length[i, j] = length[pi, pj] + extra
            start[i, j] = start[pi, pj]

    ratio_lo, ratio_hi = cfg["matched_duration_ratio"]
    options = []
    for j in range(1, m + 1):
        ratio = (j - int(start[n, j])) / n
        if ratio_lo <= ratio <= ratio_hi and np.isfinite(d[n, j]):
            options.append((float(d[n, j] / max(1, int(length[n, j]))), j))
    if not options:
        return None

    mean_dist, end = min(options)
    i, j = n, end
    path: list[tuple[int, int]] = []
    while i > 0:
        step = int(back[i, j])
        if step == 1:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == 2:
            path.extend([(i - 1, j - 1), (i - 1, j - 2)])
            i -= 1
            j -= 2
        elif step == 3:
            path.extend([(i - 1, j - 1), (i - 2, j - 1)])
            i -= 2
            j -= 1
        else:
            raise ValueError("invalid_dtw_backtrack")

    vals = np.asarray([float(costs[a, b]) for a, b in path], dtype=np.float64)
    return {
        "mean_distance": float(mean_dist),
        "target_start": int(j),
        "target_end": int(end),
        "path_frames": len(path),
        "good_path_fraction": float(np.mean(vals <= float(cfg["good_frame_distance_cap"]))),
        "matched_duration_ratio": float((end - j) / n),
    }


def analyse_repeat(audio: np.ndarray, sr: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run MFCC+DTW self-comparison acoustic repeat detection with dynamic hop_sec."""
    rcfg = cfg["repeat"]
    if not rcfg.get("enabled", True):
        return {
            "status": "not_evaluated",
            "reason": "repeat_disabled",
            "candidates": [],
            "comparisons": [],
            "limitations": [],
        }

    if len(audio) / sr > float(rcfg["max_duration_s"]):
        return {"status": "not_evaluated", "reason": "duration_cap", "candidates": [], "comparisons": [], "limitations": []}

    frame_samples = max(1, round(float(rcfg["frame_ms"]) * sr / 1000.0))
    if len(audio) < frame_samples:
        return {
            "status": "not_evaluated",
            "reason": "shorter_than_frame",
            "candidates": [],
            "comparisons": [],
            "limitations": [],
        }

    x, active, frame_samples, hop_samples = _compute_mfcc(audio, sr, rcfg)
    hop_sec = float(hop_samples / sr)
    out: dict[str, Any] = {
        "status": "evaluated",
        "candidates": [],
        "comparisons": [],
        "limitations": [],
        "hop_sec": hop_sec,
        "active_frame_fraction": float(active.mean()) if len(active) else 0.0,
        "semantic_status": "unreviewed_similarity_only",
        "candidate_interpretation": "similarity_seed_pairs_with_free_target_endpoints_not_complete_words_or_extra_reading_verdict",
    }
    if not active.any():
        out.update(status="not_evaluated", reason="no_active_frames")
        return out
    if len(x) < round(min(float(s) for s in rcfg["seed_lengths_s"]) / hop_sec):
        out.update(status="not_evaluated", reason="shorter_than_seed")
        return out

    seeds, info = _retrieve_candidates(x, active, hop_sec, rcfg)
    out.update(info)

    margin = round(float(rcfg["target_boundary_margin_s"]) / hop_sec)
    gap = round(float(rcfg["nonoverlap_gap_s"]) / hop_sec)
    dtw_cap = float(rcfg["dtw_mean_distance_cap"])
    good_frac_min = float(rcfg["min_good_path_fraction"])
    min_active = float(rcfg["min_active_fraction"])

    for score, a, b, la, lb in seeds:
        lo = max(a + la + gap, b - margin)
        hi = min(len(x), b + lb + margin)
        if hi <= lo:
            continue
        res = subsequence_dtw(x[a : a + la], x[lo:hi], rcfg)
        if res is None:
            continue
        target_a = lo + res["target_start"]
        target_b = lo + res["target_end"]
        res.update(
            first=_frame_span(a, a + la, len(audio), sr, frame_samples, hop_samples),
            second=_frame_span(target_a, target_b, len(audio), sr, frame_samples, hop_samples),
            coarse_distance=float(score),
            target_active_fraction=float(active[target_a:target_b].mean()),
        )
        res["passed"] = bool(
            res["mean_distance"] <= dtw_cap
            and res["good_path_fraction"] >= good_frac_min
            and res["target_active_fraction"] >= min_active
        )
        out["comparisons"].append(res)
        if res["passed"]:
            out["candidates"].append(res)

    if out["limitations"]:
        out["status"] = "partially_evaluated"
    out["dtw_pairs_checked"] = len(out["comparisons"])
    return out
