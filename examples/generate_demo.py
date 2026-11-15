"""Generate synthetic WAV fixtures and a manifest.jsonl for zero-corpus demo & testing."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Sequence

import numpy as np


def _write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float32 [-1, 1] mono (N,) or multi-channel (N, C) audio as 16-bit PCM WAV."""
    arr = np.asarray(audio, dtype=np.float32)
    clipped = np.clip(arr, -1.0, 0.99996948)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    channels = 1 if pcm.ndim == 1 else int(pcm.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())


def _nonrepeating_speech(duration_s: float, sr: int, seed: int, amp: float = 0.28) -> np.ndarray:
    """Synthesize non-repeating speech-like formant syllables across disjoint Mel bands."""
    rng = np.random.default_rng(seed)
    centers = [180.0, 320.0, 500.0, 750.0, 1100.0, 1550.0, 2100.0, 2800.0, 3600.0, 4600.0, 5800.0, 7000.0]
    usable_centers = [c for c in centers if c < sr * 0.48]
    chunks: list[np.ndarray] = []
    total_target = max(1, round(duration_s * sr))
    remaining = total_target
    while remaining > 0:
        seg_n = min(remaining, max(16, round(0.24 * sr)))
        freqs = np.fft.rfftfreq(seg_n, 1.0 / sr)
        mag = np.zeros_like(freqs) + 0.02
        k_pick = min(3, len(usable_centers))
        for idx in rng.choice(len(usable_centers), size=k_pick, replace=False):
            mag += np.exp(-0.5 * ((freqs - usable_centers[idx]) / 120.0) ** 2)
        spec = mag * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, len(freqs)))
        sig = np.fft.irfft(spec, n=seg_n)
        peak = max(float(np.max(np.abs(sig))), 1e-9)
        sig = (sig / peak) * amp * float(rng.uniform(0.75, 1.0))
        env = np.maximum(0.0, np.sin(np.pi * np.arange(seg_n) / max(1, seg_n - 1))) ** 0.35
        chunks.append((sig * env).astype(np.float32))
        remaining -= seg_n
    return np.concatenate(chunks)[:total_target]


def generate_demo_dataset(out_dir: str | Path) -> list[dict]:
    """Create 5 synthetic WAV scenarios + manifest.jsonl inside out_dir."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    sr = 16000

    # 1. clean_speech.wav (12s, 16kHz mono, genuinely non-repeating syllables)
    clean = _nonrepeating_speech(12.0, sr, seed=771, amp=0.28)
    clean[: round(0.4 * sr)] = 0.0
    clean[-round(0.4 * sr) :] = 0.0
    _write_pcm16_wav(root / "clean_speech.wav", clean, sr)

    # 2. clipped_and_quiet.wav (8s, 16kHz mono: full-scale clipping + quiet tail)
    clipped = np.zeros(8 * sr, dtype=np.float32)
    clipped[round(0.2 * sr) : round(1.7 * sr)] = np.clip(
        5.5 * _nonrepeating_speech(1.5, sr, seed=202, amp=0.35), -1.0, 1.0
    )
    clipped[2 * sr : 8 * sr] = _nonrepeating_speech(6.0, sr, seed=203, amp=0.008)
    _write_pcm16_wav(root / "clipped_and_quiet.wav", clipped, sr)

    # 3. short_3s_utterance.wav (3.2s, 16kHz mono with 2 short pauses)
    short = _nonrepeating_speech(3.2, sr, seed=303, amp=0.28)
    short[round(0.9 * sr) : round(1.15 * sr)] = 1e-4
    short[round(2.0 * sr) : round(2.25 * sr)] = 1e-4
    _write_pcm16_wav(root / "short_3s_utterance.wav", short, sr)

    # 4. repeated_segment.wav (10s, 16kHz mono with repeated modulated phrase at [2.4, 4.2] & [5.4, 7.2])
    rep = np.zeros(10 * sr, dtype=np.float32)
    intro = _nonrepeating_speech(1.8, sr, seed=401, amp=0.27)
    phrase = _nonrepeating_speech(1.8, sr, seed=402, amp=0.32)
    outro = _nonrepeating_speech(2.0, sr, seed=403, amp=0.27)
    rep[round(0.2 * sr) : round(2.0 * sr)] = intro
    rep[round(2.4 * sr) : round(4.2 * sr)] = phrase
    rep[round(5.4 * sr) : round(7.2 * sr)] = phrase
    rep[round(7.6 * sr) : round(9.6 * sr)] = outro
    _write_pcm16_wav(root / "repeated_segment.wav", rep, sr)

    # 5. non_16k_stereo_edge.wav (4s, 8kHz stereo, non-repeating channels)
    sr8 = 8000
    ch_l = _nonrepeating_speech(4.0, sr8, seed=501, amp=0.24)
    ch_r = _nonrepeating_speech(4.0, sr8, seed=502, amp=0.24)
    stereo = np.stack([ch_l, ch_r], axis=1)
    _write_pcm16_wav(root / "non_16k_stereo_edge.wav", stereo, sr8)

    manifest = [
        {
            "pair_id": "demo-01-clean",
            "audio_path": "clean_speech.wav",
            "category": "demo_reading",
            "question": "",
            "reference_text": "The quick brown fox jumps calmly over the sleeping river dog today.",
        },
        {
            "pair_id": "demo-02-clipped-quiet",
            "audio_path": "clipped_and_quiet.wav",
            "category": "demo_reading",
            "question": "",
            "reference_text": "Signal check with clipping at start followed by low digital level.",
        },
        {
            "pair_id": "demo-03-short-3s",
            "audio_path": "short_3s_utterance.wav",
            "category": "demo_spontaneous",
            "question": "Please talk about your favorite season",
            "reference_text": "Please talk about your favorite season, I really enjoy autumn.",
        },
        {
            "pair_id": "demo-04-repeated",
            "audio_path": "repeated_segment.wav",
            "category": "demo_reading",
            "question": "",
            "reference_text": "Morning light crosses the quiet valley twice in this sentence.",
        },
        {
            "pair_id": "demo-05-stereo-8k",
            "audio_path": "non_16k_stereo_edge.wav",
            "category": "demo_reading",
            "question": "",
            "reference_text": "Narrowband stereo sample for specification checks.",
        },
    ]

    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate 5 synthetic WAV fixtures + manifest.jsonl.")
    parser.add_argument("--out-dir", required=True, help="Target output directory.")
    args = parser.parse_args(argv)
    rows = generate_demo_dataset(args.out_dir)
    print(f"Generated {len(rows)} synthetic fixtures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
