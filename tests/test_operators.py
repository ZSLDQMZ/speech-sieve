"""Unit tests for acoustic feature, repetition, and text operators."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from speech_sieve.alignment import count_text_tokens  # noqa: E402
from speech_sieve.config import DEFAULT_CONFIG  # noqa: E402
from speech_sieve.features import analyse_basic, analyse_rhythm, analyse_spectral, analyse_speech_level  # noqa: E402
from speech_sieve.repeat import analyse_repeat  # noqa: E402
from speech_sieve.topic_prefix import analyse_topic_prefix  # noqa: E402


class TestOperators(unittest.TestCase):
    def test_spectral_nfft2048_resolution_and_8k_nyquist_clamp(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        sr = 16000
        t = np.arange(2 * sr, dtype=np.float64) / sr
        tone_504 = (0.3 * np.sin(2.0 * np.pi * 504.0 * t)).astype(np.float32)
        intervals = [{"start_sample": 0, "end_sample": len(tone_504), "start_s": 0.0, "end_s": 2.0}]

        spec_2048 = analyse_spectral(tone_504, sr, intervals, cfg)
        self.assertEqual(spec_2048["nfft"], 2048)
        self.assertAlmostEqual(spec_2048["freq_step_hz"], 16000 / 2048)
        self.assertGreater(spec_2048["voice"]["rolloff95_hz"]["median"], 500.0)

        # Narrowband 8kHz: Nyquist=4000Hz must clamp bands cleanly without NaN crash
        sr8 = 8000
        t8 = np.arange(2 * sr8, dtype=np.float64) / sr8
        sig8 = (0.25 * np.sin(2.0 * np.pi * 600.0 * t8)).astype(np.float32)
        iv8 = [{"start_sample": 0, "end_sample": len(sig8), "start_s": 0.0, "end_s": 2.0}]
        spec_8k = analyse_spectral(sig8, sr8, iv8, cfg)
        self.assertEqual(spec_8k["run_status"], "described")
        self.assertLessEqual(spec_8k["clamped_bands_hz"]["noise_high_hz"], sr8 * 0.49)
        self.assertTrue(spec_8k["clamped_bands_hz"]["above_hz_evaluated"])
        self.assertIsNotNone(spec_8k["voice"]["flatness_band"])

        # High band above Nyquist disables above_hz cleanly instead of returning bogus zeros
        cfg_high = copy.deepcopy(cfg)
        cfg_high["features"]["spectral_bands"]["above_hz"] = 5000.0
        spec_high = analyse_spectral(sig8, sr8, iv8, cfg_high)
        self.assertFalse(spec_high["clamped_bands_hz"]["above_hz_evaluated"])
        self.assertIsNone(spec_high["voice"]["above_band_fraction"])

    def test_dynamic_frame_samples_across_non16k_rates(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        sr = 8000
        n = 10 * sr  # 10s @ 8kHz -> 20ms frame = 160 samples
        t = np.arange(n, dtype=np.float64) / sr
        wave = (0.25 * np.sin(2.0 * np.pi * 300.0 * t)).astype(np.float32)
        basic, arrays = analyse_basic(wave, sr, cfg)
        self.assertEqual(arrays["frame_samples"], 160)

        intervals = [{"start_sample": 0, "end_sample": n, "start_s": 0.0, "end_s": 10.0}]
        lvl = analyse_speech_level(
            arrays["energy_frames"],
            n,
            sr,
            arrays["frame_samples"],
            intervals,
            basic["rms"],
            basic["peak"],
            basic["near_full_scale_fraction"],
            cfg,
        )
        self.assertTrue(lvl["macro_dynamic_eligible"])
        self.assertAlmostEqual(lvl["vad_voice_s"], 10.0, places=5)

    def test_repeat_respects_non_10ms_hop_sec(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        # Non-default 20ms hop (hop_sec = 0.02) verifies dynamic hop-size scaling
        cfg["repeat"]["hop_ms"] = 20.0
        cfg["repeat"]["frame_ms"] = 40.0
        sr = 16000

        def chirp(dur: float, f0: float, f1: float) -> np.ndarray:
            k = round(dur * sr)
            freq = np.linspace(f0, f1, k)
            ph = 2.0 * np.pi * np.cumsum(freq) / sr
            s = 0.6 * np.sin(ph) + 0.3 * np.sin(2.0 * ph) + 0.1 * np.sin(3.0 * ph)
            return (0.32 * s * (np.sin(np.pi * np.arange(k) / (k - 1)) ** 0.3)).astype(np.float32)

        audio = np.zeros(10 * sr, dtype=np.float32)
        phrase = chirp(1.8, 140.0, 340.0)
        audio[round(0.3 * sr) : round(2.1 * sr)] = chirp(1.8, 280.0, 160.0)
        audio[round(2.5 * sr) : round(4.3 * sr)] = phrase
        audio[round(5.6 * sr) : round(7.4 * sr)] = phrase
        audio[round(7.8 * sr) : round(9.6 * sr)] = chirp(1.8, 360.0, 200.0)

        res = analyse_repeat(audio, sr, cfg)
        self.assertEqual(res["status"], "evaluated")
        self.assertAlmostEqual(res["hop_sec"], 0.02)
        self.assertTrue(res["candidates"], "Expected repeated modulated segment candidate under hop_ms=20")
        best = res["candidates"][0]
        # Verify matched timestamps are near [2.5, 4.3] and [5.6, 7.4] rather than doubled (~2x) by /0.01!
        self.assertLess(abs(best["first"]["start_s"] - 2.5), 1.0)
        self.assertLess(abs(best["second"]["start_s"] - 5.6), 1.0)
        self.assertLess(best["second"]["end_s"], 8.5)

    def test_rhythm_short_utterance_dip_count_vs_per10s(self) -> None:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        sr = 16000
        n = round(3.2 * sr)
        t = np.arange(n, dtype=np.float64) / sr
        wave = (0.3 * np.sin(2.0 * np.pi * 210.0 * t)).astype(np.float32)
        wave[round(0.9 * sr) : round(1.15 * sr)] = 0.0
        wave[round(2.0 * sr) : round(2.25 * sr)] = 0.0
        _, arrays = analyse_basic(wave, sr, cfg)
        iv = [{"start_sample": 0, "end_sample": n, "start_s": 0.0, "end_s": 3.2}]
        rhy, err = analyse_rhythm(arrays["energy_frames"], n, sr, arrays["frame_samples"], iv, "hello world", cfg)
        self.assertIsNone(err)
        s18 = rhy["settings"]["18"]
        self.assertEqual(s18["short_dip_count"], 2)
        # 2 dips over 3.2s yields >6 per 10s mathematically, so raw count + extent are exposed explicitly
        self.assertGreater(s18["short_dips_per10s"], 4.0)
        self.assertLess(rhy["extent_duration_s"], 5.0)

    def test_multilingual_token_proxy_and_topic_prefix_plugin(self) -> None:
        self.assertEqual(count_text_tokens("Hello brave new world", unit="auto"), 4)
        self.assertEqual(count_text_tokens("今天天气不错", unit="auto"), 6)
        self.assertEqual(count_text_tokens("Hello 世界 test", unit="auto"), 4)

        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["topic_prefix"].update(
            {
                "enabled": True,
                "explicit_cues": ["please talk about", "free topic"],
                "fillers": "umuh",
            }
        )
        hit = analyse_topic_prefix("Please talk about your favorite season, I like autumn.", "What season?", cfg)
        self.assertTrue(hit["detected"])
        self.assertEqual(hit["match_kind"], "explicit_cue")

        clean = analyse_topic_prefix("Autumn brings crisp mornings and colorful leaves.", "What season?", cfg)
        self.assertFalse(clean["detected"])
        self.assertEqual(clean["status"], "clean_direct_response")


if __name__ == "__main__":
    unittest.main()
