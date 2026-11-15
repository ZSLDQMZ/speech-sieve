"""Unit tests for YAML configuration, cross-platform safe I/O, and tri-state gating."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.generate_demo import _write_pcm16_wav  # noqa: E402
from speech_sieve.audio import AudioLoadError, load_audio  # noqa: E402
from speech_sieve.config import ConfigError, load_preset, read_jsonl, write_jsonl_atomic  # noqa: E402
from speech_sieve.models import BatchedOverlapAdapter, ReceptiveFieldSpec  # noqa: E402
from speech_sieve.pipeline import evaluate_gates, route_record  # noqa: E402


class TestGatingAndConfig(unittest.TestCase):
    def test_load_default_config_and_invalid_reject(self) -> None:
        cfg = load_preset(ROOT / "configs" / "default.yaml")
        self.assertEqual(cfg["schema_version"], 1)
        self.assertEqual(cfg["preset_name"], "default")
        self.assertFalse(cfg["gates"]["rhythm_cadence_profile"]["enabled"])
        self.assertFalse(cfg["gates"]["spectral_tilt_profile"]["enabled"])
        self.assertFalse(cfg["gates"]["advisory_cer"]["enabled"])

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.yaml"
            bad.write_text("mode: unknown_mode\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_preset(bad)

    def test_overlap_short_audio_padded_remains_evaluated(self) -> None:
        # <10s audio (3.0s @ 16kHz) covers all original [0, n) frames -> must stay evaluated
        sr = 16000
        n = 3 * sr
        wave = np.zeros(n, dtype=np.float32)

        def mock_forward(batch_arr: np.ndarray, _sr: int) -> np.ndarray:
            # pyannote outputs ~589 frames per 10s window
            out = np.full((batch_arr.shape[0], 589), 0.12, dtype=np.float32)
            out[0, 50:70] = 0.85
            return out

        adapter = BatchedOverlapAdapter(
            {"window_s": 10.0, "hop_s": 5.0, "grid_step_ms": 10.0, "probability_threshold": 0.5, "batch_size": 4},
            device="cpu",
            forward_fn=mock_forward,
            receptive_field=ReceptiveFieldSpec(start=0.0, step=0.017, duration=0.032),
        )
        res = adapter.run_batch([("short-1", wave, sr)])["short-1"]
        self.assertEqual(res["evidence_status"], "evaluated")
        self.assertIn("short_audio_padded", res["flags"])
        self.assertGreaterEqual(res["max_overlap_probability"], 0.8)

        cfg = load_preset(ROOT / "configs" / "default.yaml")
        gates = evaluate_gates({}, {"overlap": res, "basic": {"all_zero": False}}, cfg)
        self.assertEqual(gates["overlap_risk"]["status"], "evaluated")
        self.assertTrue(gates["overlap_risk"]["triggered"])

    def test_rhythm_gate_precondition_blocks_short_utterance_inflation(self) -> None:
        cfg = copy.deepcopy(load_preset(ROOT / "configs" / "default.yaml"))
        cfg["gates"]["rhythm_cadence_profile"]["enabled"] = True

        evidence = {
            "basic": {"all_zero": False},
            "rhythm": {
                "extent_duration_s": 3.2,
                "tokens_per_extent_s": 1.2,
                "settings": {"18": {"short_dip_count": 2, "short_dips_per10s": 6.25, "short_dip_fraction": 0.16}},
            },
        }
        gates = evaluate_gates({}, evidence, cfg)
        self.assertEqual(gates["rhythm_cadence_profile"]["status"], "not_evaluated")
        self.assertFalse(gates["rhythm_cadence_profile"]["triggered"])
        self.assertEqual(gates["rhythm_cadence_profile"]["reason"], "insufficient_extent_or_dip_count_for_rate")

    def test_cross_platform_audio_fallback_and_windows_atomic_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            sr8 = 8000
            t = np.arange(sr8, dtype=np.float64) / sr8
            stereo = np.stack([0.2 * np.sin(2 * np.pi * 200 * t), 0.2 * np.sin(2 * np.pi * 400 * t)], axis=1)
            wav_path = tmp_dir / "stereo8k.wav"
            _write_pcm16_wav(wav_path, stereo, sr8)

            strict_cfg = copy.deepcopy(load_preset(ROOT / "configs" / "default.yaml")["audio"])
            strict_cfg.update({"require_mono": True, "allow_resample": False, "allow_downmix": False})
            with self.assertRaises(AudioLoadError):
                load_audio(wav_path, strict_cfg)

            flex_cfg = load_preset(ROOT / "configs" / "default.yaml")["audio"]
            arr, out_sr, meta = load_audio(wav_path, flex_cfg)
            self.assertEqual(out_sr, 16000)
            self.assertTrue(meta["downmixed"])
            self.assertTrue(meta["resampled"])
            self.assertEqual(arr.ndim, 1)

            jsonl_path = tmp_dir / "atomic.jsonl"
            write_jsonl_atomic(jsonl_path, [{"id": 1, "score": np.float32(0.5)}])
            loaded = read_jsonl(jsonl_path)
            self.assertEqual(len(loaded), 1)
            self.assertAlmostEqual(loaded[0]["score"], 0.5, places=4)

    def test_spontaneous_mode_skips_reading_gates(self) -> None:
        cfg = copy.deepcopy(load_preset(ROOT / "configs" / "default.yaml"))
        cfg["mode"] = "spontaneous"
        cfg["alignment"]["enabled"] = False
        cfg["repeat"]["enabled"] = False
        gates = evaluate_gates(
            {"reference_text": "What season, I enjoy autumn."},
            {
                "basic": {"all_zero": False},
                "prompt_echo": {"detected": True, "match_kind": "explicit_cue", "status": "prompt_echo_prefix"},
            },
            cfg,
        )
        self.assertEqual(gates["alignment_coverage_gap"]["status"], "skipped")
        self.assertEqual(gates["acoustic_repetition"]["status"], "skipped")
        self.assertEqual(gates["rhythm_cadence_profile"]["status"], "skipped")
        self.assertTrue(gates["prompt_echo_prefix"]["triggered"])
        route, reasons = route_record(gates)
        self.assertEqual(route, "review_recommended")
        self.assertIn("prompt_echo_prefix", reasons)


if __name__ == "__main__":
    unittest.main()
