"""End-to-end CLI and mock model pipeline (VAD/Aligner/OOM-halving) tests."""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from examples.generate_demo import generate_demo_dataset  # noqa: E402
from speech_sieve.cli import main as cli_main  # noqa: E402
from speech_sieve.config import load_preset, read_jsonl  # noqa: E402
from speech_sieve.models import ForcedAlignerAdapter, SileroVadAdapter, run_batch_with_oom_halving  # noqa: E402
from speech_sieve.pipeline import audit_batch  # noqa: E402


class _MockAlignerModel:
    def __init__(self) -> None:
        self.seen_kwargs = None

    def align(self, **kwargs):
        self.seen_kwargs = kwargs
        out = []
        for idx, (_audio, _sr) in enumerate(kwargs["audio"]):
            text = kwargs["text"][idx]
            out.append(
                [
                    SimpleNamespace(text=text, start_time=0.2, end_time=1.8, original_character_indices=list(range(len(text)))),
                ]
            )
        return out


class TestMockPipeline(unittest.TestCase):
    def test_demo_generation_and_cli_check_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows = generate_demo_dataset(work)
            self.assertEqual(len(rows), 5)

            cfg_path = str(ROOT / "configs" / "default.yaml")
            self.assertEqual(cli_main(["check", "--config", cfg_path]), 0)
            self.assertEqual(cli_main(["check", "--preset", cfg_path]), 0)

            report_path = work / "report.jsonl"
            rc = cli_main(
                [
                    "run",
                    "--input",
                    str(work / "manifest.jsonl"),
                    "--config",
                    cfg_path,
                    "--output",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            reports = read_jsonl(report_path)
            self.assertEqual(len(reports), 5)

            by_id = {r["record_id"]: r for r in reports}
            # Disabled optional models report not_evaluated; clipping triggers alert; clean sample stays normal
            self.assertEqual(by_id["demo-01-clean"]["gates"]["overlap_risk"]["status"], "not_evaluated")
            self.assertEqual(len(by_id["demo-01-clean"]["evidence"]["repeat"]["candidates"]), 0)
            self.assertEqual(by_id["demo-01-clean"]["route"], "normal")
            self.assertTrue(by_id["demo-02-clipped-quiet"]["gates"]["audio_integrity"]["triggered"])
            self.assertEqual(by_id["demo-02-clipped-quiet"]["gates"]["audio_integrity"]["reason"], "excessive_clipping")
            self.assertEqual(by_id["demo-02-clipped-quiet"]["route"], "alert")
            self.assertTrue(by_id["demo-03-short-3s"]["gates"]["prompt_echo_prefix"]["triggered"])
            self.assertEqual(by_id["demo-03-short-3s"]["route"], "review_recommended")
            rep_ev = by_id["demo-04-repeated"]["evidence"]["repeat"]
            self.assertEqual(rep_ev["status"], "evaluated")
            self.assertGreaterEqual(len(rep_ev["candidates"]), 1)
            self.assertEqual(by_id["demo-04-repeated"]["route"], "review_recommended")

            bad_cfg = work / "bad.yaml"
            bad_cfg.write_text("mode: invalid_mode\n", encoding="utf-8")
            self.assertEqual(
                cli_main(["run", "--input", str(work / "manifest.jsonl"), "--config", str(bad_cfg), "--output", str(work / "unused.jsonl")]),
                2,
            )

    def test_mock_vad_aligner_language_and_oom_halving(self) -> None:
        cfg = copy.deepcopy(load_preset(ROOT / "configs" / "default.yaml"))
        cfg["alignment"]["enabled"] = True
        cfg["models"]["vad"]["enabled"] = True
        cfg["models"]["aligner"]["enabled"] = True

        class _MockVadCallable:
            def __call__(self, arrays, sr):
                return [np.full((len(a) + 511) // 512, 0.9, dtype=np.float32) for a in arrays]

        mock_align_model = _MockAlignerModel()
        vad_adapter = SileroVadAdapter(cfg["models"]["vad"], device="cpu", model=_MockVadCallable())
        align_adapter = ForcedAlignerAdapter(cfg["models"]["aligner"], device="cpu", model=mock_align_model)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            rows = generate_demo_dataset(work)[:2]
            reports = audit_batch(
                rows,
                cfg,
                audio_root=work,
                backends={"vad": vad_adapter, "aligner": align_adapter},
            )
            self.assertEqual(len(reports), 2)
            # Language stays unset unless explicitly configured by the user
            self.assertNotIn("language", mock_align_model.seen_kwargs)
            self.assertEqual(reports[0]["evidence"]["aligner"]["run_status"], "success")
            self.assertEqual(reports[0]["evidence"]["coverage"]["state"], "coverage_measurable_not_certified")

        # Verify OOM halving isolates bad batch element cleanly
        calls: list[int] = []

        def flaky_batch(sub):
            calls.append(len(sub))
            if len(sub) > 1:
                raise RuntimeError("CUDA out of memory")
            if sub[0]["bad"]:
                raise RuntimeError("single item failure")
            return ["ok"]

        out = run_batch_with_oom_halving(
            [{"id": 1, "bad": False}, {"id": 2, "bad": True}, {"id": 3, "bad": False}],
            flaky_batch,
        )
        self.assertGreaterEqual(len(calls), 3)
        by_key = {item["id"]: (val, err) for item, val, err in out}
        self.assertEqual(by_key[1], ("ok", None))
        self.assertIsNone(by_key[2][0])
        self.assertIn("single item failure", by_key[2][1])
        self.assertEqual(by_key[3], ("ok", None))


if __name__ == "__main__":
    unittest.main()
