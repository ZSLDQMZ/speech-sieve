"""Configurable device-agnostic adapters for optional VAD, forced alignment, and overlap models."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .audio import runs, span, windows


class MissingBackendError(RuntimeError):
    """Raised when an enabled optional backend package is not installed."""


def resolve_device(device_cfg: str) -> str:
    """Resolve device preference ('auto' | 'cpu' | 'cuda' | 'mps') safely."""
    if device_cfg != "auto":
        return device_cfg
    if importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch  # type: ignore

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def run_batch_with_oom_halving(
    items: list[Any],
    batch_fn: Callable[[list[Any]], list[Any]],
    is_oom_fn: Callable[[Exception], bool] | None = None,
) -> list[tuple[Any, Any, str | None]]:
    """Execute batch_fn(pending); halve batches recursively on OOM down to singleton isolation."""
    if not items:
        return []
    todo: list[list[Any]] = [list(items)]
    results: list[tuple[Any, Any, str | None]] = []
    while todo:
        pending = todo.pop(0)
        try:
            outputs = batch_fn(pending)
            if len(outputs) != len(pending):
                raise ValueError("batch_output_count_mismatch")
            for item, out in zip(pending, outputs):
                results.append((item, out, None))
        except Exception as err:  # noqa: BLE001
            oom = is_oom_fn(err) if is_oom_fn is not None else ("out of memory" in str(err).lower())
            if oom and len(pending) > 1:
                mid = len(pending) // 2
                todo[0:0] = [pending[:mid], pending[mid:]]
                continue
            if len(pending) == 1:
                results.append((pending[0], None, f"{type(err).__name__}: {err}"))
            else:
                # Non-OOM batch error also splits down to isolate bad item without losing peers.
                mid = len(pending) // 2
                todo[0:0] = [pending[:mid], pending[mid:]]
    return results


@dataclass
class ReceptiveFieldSpec:
    start: float
    step: float
    duration: float


class SileroVadAdapter:
    """Configurable Silero VAD adapter supporting CPU/CUDA/MPS and injected mock callables."""

    def __init__(self, vad_cfg: dict[str, Any], device: str = "auto", model: Any = None) -> None:
        self.cfg = vad_cfg
        self.device = resolve_device(device)
        self.model = model

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        if importlib.util.find_spec("torch") is None:
            raise MissingBackendError("backend_missing:torch")
        import torch  # type: ignore

        weight_path = self.cfg.get("weight_path")
        if weight_path:
            self.model = torch.jit.load(str(weight_path), map_location=self.device).eval()
            return self.model
        if importlib.util.find_spec("silero_vad") is None:
            raise MissingBackendError("backend_missing:silero-vad")
        from silero_vad import load_silero_vad  # type: ignore

        self.model = load_silero_vad().to(self.device).eval()
        return self.model

    def infer_batch(self, arrays: list[np.ndarray], sample_rate: int) -> list[np.ndarray]:
        model = self.load()
        step_samples = int(self.cfg.get("batch_frame_samples", 512))
        if callable(model) and not hasattr(model, "reset_states"):
            return [np.asarray(p, dtype=np.float32) for p in model(arrays, sample_rate)]

        import torch  # type: ignore

        lengths = [len(x) for x in arrays]
        steps = max(1, math.ceil(max(lengths) / step_samples))
        batch = torch.zeros((len(arrays), steps * step_samples), dtype=torch.float32, device=self.device)
        for i, arr in enumerate(arrays):
            batch[i, : len(arr)] = torch.as_tensor(arr, dtype=torch.float32, device=self.device)
        if hasattr(model, "reset_states"):
            model.reset_states()
        probs = []
        with torch.inference_mode():
            for pos in range(0, batch.shape[1], step_samples):
                out = model(batch[:, pos : pos + step_samples], sample_rate)
                probs.append(out.detach().cpu().numpy().ravel())
        stacked = np.stack(probs, axis=1)
        return [stacked[i, : math.ceil(lengths[i] / step_samples)].astype(np.float32) for i in range(len(arrays))]

    def analyse_probabilities(self, probs: np.ndarray, n: int, sr: int) -> dict[str, Any]:
        step_samples = int(self.cfg.get("batch_frame_samples", 512))
        on_th = float(self.cfg.get("threshold_on", 0.5))
        off_th = float(self.cfg.get("threshold_off", 0.35))
        min_sil_samples = max(step_samples, round(float(self.cfg.get("min_silence_ms", 100.0)) * sr / 1000.0))
        min_sp_samples = max(step_samples, round(float(self.cfg.get("min_speech_ms", 32.0)) * sr / 1000.0))

        expected = math.ceil(n / step_samples)
        if len(probs) != expected:
            raise ValueError("vad_frame_count_mismatch")

        speech: list[tuple[int, int]] = []
        start: int | None = None
        silent: int | None = None
        for i, prob in enumerate(probs):
            pos = i * step_samples
            if prob >= on_th:
                if start is None:
                    start = pos
                silent = None
            elif start is not None and prob < off_th:
                if silent is None:
                    silent = pos
                if pos + step_samples - silent >= min_sil_samples:
                    if silent - start >= min_sp_samples:
                        speech.append((start, silent))
                    start = None
                    silent = None
        if start is not None and n - start >= min_sp_samples:
            speech.append((start, n))

        pauses = [(speech[i - 1][1], speech[i][0]) for i in range(1, len(speech))]
        speech_samples = sum(b - a for a, b in speech)
        return {
            "run_status": "success",
            "frames": n,
            "sample_rate": sr,
            "duration_s": float(n / sr),
            "intervals": [span(a, b, n, sr) for a, b in speech],
            "probability_step_samples": step_samples,
            "probability_frame_count": len(probs),
            "speech_fraction": float(speech_samples / n) if n else 0.0,
            "leading_non_speech_s": float(speech[0][0] / sr) if speech else None,
            "trailing_non_speech_s": float((n - speech[-1][1]) / sr) if speech else None,
            "internal_pause_count": len(pauses),
            "internal_pause_s": float(sum(b - a for a, b in pauses) / sr),
            "longest_internal_pause_s": float(max((b - a for a, b in pauses), default=0) / sr),
            "flags": [] if speech else ["no_speech_detected"],
        }


class ForcedAlignerAdapter:
    """Configurable forced aligner adapter with duration-tiered batching."""

    def __init__(self, align_cfg: dict[str, Any], device: str = "auto", model: Any = None) -> None:
        self.cfg = align_cfg
        self.device = resolve_device(device)
        self.model = model
        self.last_align_kwargs: dict[str, Any] = {}

    def load(self) -> Any:
        if self.model is not None:
            return self.model
        weight_path = self.cfg.get("weight_path")
        if not weight_path:
            raise MissingBackendError("backend_missing:aligner_weight_path")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("qwen_asr") is None:
            raise MissingBackendError("backend_missing:qwen_asr")
        import torch  # type: ignore
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner  # type: ignore

        dtype_cfg = self.cfg.get("dtype", "auto")
        if dtype_cfg == "auto":
            torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, str(dtype_cfg))
        self.model = Qwen3ForcedAligner.from_pretrained(
            str(weight_path),
            dtype=torch_dtype,
            device_map=self.device,
            attn_implementation=self.cfg.get("attn_implementation", "sdpa"),
        )
        return self.model

    def eligible(self, n: int, sr: int, has_signal: bool, reference_text: str | None) -> tuple[bool, str | None]:
        if n <= 0:
            return False, "empty_audio"
        if not has_signal:
            return False, "all_zero_audio"
        if reference_text is None:
            return False, "missing_reference_text"
        if not str(reference_text).strip():
            return False, "empty_reference_text"
        dur_s = n / sr
        if dur_s > float(self.cfg.get("max_duration_s", 300.0)):
            return False, "duration_cap_exceeded"
        if dur_s <= float(self.cfg.get("min_duration_s", 0.0125)):
            return False, "feature_input_too_short"
        return True, None

    def tier_groups(self, items: list[dict[str, Any]], sr: int) -> list[list[dict[str, Any]]]:
        tiers = sorted([(float(c), int(lim)) for c, lim in self.cfg.get("tiers", [[9.0, 8], [30.0, 4], [300.0, 1]])])
        ordered = sorted(items, key=lambda x: len(x["wave"]))
        groups_out: list[list[dict[str, Any]]] = []
        cursor = 0
        while cursor < len(ordered):
            dur = len(ordered[cursor]["wave"]) / sr
            ceiling, limit = next(((c, l) for c, l in tiers if dur <= c), tiers[-1])
            group = []
            for item in ordered[cursor : cursor + limit]:
                if len(item["wave"]) / sr > ceiling:
                    break
                group.append(item)
            if not group:
                group = [ordered[cursor]]
            groups_out.append(group)
            cursor += len(group)
        return groups_out

    def align_batch(self, items: list[dict[str, Any]], sr: int) -> list[dict[str, Any]]:
        """Align a batch cleanly; supports configurable language without mutating global classes."""
        model = self.load()
        lang = self.cfg.get("language")
        kwargs: dict[str, Any] = {
            "audio": [(it["wave"], sr) for it in items],
            "text": [it["reference_text"] for it in items],
        }
        if lang is not None:
            kwargs["language"] = lang
        self.last_align_kwargs = {"language": lang, "batch_size": len(items), "device": self.device}
        outputs = model.align(**kwargs)
        results = []
        for it, out in zip(items, outputs):
            if isinstance(out, dict) and "items" in out:
                results.append(
                    {
                        "run_status": "success",
                        "evidence_status": "indeterminate" if out.get("anomalies") else "valid",
                        "items": out["items"],
                        "anomalies": out.get("anomalies", []),
                        "timestamp_audit": out.get("timestamp_audit", {}),
                        "flags": out.get("flags", []),
                        "actual_batch_size": len(items),
                    }
                )
            else:
                parsed_items = []
                anomalies = []
                n = len(it["wave"])
                for idx, unit in enumerate(out):
                    a, b = float(unit.start_time), float(unit.end_time)
                    flags = []
                    if a < 0 or b < a or round(b * sr) > n:
                        flags.append("invalid_or_out_of_bounds")
                    if a == b:
                        flags.append("zero_duration")
                    if flags:
                        anomalies.append({"unit_index": idx, "flags": flags})
                    parsed_items.append(
                        {
                            "unit_index": idx,
                            "text": str(unit.text),
                            "original_character_indices": getattr(unit, "original_character_indices", []),
                            "start_s": a,
                            "end_s": b,
                            "flags": flags,
                        }
                    )
                results.append(
                    {
                        "run_status": "success",
                        "evidence_status": "indeterminate" if anomalies else "valid",
                        "items": parsed_items,
                        "anomalies": anomalies,
                        "timestamp_audit": {},
                        "flags": [],
                        "actual_batch_size": len(items),
                    }
                )
        return results


class BatchedOverlapAdapter:
    """Cross-item batched window inference for frame-level speech overlap estimation."""

    def __init__(
        self,
        overlap_cfg: dict[str, Any],
        device: str = "auto",
        forward_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
        receptive_field: ReceptiveFieldSpec | None = None,
    ) -> None:
        self.cfg = overlap_cfg
        self.device = resolve_device(device)
        self.forward_fn = forward_fn
        self.receptive_field = receptive_field or ReceptiveFieldSpec(start=0.0, step=0.017, duration=0.032)
        self._model = None
        self._overlap_states = None

    def _ensure_loaded(self) -> None:
        if self.forward_fn is not None or self._model is not None:
            return
        weight_path = self.cfg.get("weight_path")
        if not weight_path:
            raise MissingBackendError("backend_missing:overlap_weight_path")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("pyannote.audio") is None:
            raise MissingBackendError("backend_missing:pyannote.audio")
        import torch  # type: ignore
        from pyannote.audio import Model  # type: ignore
        from pyannote.audio.utils.powerset import Powerset  # type: ignore

        model = Model.from_pretrained(str(weight_path), map_location="cpu").to(self.device).eval()
        powerset = Powerset(len(model.specifications.classes), model.specifications.powerset_max_classes)
        self._model = model
        self._overlap_states = (powerset.mapping.sum(-1) >= 2).to(self.device)
        self.receptive_field = ReceptiveFieldSpec(
            start=float(model.receptive_field.start),
            step=float(model.receptive_field.step),
            duration=float(model.receptive_field.duration),
        )

    def _infer_windows(self, batch_array: np.ndarray, sr: int) -> np.ndarray:
        self._ensure_loaded()
        if self.forward_fn is not None:
            probs = np.asarray(self.forward_fn(batch_array, sr), dtype=np.float32)
        else:
            import torch  # type: ignore

            with torch.inference_mode():
                tensor_in = torch.as_tensor(batch_array[:, None], dtype=torch.float32, device=self.device)
                raw = self._model(tensor_in)
                probs = raw.float().exp()[..., self._overlap_states].sum(-1).cpu().numpy()
        if not np.isfinite(probs).all():
            raise ValueError("nonfinite_overlap_output")
        return probs

    def run_batch(self, items: list[tuple[str, np.ndarray, int]]) -> dict[str, dict[str, Any]]:
        """Process [(record_id, waveform, sr), ...] via chunked cross-item window batching."""
        if not items:
            return {}
        win_s = float(self.cfg.get("window_s", 10.0))
        hop_s = float(self.cfg.get("hop_s", 5.0))
        grid_step_ms = float(self.cfg.get("grid_step_ms", 10.0))
        prob_th = float(self.cfg.get("probability_threshold", 0.5))
        batch_size = int(self.cfg.get("batch_size", 16))
        row_chunk = int(self.cfg.get("row_chunk_size", 64))

        out: dict[str, dict[str, Any]] = {}
        for c_lo in range(0, len(items), row_chunk):
            chunk = items[c_lo : c_lo + row_chunk]
            work = []
            window_tasks: list[tuple[int, int, int, np.ndarray]] = []
            for idx, (rec_id, wave, sr) in enumerate(chunk):
                n = len(wave)
                win_samples = max(1, round(win_s * sr))
                hop_samples = max(1, round(hop_s * sr))
                grid_step = max(1, round(grid_step_ms * sr / 1000.0))
                starts = windows(n, win_samples, hop_samples)
                count = math.ceil(n / grid_step)
                centers = (np.arange(count) * grid_step + np.minimum(n, np.arange(1, count + 1) * grid_step) - 1) / 2.0
                work.append(
                    {
                        "id": rec_id,
                        "n": n,
                        "sr": sr,
                        "win_samples": win_samples,
                        "grid_step": grid_step,
                        "count": count,
                        "centers": centers,
                        "total": np.zeros(count, dtype=np.float64),
                        "weights": np.zeros(count, dtype=np.int32),
                        "coverage": [],
                    }
                )
                for st in starts:
                    sl = np.pad(wave[st : min(n, st + win_samples)], (0, max(0, st + win_samples - n)))
                    window_tasks.append((idx, sr, st, sl))

            # Run batched inference grouped by equal window length
            for lo in range(0, len(window_tasks), batch_size):
                sub = window_tasks[lo : lo + batch_size]
                max_w = max(len(t[3]) for t in sub)
                batch_arr = np.stack([np.pad(t[3], (0, max_w - len(t[3]))) for t in sub], axis=0)
                probs_batch = self._infer_windows(batch_arr, sub[0][1])

                field = self.receptive_field
                for (item_idx, sr, start, _), p in zip(sub, probs_batch):
                    witem = work[item_idx]
                    n = witem["n"]
                    win_samples = witem["win_samples"]
                    centers = witem["centers"]
                    times = np.array([(start + (field.start + field.step * i + field.duration / 2.0) * sr) for i in range(len(p))])
                    first = int(np.searchsorted(centers, start))
                    last = int(np.searchsorted(centers, min(n, start + win_samples)))
                    grid = centers[first:last]
                    witem["total"][first:last] += np.interp(grid, times, p, left=float(p[0]), right=float(p[-1]))
                    witem["weights"][first:last] += 1
                    witem["coverage"].append(
                        {
                            **span(start, min(n, start + win_samples), n, sr),
                            "model_window_samples": win_samples,
                            "padding_samples": max(0, start + win_samples - n),
                            "output_frame_count": len(p),
                        }
                    )

            for witem in work:
                weights = witem["weights"]
                n = witem["n"]
                sr = witem["sr"]
                win_samples = witem["win_samples"]
                grid_step = witem["grid_step"]
                fully_covered = bool(np.all(weights > 0))
                safe_weights = np.maximum(weights, 1)
                probability = witem["total"] / safe_weights
                intervals = [span(int(a * grid_step), min(n, int(b * grid_step)), n, sr) for a, b in runs(probability >= prob_th)]
                short_padded = n < win_samples
                flags = ["short_audio_padded"] if short_padded else []
                # Short audio padded to window_samples still covers all [0, n) original frames.
                evidence_status = "evaluated" if fully_covered else "indeterminate"
                out[witem["id"]] = {
                    "run_status": "success",
                    "evidence_status": evidence_status,
                    "frames": n,
                    "sample_rate": sr,
                    "duration_s": float(n / sr),
                    "intervals": intervals,
                    "total_overlap_s": float(sum(x["end_sample"] - x["start_sample"] for x in intervals) / sr),
                    "window_coverage": witem["coverage"],
                    "threshold": prob_th,
                    "flags": flags,
                    "max_overlap_probability": float(probability.max(initial=0.0)),
                }
        return out
