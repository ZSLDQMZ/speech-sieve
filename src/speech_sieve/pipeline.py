"""End-to-end audit orchestration, honest tri-state gate policy, and routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .alignment import analyse_coverage, analyse_cross_explanation
from .audio import AudioLoadError, load_audio
from .features import analyse_basic, analyse_rhythm, analyse_spectral, analyse_speech_level
from .models import (
    BatchedOverlapAdapter,
    ForcedAlignerAdapter,
    MissingBackendError,
    SileroVadAdapter,
    run_batch_with_oom_halving,
)
from .repeat import analyse_repeat
from .topic_prefix import analyse_topic_prefix


def _gate(
    status: str,
    triggered: bool = False,
    reason: str | None = None,
    severity: str = "review_recommended",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {"status": status, "triggered": bool(triggered), "severity": severity, "reason": reason}
    if details:
        out["details"] = details
    return out


def evaluate_gates(row: dict[str, Any], evidence: dict[str, Any], cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Evaluate all configured gates with honest tri-state outcomes (evaluated | not_evaluated | skipped)."""
    gates_cfg = cfg["gates"]
    acfg = cfg["alignment"]
    is_reading = cfg.get("mode") == "scripted_reading"
    gates: dict[str, dict[str, Any]] = {}

    basic = evidence.get("basic")
    vad = evidence.get("vad")
    level = evidence.get("level")
    coverage = evidence.get("coverage")
    rhythm = evidence.get("rhythm")
    spectral = evidence.get("spectral")
    overlap = evidence.get("overlap")
    cross_pairs = evidence.get("cross_pairs") or []
    prompt_echo = evidence.get("prompt_echo")

    voice = level.get("vad_voice_s") if level else None
    duration = level.get("duration_s") if level else (vad.get("duration_s") if vad else None)

    # 1. Audio integrity (decode error, all-zero, severe clipping, or zero VAD speech)
    g = gates_cfg["audio_integrity"]
    if not g.get("enabled", True):
        gates["audio_integrity"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif evidence.get("load_error"):
        gates["audio_integrity"] = _gate("evaluated", triggered=True, reason=evidence["load_error"], severity=g["severity"])
    elif basic is None:
        gates["audio_integrity"] = _gate("not_evaluated", reason="basic_missing", severity=g["severity"])
    else:
        no_vad_speech = vad is not None and vad.get("run_status") == "success" and not vad.get("intervals")
        clip_frac = float(basic.get("near_full_scale_fraction") or 0.0)
        clip_run = int(basic.get("max_near_full_scale_samples") or 0)
        clipped = (
            clip_frac >= float(g.get("maximum_clipping_fraction", 0.001))
            and clip_run >= int(g.get("minimum_clipped_run_samples", 4))
        )
        trig = bool(basic.get("all_zero")) or clipped or no_vad_speech
        if basic.get("all_zero"):
            reason = "all_zero"
        elif clipped:
            reason = "excessive_clipping"
        elif no_vad_speech:
            reason = "no_vad_speech"
        else:
            reason = None
        details = {"near_full_scale_fraction": clip_frac, "max_near_full_scale_samples": clip_run} if clipped else None
        gates["audio_integrity"] = _gate("evaluated", triggered=trig, reason=reason, severity=g["severity"], details=details)

    # 2. Digital signal level low
    g = gates_cfg["signal_level_low"]
    if not g.get("enabled", True):
        gates["signal_level_low"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif level is None or level.get("speech_rms_dbfs") is None:
        reason = level.get("state") if level else "level_unavailable"
        gates["signal_level_low"] = _gate("not_evaluated", reason=reason, severity=g["severity"])
    else:
        speech_db = float(level["speech_rms_dbfs"])
        trig = speech_db <= float(g["maximum_dbfs"])
        gates["signal_level_low"] = _gate(
            "evaluated",
            triggered=trig,
            reason="digital_speech_rms_below_threshold" if trig else None,
            severity=g["severity"],
            details={"speech_rms_dbfs": speech_db, "note": level.get("metric_note")},
        )

    # 3. Macro level variance
    g = gates_cfg["signal_level_variance"]
    if not g.get("enabled", True):
        gates["signal_level_variance"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif level is None:
        gates["signal_level_variance"] = _gate("not_evaluated", reason="level_unavailable", severity=g["severity"])
    elif not level.get("macro_dynamic_eligible"):
        min_req_dur = float(level.get("min_required_windows", 3)) * float(level.get("window_seconds", 3.0))
        if level.get("state") == "not_evaluated_vad_disabled":
            reason = "vad_disabled"
        elif duration is not None and duration < min_req_dur:
            reason = "insufficient_recording_duration"
        else:
            reason = "insufficient_eligible_voice_windows"
        gates["signal_level_variance"] = _gate("not_evaluated", reason=reason, severity=g["severity"])
    else:
        delta = float(level["window_p90_minus_p10_db"])
        trig = delta >= float(g["minimum_db"])
        gates["signal_level_variance"] = _gate(
            "evaluated",
            triggered=trig,
            reason="window_level_variance_exceeded" if trig else None,
            severity=g["severity"],
            details={"window_p90_minus_p10_db": delta},
        )

    # 4. Alignment coverage gap
    g = gates_cfg["alignment_coverage_gap"]
    if not g.get("enabled", True):
        gates["alignment_coverage_gap"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif not is_reading:
        gates["alignment_coverage_gap"] = _gate("skipped", reason="out_of_scope_spontaneous", severity=g["severity"])
    elif coverage is None or coverage.get("state") != "coverage_measurable_not_certified":
        gates["alignment_coverage_gap"] = _gate(
            "not_evaluated", reason=coverage.get("state") if coverage else "coverage_unavailable", severity=g["severity"]
        )
    else:
        margin_key = str(float(g.get("margin_s", 0.5)))
        cov_entry = coverage.get("coverage", {}).get(margin_key)
        if cov_entry is None:
            gates["alignment_coverage_gap"] = _gate("not_evaluated", reason="margin_unavailable", severity=g["severity"])
        else:
            outside = float(cov_entry["outer_unexplained_voice_s"])
            frac = outside / float(voice) if voice else 0.0
            trig = outside >= float(g["minimum_unexplained_voice_s"]) and frac >= float(g["minimum_unexplained_fraction"])
            gates["alignment_coverage_gap"] = _gate(
                "evaluated",
                triggered=trig,
                reason="outer_unexplained_voice_exceeded" if trig else None,
                severity=g["severity"],
                details={"outer_unexplained_voice_s": outside, "fraction": frac},
            )

    # 5. Acoustic repetition + alignment cross-explanation
    g = gates_cfg["acoustic_repetition"]
    repeat_ev = evidence.get("repeat")
    if not g.get("enabled", True):
        gates["acoustic_repetition"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif not is_reading:
        gates["acoustic_repetition"] = _gate("skipped", reason="out_of_scope_spontaneous", severity=g["severity"])
    elif repeat_ev is None or repeat_ev.get("status") == "not_evaluated":
        gates["acoustic_repetition"] = _gate(
            "not_evaluated", reason=repeat_ev.get("reason") if repeat_ev else "repeat_unavailable", severity=g["severity"]
        )
    else:
        cov_key = str(float(g.get("coverage", acfg.get("cross_default_coverage", 0.8))))
        hull_margin = float(g.get("outside_hull_margin_s", acfg.get("cross_outside_hull_margin_s", 0.5)))
        confirmed = any(
            p["classification_by_coverage"].get(cov_key) == "unexplained_voiced_side"
            and any(x["gap_from_hull_s"] >= hull_margin for x in p.get("fully_outside_hull_sides", []))
            for p in cross_pairs
        )
        cands = repeat_ev.get("candidates") or []
        if confirmed:
            gates["acoustic_repetition"] = _gate(
                "evaluated",
                triggered=True,
                reason="unexplained_voiced_repeat_outside_hull",
                severity=g["severity"],
                details={"candidate_count": len(cands)},
            )
        elif cands and not cross_pairs:
            gates["acoustic_repetition"] = _gate(
                "not_evaluated",
                triggered=False,
                reason="cross_explanation_unavailable",
                severity="review_recommended",
                details={"acoustic_candidate_count": len(cands), "unresolved_candidates": True},
            )
        else:
            gates["acoustic_repetition"] = _gate(
                "evaluated",
                triggered=False,
                severity=g["severity"],
                details={"candidate_count": len(cands)},
            )

    # 6. Rhythm cadence profile (continuous descriptor by default; opt-in gate)
    g = gates_cfg["rhythm_cadence_profile"]
    if not g.get("enabled", False):
        gates["rhythm_cadence_profile"] = _gate("skipped", reason="profile_only_gate_disabled", severity=g["severity"])
    elif not is_reading:
        gates["rhythm_cadence_profile"] = _gate("skipped", reason="out_of_scope_spontaneous", severity=g["severity"])
    elif rhythm is None:
        gates["rhythm_cadence_profile"] = _gate(
            "not_evaluated", reason=evidence.get("rhythm_state") or "rhythm_unavailable", severity=g["severity"]
        )
    else:
        depth_key = str(int(g.get("energy_depth_db", 18)))
        setting = rhythm.get("settings", {}).get(depth_key)
        extent_s = float(rhythm.get("extent_duration_s", 0.0))
        min_extent = float(g.get("minimum_extent_s", 5.0))
        min_dips = int(g.get("minimum_dip_count", 4))
        if setting is None:
            gates["rhythm_cadence_profile"] = _gate("not_evaluated", reason="depth_setting_missing", severity=g["severity"])
        elif extent_s < min_extent or int(setting.get("short_dip_count", 0)) < min_dips:
            gates["rhythm_cadence_profile"] = _gate(
                "not_evaluated",
                reason="insufficient_extent_or_dip_count_for_rate",
                severity=g["severity"],
                details={"extent_duration_s": extent_s, "short_dip_count": setting.get("short_dip_count")},
            )
        else:
            trig = (
                float(setting["short_dips_per10s"]) >= float(g["minimum_dips_per_10s"])
                and float(setting["short_dip_fraction"]) >= float(g["minimum_dip_fraction"])
                and float(rhythm["tokens_per_extent_s"]) <= float(g["maximum_tokens_per_extent_s"])
            )
            gates["rhythm_cadence_profile"] = _gate(
                "evaluated",
                triggered=trig,
                reason="rhythm_cadence_threshold_matched" if trig else None,
                severity=g["severity"],
                details={"short_dip_count": setting["short_dip_count"], "short_dips_per10s": setting["short_dips_per10s"]},
            )

    # 7. Spectral tilt profile (continuous descriptor by default; opt-in gate)
    g = gates_cfg["spectral_tilt_profile"]
    if not g.get("enabled", False):
        gates["spectral_tilt_profile"] = _gate("skipped", reason="profile_only_gate_disabled", severity=g["severity"])
    elif (
        spectral is None
        or spectral.get("run_status") != "described"
        or level is None
        or level.get("speech_rms_dbfs") is None
        or not (spectral.get("voice") and spectral["voice"].get("rolloff95_hz"))
    ):
        gates["spectral_tilt_profile"] = _gate("not_evaluated", reason="insufficient_spectral_or_level_evidence", severity=g["severity"])
    else:
        roll_med = float(spectral["voice"]["rolloff95_hz"]["median"])
        speech_db = float(level["speech_rms_dbfs"])
        trig = roll_med <= float(g["maximum_rolloff95_median_hz"]) and speech_db <= float(g["maximum_speech_rms_dbfs"])
        gates["spectral_tilt_profile"] = _gate(
            "evaluated",
            triggered=trig,
            reason="low_rolloff_and_low_rms" if trig else None,
            severity=g["severity"],
            details={"rolloff95_median_hz": roll_med, "speech_rms_dbfs": speech_db},
        )

    # 8. Overlap risk
    g = gates_cfg["overlap_risk"]
    if not g.get("enabled", True):
        gates["overlap_risk"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif overlap is None or overlap.get("run_status") != "success":
        gates["overlap_risk"] = _gate(
            "not_evaluated", reason=overlap.get("reason") if overlap else "overlap_unavailable", severity=g["severity"]
        )
    elif overlap.get("evidence_status") != "evaluated" or overlap.get("max_overlap_probability") is None:
        gates["overlap_risk"] = _gate(
            "not_evaluated", reason=overlap.get("evidence_status") or "indeterminate_overlap", severity=g["severity"]
        )
    else:
        max_p = float(overlap["max_overlap_probability"])
        trig = max_p >= float(g["minimum_max_probability"])
        gates["overlap_risk"] = _gate(
            "evaluated",
            triggered=trig,
            reason="high_overlap_probability_risk" if trig else None,
            severity=g["severity"],
            details={"max_overlap_probability": max_p, "flags": overlap.get("flags", [])},
        )

    # 9. Edge truncation / long outer non-speech
    g = gates_cfg["edge_truncation_risk"]
    if not g.get("enabled", True):
        gates["edge_truncation_risk"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif vad is None or vad.get("run_status") != "success" or not vad.get("intervals"):
        gates["edge_truncation_risk"] = _gate("not_evaluated", reason="vad_unavailable_or_empty", severity=g["severity"])
    else:
        edge_s = max(float(vad["leading_non_speech_s"] or 0.0), float(vad["trailing_non_speech_s"] or 0.0))
        trig = edge_s >= float(g["minimum_seconds"])
        gates["edge_truncation_risk"] = _gate(
            "evaluated",
            triggered=trig,
            reason="long_edge_non_speech" if trig else None,
            severity=g["severity"],
            details={"max_edge_non_speech_s": edge_s},
        )

    # 10. Excessive silence
    g = gates_cfg["excessive_silence"]
    if not g.get("enabled", True):
        gates["excessive_silence"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif voice is None or not duration:
        gates["excessive_silence"] = _gate("not_evaluated", reason="vad_unavailable", severity=g["severity"])
    else:
        frac = float(voice / duration)
        trig = float(voice) <= float(g["maximum_vad_voice_s"]) and frac <= float(g["maximum_vad_voice_fraction"])
        gates["excessive_silence"] = _gate(
            "evaluated",
            triggered=trig,
            reason="minimal_speech_in_recording" if trig else None,
            severity=g["severity"],
            details={"vad_voice_s": voice, "vad_voice_fraction": frac},
        )

    # 11. Non-speech dominant vs text tokens
    g = gates_cfg["non_speech_dominant"]
    if not g.get("enabled", True):
        gates["non_speech_dominant"] = _gate("skipped", reason="gate_disabled", severity=g["severity"])
    elif not is_reading:
        gates["non_speech_dominant"] = _gate("skipped", reason="out_of_scope_spontaneous", severity=g["severity"])
    elif coverage is None or coverage.get("reference_token_count") in (None, 0) or voice is None:
        gates["non_speech_dominant"] = _gate("not_evaluated", reason="missing_tokens_or_vad", severity=g["severity"])
    else:
        toks = int(coverage["reference_token_count"])
        rate = toks / voice if voice > 0 else 0.0
        trig = float(voice) >= float(g["minimum_vad_voice_s"]) and rate <= float(g["maximum_tokens_per_vad_s"])
        gates["non_speech_dominant"] = _gate(
            "evaluated",
            triggered=trig,
            reason="very_low_token_rate" if trig else None,
            severity=g["severity"],
            details={"tokens_per_vad_s": rate},
        )

    # 12. Prompt echo prefix (opt-in plugin)
    g = gates_cfg["prompt_echo_prefix"]
    if not g.get("enabled", False) or (prompt_echo and prompt_echo.get("status") == "disabled"):
        gates["prompt_echo_prefix"] = _gate("skipped", reason="plugin_disabled", severity=g["severity"])
    elif prompt_echo is None or prompt_echo.get("status") == "missing_text":
        gates["prompt_echo_prefix"] = _gate("not_evaluated", reason="missing_reference_text", severity=g["severity"])
    else:
        trig = bool(prompt_echo.get("detected"))
        gates["prompt_echo_prefix"] = _gate(
            "evaluated",
            triggered=trig,
            reason=prompt_echo.get("match_kind") if trig else None,
            severity=g["severity"],
            details=prompt_echo,
        )

    # 13. Advisory CER (never auto-reject by fixed CER threshold)
    g = gates_cfg["advisory_cer"]
    if not g.get("enabled", False):
        gates["advisory_cer"] = _gate("skipped", reason="advisory_disabled_by_policy", severity="advisory")
    elif row.get("cer") is None:
        gates["advisory_cer"] = _gate("not_evaluated", reason="cer_not_supplied", severity="advisory")
    else:
        gates["advisory_cer"] = _gate(
            "evaluated",
            triggered=False,
            reason="advisory_only_no_auto_reject",
            severity="advisory",
            details={"cer": float(row["cer"])},
        )

    return gates


def route_record(gates: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Assign primary neutral route bucket: alert | review_recommended | normal."""
    alerts = [k for k, v in gates.items() if v["status"] == "evaluated" and v["triggered"] and v["severity"] == "alert"]
    if alerts:
        return "alert", alerts
    reviews = [
        k
        for k, v in gates.items()
        if (v["status"] == "evaluated" and v["triggered"] and v["severity"] == "review_recommended")
        or (v.get("details") or {}).get("unresolved_candidates")
    ]
    if reviews:
        return "review_recommended", reviews
    return "normal", []


def audit_batch(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    audio_root: str | Path | None = None,
    backends: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run decoupled CPU + optional batched model stages across a list of manifest rows."""
    backends = backends or {}
    root = Path(audio_root) if audio_root else None
    mcfg = cfg["models"]
    device = mcfg.get("device", "auto")

    decoded: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        rec_id = str(row.get("pair_id") or row.get("record_id") or f"row-{idx:05d}")
        raw_path = row.get("audio_path", "")
        resolved = (root / raw_path) if (root and raw_path and not Path(raw_path).is_absolute()) else Path(raw_path)
        entry: dict[str, Any] = {"id": rec_id, "row": row, "resolved_path": str(resolved), "wave": None, "sr": None, "meta": None, "error": None}
        try:
            wave, sr, meta = load_audio(resolved, cfg["audio"])
            entry.update(wave=wave, sr=sr, meta=meta)
        except AudioLoadError as err:
            entry["error"] = str(err)
        decoded.append(entry)

    # Batch optional VAD (length-bucketed)
    vad_results: dict[str, dict[str, Any]] = {}
    vad_cfg = mcfg["vad"]
    vad_enabled = bool(vad_cfg.get("enabled", False))
    vad_adapter = backends.get("vad") or (SileroVadAdapter(vad_cfg, device=device) if vad_enabled else None)
    if vad_enabled and vad_adapter is not None:
        valid_entries = sorted([e for e in decoded if e["wave"] is not None], key=lambda e: len(e["wave"]))
        bs = int(vad_cfg.get("batch_size", 32))
        for lo in range(0, len(valid_entries), bs):
            grp = valid_entries[lo : lo + bs]
            try:
                probs_list = vad_adapter.infer_batch([e["wave"] for e in grp], grp[0]["sr"])
                for e, probs in zip(grp, probs_list):
                    vad_results[e["id"]] = vad_adapter.analyse_probabilities(probs, len(e["wave"]), e["sr"])
            except MissingBackendError as err:
                for e in grp:
                    vad_results[e["id"]] = {"run_status": "not_run", "reason": str(err), "duration_s": e["meta"]["duration_s"], "intervals": []}
            except Exception as err:  # noqa: BLE001
                for e in grp:
                    vad_results[e["id"]] = {"run_status": "failed", "reason": f"{type(err).__name__}: {err}", "duration_s": e["meta"]["duration_s"], "intervals": []}

    # Batch optional Overlap (BatchedOverlapAdapter)
    overlap_results: dict[str, dict[str, Any]] = {}
    ov_cfg = mcfg["overlap"]
    ov_enabled = bool(ov_cfg.get("enabled", False))
    ov_adapter = backends.get("overlap") or (BatchedOverlapAdapter(ov_cfg, device=device) if ov_enabled else None)
    if ov_enabled and ov_adapter is not None:
        valid_ov = [(e["id"], e["wave"], e["sr"]) for e in decoded if e["wave"] is not None]
        try:
            overlap_results = ov_adapter.run_batch(valid_ov)
        except MissingBackendError as err:
            for rec_id, _, sr in valid_ov:
                overlap_results[rec_id] = {"run_status": "not_run", "evidence_status": "unavailable", "reason": str(err)}
        except Exception as err:  # noqa: BLE001
            for rec_id, _, sr in valid_ov:
                overlap_results[rec_id] = {"run_status": "failed", "evidence_status": "unavailable", "reason": f"{type(err).__name__}: {err}"}

    # Batch optional Forced Aligner (tiered + OOM halving)
    align_results: dict[str, dict[str, Any]] = {}
    al_cfg = mcfg["aligner"]
    al_enabled = bool(al_cfg.get("enabled", False))
    al_adapter = backends.get("aligner") or (ForcedAlignerAdapter(al_cfg, device=device) if al_enabled else None)
    if al_enabled and al_adapter is not None:
        eligible_items: list[dict[str, Any]] = []
        for e in decoded:
            if e["wave"] is None:
                continue
            ok, skip_reason = al_adapter.eligible(len(e["wave"]), e["sr"], bool(e["wave"].any()), e["row"].get("reference_text"))
            if ok:
                eligible_items.append({"id": e["id"], "wave": e["wave"], "sr": e["sr"], "reference_text": e["row"].get("reference_text")})
            else:
                align_results[e["id"]] = {"run_status": "not_run", "evidence_status": "unavailable", "reason": skip_reason, "items": [], "anomalies": []}
        if eligible_items:
            for tier_grp in al_adapter.tier_groups(eligible_items, eligible_items[0]["sr"]):
                try:
                    batch_out = run_batch_with_oom_halving(tier_grp, lambda sub: al_adapter.align_batch(sub, sub[0]["sr"]))
                    for it, res, err_msg in batch_out:
                        if err_msg is not None:
                            align_results[it["id"]] = {
                                "run_status": "failed" if "MissingBackendError" not in err_msg else "not_run",
                                "evidence_status": "unavailable",
                                "reason": err_msg,
                                "items": [],
                                "anomalies": [],
                            }
                        else:
                            align_results[it["id"]] = res
                except MissingBackendError as err:
                    for it in tier_grp:
                        align_results[it["id"]] = {"run_status": "not_run", "evidence_status": "unavailable", "reason": str(err), "items": [], "anomalies": []}

    reports: list[dict[str, Any]] = []
    for e in decoded:
        rec_id = e["id"]
        row = e["row"]
        if e["wave"] is None:
            evidence: dict[str, Any] = {"load_error": e["error"]}
            gates = evaluate_gates(row, evidence, cfg)
            route, triggered = route_record(gates)
            reports.append(
                {
                    "record_id": rec_id,
                    "audio_path": str(row.get("audio_path", "")),
                    "category": row.get("category", "uncategorized"),
                    "mode": cfg["mode"],
                    "reference_text": row.get("reference_text", ""),
                    "question": row.get("question", ""),
                    "route": route,
                    "triggered_gates": triggered,
                    "certification_note": "no_triggered_gates_not_certified_clean",
                    "gates": gates,
                    "evidence": evidence,
                }
            )
            continue

        wave = e["wave"]
        sr = e["sr"]
        n = len(wave)
        basic_rec, basic_arr = analyse_basic(wave, sr, cfg)
        energy = basic_arr["energy_frames"]
        frame_samples = basic_arr["frame_samples"]

        vad_rec = vad_results.get(rec_id)
        if vad_rec is None:
            vad_rec = {"run_status": "not_run", "reason": "backend_disabled", "duration_s": float(n / sr), "intervals": None}
        vad_intervals = vad_rec["intervals"] if vad_rec.get("run_status") == "success" else None

        level_rec = analyse_speech_level(
            energy,
            n,
            sr,
            frame_samples,
            vad_intervals,
            basic_rec["rms"],
            basic_rec["peak"],
            basic_rec["near_full_scale_fraction"],
            cfg,
        )
        rhythm_rec, rhythm_state = analyse_rhythm(energy, n, sr, frame_samples, vad_intervals, row.get("reference_text"), cfg)
        spectral_rec = analyse_spectral(wave, sr, vad_intervals, cfg)
        repeat_rec = analyse_repeat(wave, sr, cfg)

        align_rec = align_results.get(rec_id)
        if align_rec is None:
            align_rec = {"run_status": "not_run", "evidence_status": "unavailable", "reason": "backend_disabled", "items": [], "anomalies": []}

        overlap_rec = overlap_results.get(rec_id)
        if overlap_rec is None:
            overlap_rec = {"run_status": "not_run", "evidence_status": "unavailable", "reason": "backend_disabled", "max_overlap_probability": None}

        cov_rec = analyse_coverage(row, vad_rec if vad_intervals is not None else None, align_rec, cfg) if cfg["alignment"].get("enabled", True) else {
            "state": "alignment_disabled",
            "reference_token_count": cov_count_fallback(row, cfg),
        }
        cross_pairs = (
            analyse_cross_explanation(rec_id, repeat_rec, align_rec, vad_rec, float(n / sr), cfg)
            if (vad_intervals is not None and align_rec.get("run_status") == "success")
            else []
        )
        prompt_echo = analyse_topic_prefix(row.get("reference_text"), row.get("question"), cfg)

        evidence = {
            "audio_meta": e["meta"],
            "basic": basic_rec,
            "vad": vad_rec,
            "level": level_rec,
            "rhythm": rhythm_rec,
            "rhythm_state": rhythm_state,
            "spectral": spectral_rec,
            "repeat": {k: v for k, v in repeat_rec.items() if k != "comparisons"},
            "aligner": align_rec,
            "overlap": overlap_rec,
            "coverage": cov_rec,
            "cross_pairs": cross_pairs,
            "prompt_echo": prompt_echo,
        }
        gates = evaluate_gates(row, evidence, cfg)
        route, triggered = route_record(gates)
        reports.append(
            {
                "record_id": rec_id,
                "audio_path": str(row.get("audio_path", "")),
                "category": row.get("category", "uncategorized"),
                "mode": cfg["mode"],
                "duration_s": float(n / sr),
                "reference_text": row.get("reference_text", ""),
                "question": row.get("question", ""),
                "route": route,
                "triggered_gates": triggered,
                "certification_note": "no_triggered_gates_not_certified_clean",
                "gates": gates,
                "evidence": evidence,
            }
        )
    return reports


def cov_count_fallback(row: dict[str, Any], cfg: dict[str, Any]) -> int | None:
    from .alignment import count_text_tokens

    text = row.get("reference_text")
    if not isinstance(text, str):
        return None
    return count_text_tokens(text, unit=cfg.get("alignment", {}).get("token_unit", "auto"))


def audit_record(
    row: dict[str, Any],
    cfg: dict[str, Any],
    audio_root: str | Path | None = None,
    backends: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit a single manifest row."""
    return audit_batch([row], cfg, audio_root=audio_root, backends=backends)[0]
