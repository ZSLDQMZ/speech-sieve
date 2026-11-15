"""Language-agnostic text token proxy, alignment coverage, and repeat cross-explanation."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")


def count_text_tokens(text: str | None, unit: str = "auto") -> int:
    """Estimate linguistic token count across CJK characters, whitespace words, or mixed text."""
    if not text or not isinstance(text, str):
        return 0
    if unit == "char":
        return sum(1 for ch in text if ch.isalnum())
    if unit == "whitespace":
        return len([tok for tok in text.split() if any(ch.isalnum() for ch in tok)])

    # Auto: count each CJK/Kana/Hangul symbol as 1 token and contiguous non-CJK alnum runs as words.
    cjk_count = len(_CJK_RE.findall(text))
    residual = _CJK_RE.sub(" ", text)
    word_count = sum(1 for tok in residual.split() if any(ch.isalnum() for ch in tok))
    return cjk_count + word_count


def text_norm(text: str) -> str:
    """Normalize text for literal similarity comparison across writing systems."""
    chars = []
    for ch in text:
        if unicodedata.category(ch)[0] in ("P", "Z", "C"):
            continue
        chars.append(ch.casefold())
    return "".join(chars)


def merge_spans(spans: list[tuple[float, float]] | list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(float(b), out[-1][1])
        else:
            out.append([float(a), float(b)])
    return out


def span_length(spans: list[list[float]]) -> float:
    return float(sum(b - a for a, b in spans))


def intersect_spans(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append([lo, hi])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract_spans(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for lo, hi in a:
        cursor = lo
        for start, end in b:
            if end <= cursor:
                continue
            if start >= hi:
                break
            if start > cursor:
                out.append([cursor, min(start, hi)])
            cursor = max(cursor, end)
            if cursor >= hi:
                break
        if cursor < hi:
            out.append([cursor, hi])
    return out


def pad_spans(spans: list[list[float]], margin: float, duration: float) -> list[list[float]]:
    return merge_spans([(max(0.0, a - margin), min(duration, b + margin)) for a, b in spans])


def analyse_coverage(
    record: dict[str, Any],
    vad_record: dict[str, Any] | None,
    align_record: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Measure VAD speech left outside aligned text time spans."""
    acfg = cfg["alignment"]
    text = record.get("reference_text")
    token_count = count_text_tokens(text, unit=acfg.get("token_unit", "auto")) if isinstance(text, str) else None

    if vad_record is None or not vad_record.get("intervals"):
        reason = "vad_disabled" if vad_record is None or vad_record.get("run_status") == "not_run" else "no_vad_speech"
        return {
            "duration_s": vad_record.get("duration_s") if vad_record else None,
            "vad_speech_s": 0.0,
            "reference_token_count": token_count,
            "reference_tokens_per_vad_s": None,
            "aligner_run_status": align_record["run_status"] if align_record else "not_run",
            "state": reason,
            "coverage": {},
        }

    duration = float(vad_record["duration_s"])
    speech = merge_spans([(float(i["start_s"]), float(i["end_s"])) for i in vad_record["intervals"]])
    voice = span_length(speech)
    out: dict[str, Any] = {
        "duration_s": duration,
        "vad_speech_s": voice,
        "reference_token_count": token_count,
        "reference_tokens_per_vad_s": (float(token_count / voice) if voice > 0 and token_count is not None else None),
        "aligner_run_status": align_record["run_status"] if align_record else "not_run",
        "aligner_reason": align_record.get("reason") if align_record else "not_run",
        "aligner_anomaly_units": len(align_record.get("anomalies", [])) if align_record else 0,
        "aligner_total_units": len(align_record.get("items", [])) if align_record else 0,
        "state": "unassessed",
        "coverage": {},
    }

    if not isinstance(text, str) or not text.strip():
        out["state"] = "missing_reference_text"
        return out
    if align_record is None or align_record.get("run_status") != "success":
        out["state"] = "aligner_unavailable"
        return out

    items = align_record.get("items", [])
    usable = [i for i in items if not i.get("flags") and 0 <= i["start_s"] < i["end_s"] <= duration]
    declared = [
        i
        for i in items
        if 0 <= i["start_s"] <= i["end_s"] <= duration and "invalid_or_out_of_bounds" not in i.get("flags", [])
    ]
    out["aligner_valid_units"] = len(usable)
    out["aligner_valid_unit_fraction"] = len(usable) / len(items) if items else 0.0
    if not usable or not declared:
        out["state"] = "no_trusted_aligner_positions"
        return out

    spans = merge_spans([(float(i["start_s"]), float(i["end_s"])) for i in usable])
    lo = min(float(i["start_s"]) for i in declared)
    hi = max(float(i["end_s"]) for i in declared)
    out["reported_text_hull_s"] = [lo, hi]
    out["state"] = "coverage_measurable_not_certified"

    win_s = float(acfg.get("window_s", 5.0))
    step_s = max(1, int(round(float(acfg.get("window_step_s", 2.0)))))
    min_win_voice = float(acfg.get("min_window_voice_s", 2.0))

    for margin in [float(m) for m in acfg.get("coverage_margins_s", [0.2, 0.5])]:
        covered = intersect_spans(speech, pad_spans(spans, margin, duration))
        uncovered = subtract_spans(speech, covered)
        hull_lo = max(0.0, lo - margin)
        hull_hi = min(duration, hi + margin)
        outer_lead = intersect_spans(speech, [[0.0, hull_lo]])
        outer_tail = intersect_spans(speech, [[hull_hi, duration]])
        inside_voice = intersect_spans(speech, [[hull_lo, hull_hi]])
        inside_uncovered = intersect_spans(uncovered, [[hull_lo, hull_hi]])

        positions = {hull_lo, max(0.0, hi - win_s)}
        positions.update(float(x) for x in range(math.floor(hull_lo), math.ceil(hull_hi), step_s))
        windows_list = []
        for start in sorted(positions):
            end = min(duration, hull_hi, start + win_s)
            if end <= start:
                continue
            voiced = span_length(intersect_spans(inside_voice, [[start, end]]))
            missing = span_length(intersect_spans(inside_uncovered, [[start, end]]))
            if voiced >= min_win_voice:
                flagged = sum(
                    bool(i.get("flags")) and start <= (i["start_s"] + i["end_s"]) / 2.0 <= end for i in items
                )
                windows_list.append(
                    {
                        "start_s": start,
                        "end_s": end,
                        "vad_speech_s": voiced,
                        "uncovered_voice_s": missing,
                        "uncovered_fraction": missing / voiced,
                        "aligner_flagged_units_in_window": flagged,
                    }
                )
        best = max(windows_list, key=lambda w: (w["uncovered_voice_s"], w["uncovered_fraction"]), default=None)
        out["coverage"][str(margin)] = {
            "uncovered_voice_s": span_length(uncovered),
            "uncovered_voice_fraction": span_length(uncovered) / voice if voice else 0.0,
            "leading_unexplained_voice_s": span_length(outer_lead),
            "trailing_unexplained_voice_s": span_length(outer_tail),
            "outer_unexplained_voice_s": span_length(outer_lead) + span_length(outer_tail),
            "leading_unexplained_voice_intervals": outer_lead,
            "trailing_unexplained_voice_intervals": outer_tail,
            "interior_unexplained_voice_s": span_length(inside_uncovered),
            "best_interior_window": best,
        }
    return out


def _side_evidence(span_dict: dict[str, Any], align_record: dict[str, Any], vad_record: dict[str, Any], duration: float) -> dict[str, Any]:
    a, b = float(span_dict["start_s"]), float(span_dict["end_s"])
    span_len = max(b - a, 1e-9)
    all_spans: list[list[float]] = []
    clean_spans: list[list[float]] = []
    items: list[dict[str, Any]] = []
    clean_indices: set[int] = set()
    all_indices: set[int] = set()

    audit = align_record.get("timestamp_audit", {})
    raw = audit.get("raw_ms", [])
    fixed = audit.get("fixed_ms", [])
    for i, item in enumerate(align_record.get("items", [])):
        x, z = float(item["start_s"]), float(item["end_s"])
        if z <= x or z <= a or x >= b:
            continue
        indices = item.get("original_character_indices", [])
        repaired_here = bool(
            len(raw) >= 2 * i + 2 and len(fixed) >= 2 * i + 2 and raw[2 * i : 2 * i + 2] != fixed[2 * i : 2 * i + 2]
        )
        clean = not item.get("flags") and not repaired_here and 0 <= x < z <= duration
        overlap = [max(a, x), min(b, z)]
        all_spans.append(overlap)
        all_indices.update(indices)
        if clean:
            clean_spans.append(overlap)
            clean_indices.update(indices)
        items.append(
            {
                "unit_index": i,
                "text": item["text"],
                "original_character_indices": indices,
                "overlap_s": overlap,
                "item_interval_s": [x, z],
                "flags": item.get("flags", []),
                "timestamp_repaired_here": repaired_here,
                "local_time_clean": clean,
            }
        )

    voice = span_length(
        merge_spans(
            [
                (max(a, float(iv["start_s"])), min(b, float(iv["end_s"])))
                for iv in vad_record.get("intervals", [])
                if float(iv["end_s"]) > a and float(iv["start_s"]) < b
            ]
        )
    )
    return {
        "span": span_dict,
        "aligned_overlap_fraction": span_length(merge_spans(all_spans)) / span_len,
        "clean_aligned_fraction": span_length(merge_spans(clean_spans)) / span_len,
        "vad_voice_s": voice,
        "vad_voice_fraction": voice / span_len,
        "original_indices": sorted(all_indices),
        "clean_original_indices": sorted(clean_indices),
        "aligned_text": "".join(x["text"] for x in items),
        "items": items,
    }


def _classify_pair(
    a: dict[str, Any],
    b: dict[str, Any],
    coverage: float,
    near_zero_fraction: float,
    min_voice_s: float,
) -> str:
    if (
        a["clean_aligned_fraction"] >= coverage
        and b["clean_aligned_fraction"] >= coverage
        and a["clean_original_indices"]
        and b["clean_original_indices"]
        and not (set(a["original_indices"]) & set(b["original_indices"]))
    ):
        return "two_distinct_target_positions_explain_times"
    if min(a["aligned_overlap_fraction"], b["aligned_overlap_fraction"]) <= near_zero_fraction:
        if min(a["vad_voice_s"], b["vad_voice_s"]) >= min_voice_s:
            return "unexplained_voiced_side"
        return "unexplained_side_weak_vad_support"
    return "boundary_or_alignment_ambiguous"


def analyse_cross_explanation(
    record_id: str,
    repeat_result: dict[str, Any],
    align_record: dict[str, Any] | None,
    vad_record: dict[str, Any] | None,
    duration: float | None,
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cross-explain acoustic repeat pairs against forced-alignment token spans."""
    candidates = repeat_result.get("candidates") or []
    if not candidates or align_record is None or align_record.get("run_status") != "success":
        return []
    if vad_record is None or not duration:
        return []

    timed = [i for i in align_record.get("items", []) if float(i["end_s"]) > float(i["start_s"])]
    hull = [min(float(i["start_s"]) for i in timed), max(float(i["end_s"]) for i in timed)] if timed else None

    acfg = cfg["alignment"]
    coverages = [float(c) for c in acfg.get("cross_coverages", [0.6, 0.8, 0.9])]
    near_zero = float(acfg.get("cross_near_zero_fraction", 0.2))
    min_voice = float(acfg.get("cross_min_voice_s", 0.2))

    pair_items: list[dict[str, Any]] = []
    for num, cand in enumerate(candidates, 1):
        a = _side_evidence(cand["first"], align_record, vad_record, duration)
        b = _side_evidence(cand["second"], align_record, vad_record, duration)
        labels = {str(cov): _classify_pair(a, b, cov, near_zero, min_voice) for cov in coverages}
        outside: list[dict[str, Any]] = []
        if hull:
            for side_name, side in (("first", a), ("second", b)):
                begin = float(side["span"]["start_s"])
                end = float(side["span"]["end_s"])
                if end <= hull[0] or begin >= hull[1]:
                    outside.append({"side": side_name, "gap_from_hull_s": max(hull[0] - end, begin - hull[1], 0.0)})
        pair_items.append(
            {
                "record_id": record_id,
                "candidate_number": num,
                "first": a,
                "second": b,
                "classification_by_coverage": labels,
                "text_relation": (
                    "same_literal_text"
                    if text_norm(a["aligned_text"]) and text_norm(a["aligned_text"]) == text_norm(b["aligned_text"])
                    else "different_or_unavailable_literal_text"
                ),
                "acoustic_mean_distance": cand.get("mean_distance"),
                "aligner_global_flags": align_record.get("flags", []),
                "semantic_verdict": "not_decided",
                "text_hull_s": hull,
                "fully_outside_hull_sides": outside,
            }
        )
    return pair_items
