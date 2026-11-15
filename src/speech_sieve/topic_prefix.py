"""Opt-in prompt-echo / opening cue detector configured purely from YAML."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def _clean_question_core(question: str, strip_patterns: list[str]) -> str:
    core = question or ""
    for pat in strip_patterns:
        core = re.sub(pat, "", core, flags=re.IGNORECASE).strip()
    return core


def _disabled_result() -> dict[str, Any]:
    return {
        "detected": False,
        "status": "disabled",
        "match_kind": None,
        "matched_text": "",
        "prefix_len": 0,
        "cut_offset_char": 0,
        "confidence": 0.0,
    }


def analyse_topic_prefix(
    reference_text: str | None,
    question: str | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Detect prompt-echo prefixes or opening template cues without hardcoded dialect/language rules."""
    tcfg = cfg.get("topic_prefix", {})
    if not tcfg.get("enabled", False):
        return _disabled_result()

    text = reference_text or ""
    if not text.strip():
        return {
            "detected": False,
            "status": "missing_text",
            "match_kind": "empty",
            "matched_text": "",
            "prefix_len": 0,
            "cut_offset_char": 0,
            "confidence": 1.0,
        }

    split_pat = str(tcfg.get("punctuation_split", r"([\n,!?.;:，。！？；：])"))
    clauses = re.split(split_pat, text)
    first_clause = clauses[0].strip() if clauses else ""
    sep_len = len(clauses[1]) if len(clauses) > 1 else 0
    clause_end_idx = len(clauses[0]) + sep_len if clauses else len(text)

    explicit_cues = [str(c) for c in tcfg.get("explicit_cues", []) if str(c)]
    fillers = set(str(tcfg.get("fillers", "")))
    filler_max = int(tcfg.get("filler_max_tokens", 6))
    near_start_max = int(tcfg.get("near_start_max_tokens", 12))

    # 1. Explicit cue check (supports optional short filler prefix before cue)
    lowered_text = text.casefold()
    lowered_first = first_clause.casefold()
    for cue in explicit_cues:
        cue_low = cue.casefold()
        pos = lowered_text.find(cue_low)
        if lowered_text.startswith(cue_low) or cue_low in lowered_first:
            kind = "explicit_cue"
            if pos > 0:
                lead_sig = [ch for ch in text[:pos] if ch.isalnum()]
                if lead_sig and all(ch.casefold() in fillers or ch in fillers for ch in lead_sig) and len(lead_sig) <= filler_max:
                    kind = "opening_filler_cue"
                elif len(lead_sig) > near_start_max:
                    continue
                elif lead_sig:
                    kind = "near_start_requires_review"
            end_idx = min(len(text), max(clause_end_idx, pos + len(cue)))
            return {
                "detected": True,
                "status": "prompt_echo_prefix",
                "match_kind": kind,
                "matched_text": text[:end_idx],
                "prefix_len": end_idx,
                "cut_offset_char": end_idx,
                "confidence": 0.98 if kind == "explicit_cue" else 0.85,
            }

    # 2. Prompt lead echo check against question core
    strip_pats = [str(p) for p in tcfg.get("strip_question_prefixes", [])]
    q_core = _clean_question_core(question or "", strip_pats)
    min_lead = int(tcfg.get("min_lead_len", 5))
    q_lead = re.sub(r"^[\s,!?.;:，。！？；：]+", "", q_core)[:15]
    if len(q_lead) >= max(1, min_lead - 1) and first_clause:
        lead_len = 0
        q_lead_low = q_lead.casefold()
        for span_len in range(min(len(q_lead_low), len(lowered_first)), min_lead - 1, -1):
            if q_lead_low[:span_len] in lowered_first:
                lead_len = span_len
                break
        if lead_len >= min_lead:
            end_idx = min(len(text), clause_end_idx)
            return {
                "detected": True,
                "status": "prompt_echo_prefix",
                "match_kind": "prompt_lead_echo",
                "matched_text": text[:end_idx],
                "prefix_len": end_idx,
                "cut_offset_char": end_idx,
                "confidence": 0.95,
            }

    # 3. Fuzzy overlap between first clause and question core
    min_fuzzy = int(tcfg.get("min_fuzzy_len", 5))
    fuzzy_min = float(tcfg.get("fuzzy_ratio_min", 0.65))
    if len(first_clause) >= min_fuzzy and len(q_core) >= min_fuzzy:
        target_q = q_core[: len(first_clause) + 6]
        sm = SequenceMatcher(None, first_clause.casefold(), target_q.casefold())
        ratio = sm.ratio()
        match = sm.find_longest_match(0, len(first_clause), 0, len(target_q))
        if ratio >= fuzzy_min and match.size >= min_fuzzy:
            end_idx = min(len(text), clause_end_idx)
            return {
                "detected": True,
                "status": "prompt_echo_prefix",
                "match_kind": "fuzzy_overlap",
                "matched_text": text[:end_idx],
                "prefix_len": end_idx,
                "cut_offset_char": end_idx,
                "confidence": round(float(ratio), 3),
            }

    return {
        "detected": False,
        "status": "clean_direct_response",
        "match_kind": "direct_answer",
        "matched_text": "",
        "prefix_len": 0,
        "cut_offset_char": 0,
        "confidence": 1.0,
    }
