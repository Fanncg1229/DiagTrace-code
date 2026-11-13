from __future__ import annotations

import re
from typing import Any


INTERMEDIATE_DIAGNOSIS_PATTERNS = {
    "sweet_syndrome": re.compile(r"\b(sweet'?s?\s+syndrome|acute\s+febrile\s+neutrophilic\s+dermatosis)\b", re.I),
    "li_rads_3": re.compile(r"\bli[- ]?rads\s*3\b", re.I),
    "regenerative_dysplastic_nodule": re.compile(r"\b(regenerative|dysplastic)\s+(hepatic\s+)?nodule\b", re.I),
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _normalized(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _text(value).lower()).strip()


def disclosed_text(turns: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for turn in turns:
        observation = turn.get("information_observation") or {}
        for item in observation.get("available_information") or []:
            chunks.append(_text(item.get("finding")))
            chunks.append(_text(item.get("category")))
    return "\n".join(chunk for chunk in chunks if chunk)


def intermediate_label(final_diagnosis: str, case_level_diagnosis: str, evidence_text: str) -> str | None:
    final_norm = _normalized(final_diagnosis)
    case_norm = _normalized(case_level_diagnosis)
    evidence = evidence_text.lower()
    if "vexas" in case_norm and INTERMEDIATE_DIAGNOSIS_PATTERNS["sweet_syndrome"].search(final_diagnosis):
        return "sweet_syndrome_manifestation"
    if "hepatocellular carcinoma" in case_norm or case_norm == "hcc":
        if INTERMEDIATE_DIAGNOSIS_PATTERNS["li_rads_3"].search(final_diagnosis):
            return "imaging_category_not_case_diagnosis"
        if INTERMEDIATE_DIAGNOSIS_PATTERNS["regenerative_dysplastic_nodule"].search(final_diagnosis):
            return "benign_or_premalignant_intermediate_nodule"
    for label, pattern in INTERMEDIATE_DIAGNOSIS_PATTERNS.items():
        if pattern.search(final_diagnosis) and pattern.search(evidence):
            return label
    return None


def evaluate_final_outcome(
    *,
    final_action: dict[str, Any],
    final_correct: bool,
    case_data: dict[str, Any],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    final_diagnosis = _text(final_action.get("final_diagnosis"))
    case_level = _text(hidden.get("case_level_diagnosis") or hidden.get("final_diagnosis_verbatim"))
    evidence = disclosed_text(turns)

    if final_correct:
        category = "correct_case_level_diagnosis"
        subtype = ""
    else:
        subtype = intermediate_label(final_diagnosis, case_level, evidence)
        category = "intermediate_diagnosis_as_final" if subtype else "incorrect_other"

    return {
        "outcome_category": category,
        "outcome_subtype": subtype,
        "predicted_diagnosis": final_diagnosis,
        "case_level_reference": case_level,
        "intermediate_evidence_supported": bool(subtype),
    }
