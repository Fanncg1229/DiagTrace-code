from __future__ import annotations

import re
import unicodedata
from collections import Counter
from statistics import mean
from typing import Any


def normalize_diagnosis(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def diagnosis_match(predicted: str, references: list[str]) -> tuple[bool, str, str]:
    normalized_prediction = normalize_diagnosis(predicted)
    for reference in references:
        normalized_reference = normalize_diagnosis(reference)
        if not normalized_reference:
            continue
        if normalized_prediction == normalized_reference:
            return True, "exact_normalized", reference
        if normalized_reference in normalized_prediction:
            return True, "reference_contained_in_prediction", reference
    return False, "no_match", ""


def evaluate_episode(case_data: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    hidden = case_data.get("hidden_reference") or {}
    gold = str(hidden.get("final_diagnosis_verbatim") or hidden.get("final_diagnosis") or "")
    aliases = [str(value) for value in hidden.get("accepted_aliases", []) if value]
    references = [gold, *aliases]
    final = episode.get("final") or {}
    predicted = str(final.get("diagnosis") or "")
    correct, match_type, matched_reference = diagnosis_match(predicted, references)
    try:
        confidence = float(final.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    turns = episode.get("turns") or []
    action_counts: Counter[str] = Counter()
    report_counts: Counter[str] = Counter()
    report_repair_attempted = 0
    report_repair_failed = 0
    report_repair_validation_errors: Counter[str] = Counter()
    phase_trajectory: list[int] = []
    for turn in turns:
        clinician = turn.get("clinician") or {}
        action_counts[str(clinician.get("action") or "UNKNOWN")] += 1
        report = turn.get("report")
        if report:
            report_counts[str(report.get("status") or "unknown")] += 1
            if report.get("repair_attempted"):
                report_repair_attempted += 1
                if report.get("repair_failed"):
                    report_repair_failed += 1
                for error in report.get("validation_errors") or []:
                    report_repair_validation_errors[str(error)] += 1
            if isinstance(report.get("phase_index"), int):
                phase_trajectory.append(report["phase_index"])

    disclosed = set(episode.get("disclosed_evidence_ids") or [])
    supporting_ids = [str(value) for value in final.get("supporting_evidence_ids", []) if value]
    invalid_supporting_ids = sorted(set(supporting_ids) - disclosed)
    non_final_turns = sum(count for action, count in action_counts.items() if action != "FINAL")
    answered_turns = report_counts.get("found", 0) + report_counts.get("already_disclosed", 0)
    answer_yield = answered_turns / non_final_turns if non_final_turns else 0.0

    return {
        "case_id": case_data.get("case_id", ""),
        "gold_diagnosis": gold,
        "accepted_aliases": aliases,
        "predicted_diagnosis": predicted,
        "diagnosis_correct": correct,
        "match_type": match_type,
        "matched_reference": matched_reference,
        "confidence": confidence,
        "brier_score": (confidence - float(correct)) ** 2,
        "reasoning_present": bool(str(final.get("reasoning") or "").strip()),
        "supporting_evidence_ids": supporting_ids,
        "invalid_supporting_evidence_ids": invalid_supporting_ids,
        "turns_used": int(episode.get("turns_used") or len(turns)),
        "max_turns": int(episode.get("max_turns") or 0),
        "termination": episode.get("termination", ""),
        "action_counts": dict(sorted(action_counts.items())),
        "report_status_counts": dict(sorted(report_counts.items())),
        "report_repair_attempted_count": report_repair_attempted,
        "report_repair_failed_count": report_repair_failed,
        "report_repair_validation_error_counts": dict(sorted(report_repair_validation_errors.items())),
        "disclosed_evidence_count": len(disclosed),
        "answer_yield": answer_yield,
        "final_phase_index": phase_trajectory[-1] if phase_trajectory else 0,
        "phase_trajectory": phase_trajectory,
        "model_usage": episode.get("model_usage") or {},
    }


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in evaluations if "diagnosis_correct" in item]
    if not completed:
        return {"num_cases": 0}
    return {
        "num_cases": len(completed),
        "accuracy": mean(float(item["diagnosis_correct"]) for item in completed),
        "mean_turns_used": mean(float(item["turns_used"]) for item in completed),
        "mean_confidence": mean(float(item["confidence"]) for item in completed),
        "mean_brier_score": mean(float(item["brier_score"]) for item in completed),
        "mean_answer_yield": mean(float(item["answer_yield"]) for item in completed),
        "all_final_reasoning_present": all(bool(item["reasoning_present"]) for item in completed),
        "total_report_repair_attempted": sum(int(item.get("report_repair_attempted_count") or 0) for item in completed),
        "total_report_repair_failed": sum(int(item.get("report_repair_failed_count") or 0) for item in completed),
        "total_clinician_tokens": sum(
            int(item.get("model_usage", {}).get("clinician", {}).get("total_tokens", 0))
            for item in completed
        ),
        "total_report_tokens": sum(
            int(item.get("model_usage", {}).get("report", {}).get("total_tokens", 0))
            for item in completed
        ),
    }
