from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


def evaluate_episode(case_data: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    final = episode.get("final") if isinstance(episode.get("final"), dict) else {}
    disclosed = set(episode.get("disclosed_evidence_ids") or [])
    key_supporting = final.get("key_supporting_evidence") if isinstance(final, dict) else []
    cited_ids: list[str] = []
    if isinstance(key_supporting, list):
        for item in key_supporting:
            if isinstance(item, dict):
                cited_ids.extend(str(value) for value in item.get("evidence_ids", []) if value)
    invalid_citations = sorted(set(cited_ids) - disclosed)

    report_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    report_repair_attempted = 0
    report_repair_failed = 0
    report_repair_validation_errors: Counter[str] = Counter()
    for turn in episode.get("turns", []) or []:
        clinician = turn.get("clinician") if isinstance(turn, dict) else {}
        report = turn.get("report") if isinstance(turn, dict) else {}
        if isinstance(clinician, dict):
            action_counts[str(clinician.get("action") or "UNKNOWN")] += 1
        if isinstance(report, dict):
            report_counts[str(report.get("status") or "unknown")] += 1
            if report.get("repair_attempted"):
                report_repair_attempted += 1
                if report.get("repair_failed"):
                    report_repair_failed += 1
                for error in report.get("validation_errors") or []:
                    report_repair_validation_errors[str(error)] += 1

    non_final_turns = sum(count for action, count in action_counts.items() if action != "FINAL_TRAJECTORY")
    answered_turns = report_counts.get("found", 0) + report_counts.get("already_disclosed", 0)
    unavailable = sum(
        report_counts.get(status, 0)
        for status in (
            "not_reported",
            "ordered_but_not_documented",
            "too_broad",
            "invalid_request",
            "request_too_diagnostic",
            "no_new_update",
        )
    )

    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    gold_spans = hidden.get("diagnostic_reasoning_verbatim_spans", []) if isinstance(hidden, dict) else []
    return {
        "case_id": case_data.get("case_id", ""),
        "known_final_diagnosis": episode.get("known_final_diagnosis", ""),
        "turns_used": int(episode.get("turns_used") or 0),
        "max_turns": int(episode.get("max_turns") or 0),
        "termination": episode.get("termination", ""),
        "final_present": bool(final),
        "trajectory_summary_present": bool(str(final.get("trajectory_summary") or "").strip()) if isinstance(final, dict) else False,
        "key_supporting_evidence_count": len(key_supporting) if isinstance(key_supporting, list) else 0,
        "invalid_cited_evidence_ids": invalid_citations,
        "disclosed_evidence_count": len(disclosed),
        "gold_reasoning_span_count": len(gold_spans) if isinstance(gold_spans, list) else 0,
        "action_counts": dict(sorted(action_counts.items())),
        "report_status_counts": dict(sorted(report_counts.items())),
        "report_repair_attempted_count": report_repair_attempted,
        "report_repair_failed_count": report_repair_failed,
        "report_repair_validation_error_counts": dict(sorted(report_repair_validation_errors.items())),
        "answer_yield": answered_turns / non_final_turns if non_final_turns else 0.0,
        "unavailable_turn_rate": unavailable / non_final_turns if non_final_turns else 0.0,
        "model_usage": episode.get("model_usage") or {},
    }


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in evaluations if item.get("final_present")]
    if not completed:
        return {"num_cases": 0}
    return {
        "num_cases": len(completed),
        "mean_turns_used": mean(float(item["turns_used"]) for item in completed),
        "mean_disclosed_evidence_count": mean(float(item["disclosed_evidence_count"]) for item in completed),
        "mean_key_supporting_evidence_count": mean(float(item["key_supporting_evidence_count"]) for item in completed),
        "mean_answer_yield": mean(float(item["answer_yield"]) for item in completed),
        "mean_unavailable_turn_rate": mean(float(item["unavailable_turn_rate"]) for item in completed),
        "all_final_trajectory_summary_present": all(bool(item["trajectory_summary_present"]) for item in completed),
        "total_invalid_cited_evidence_ids": sum(len(item["invalid_cited_evidence_ids"]) for item in completed),
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
