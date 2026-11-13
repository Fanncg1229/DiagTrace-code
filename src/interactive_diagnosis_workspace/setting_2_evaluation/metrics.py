from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COARSE_CATEGORY_WEIGHTS: dict[str, float] = {
    "initial_presentation": 1.0,
    "clinical_history": 0.5,
    "physical_examination": 0.5,
    "lab_tests": 1.0,
    "microbiology": 1.0,
    "imaging": 1.5,
    "pathology": 2.0,
    "procedure_or_intervention": 1.0,
    "medication_or_treatment": 0.75,
    "admission_or_clinical_course": 0.75,
    "other_or_off_protocol": 0.25,
}


FINAL_TRAJECTORY_ACTIONS = {"FINAL_TRAJECTORY", "FINAL"}


@dataclass(frozen=True)
class SequenceStep:
    category: str
    weight: float
    source_id: str = ""
    label: str = ""
    raw_category: str = ""
    turn_number: int | None = None
    action: str = ""
    request: str = ""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold()).strip()


def weight_for_category(category: str) -> float:
    return COARSE_CATEGORY_WEIGHTS.get(category, COARSE_CATEGORY_WEIGHTS["other_or_off_protocol"])


def categorize_source_evidence(raw: dict[str, Any]) -> str:
    category = normalize_text(raw.get("category") or raw.get("evidence_type"))
    source_text = normalize_text(
        " ".join(
            str(raw.get(field) or "")
            for field in ("source_text", "source_quote", "source_span", "repeat_key")
        )
    )
    image_finding = bool(raw.get("image_finding") or raw.get("is_image_finding"))

    if category in {"history", "symptom"}:
        return "clinical_history"
    if category in {"physical_exam", "exam"}:
        return "physical_examination"
    if category in {"microbiology", "culture"}:
        return "microbiology"
    if category in {"imaging_report", "imaging", "radiology", "image_finding"} or image_finding:
        return "imaging"
    if category in {"pathology_report", "pathology", "histology", "cytology"}:
        return "pathology"
    if category in {"procedure", "intervention"}:
        return "procedure_or_intervention"
    if category in {"treatment_response", "treatment", "medication"}:
        return "medication_or_treatment"
    if category in {"clinical_course", "outcome", "follow_up", "followup"}:
        return "admission_or_clinical_course"
    if category in {"routine_lab", "targeted_lab", "serology", "molecular_test", "genetic"}:
        return "lab_tests"

    return categorize_request_text(source_text, action="")


def categorize_request_text(request: str, action: str) -> str:
    text = normalize_text(request)
    action = action.upper()

    if action == "ASK_HISTORY":
        return "clinical_history"
    if action == "REQUEST_EXAM":
        return "physical_examination"

    if re.search(r"\b(search|literature|pubmed|internet|guideline|article)\b", text):
        return "other_or_off_protocol"
    if re.search(r"\b(plan|management plan|next step|strategy)\b", text):
        return "other_or_off_protocol"
    if re.search(
        r"\b(admission|admit|hospitali[sz]ation|icu|ward|transfer|course|progression|follow[- ]?up|outcome|recurrence|"
        r"reassessment|re[- ]?evaluation|interval change|subsequent status|clinical update|later update)\b",
        text,
    ):
        return "admission_or_clinical_course"
    if re.search(r"\b(medication|drug|antibiotic|steroid|prednisone|therapy|treatment|chemotherapy|immunosuppress\w*|response)\b", text):
        return "medication_or_treatment"
    if re.search(r"\b(culture|microbiology|bacterial|fungal|mycobacterial|viral|pathogen|organism|gram stain|blood culture|urine culture|sputum)\b", text):
        return "microbiology"
    if re.search(r"\b([a-z]*patholog\w*|[a-z]*histolog\w*|[a-z]*cytolog\w*|biopsy result|resection specimen|immunohistochem\w*|ihc|stain|bone marrow|marrow|smear|morpholog\w*|hematoxylin|eosin)\b", text):
        return "pathology"
    if re.search(r"\b(ct|mri|ultrasound|sonograph\w*|x[- ]?ray|radiograph|pet|scan|imaging|radiology|ceus|elastograph\w*|echocardiograph\w*|doppler|angiograph\w*|endoscop\w*)\b", text):
        return "imaging"
    if re.search(r"\b(biopsy|resection|surgery|surgical|procedure|intervention|aspiration|sampling|laparoscopy|bronchoscopy|colonoscopy)\b", text):
        return "procedure_or_intervention"
    if re.search(r"\b(lab|laborator\w*|blood|cbc|hemoglobin|platelet|leukocyte|wbc|rbc|creatinine|bilirubin|enzyme|ast|alt|alp|ggt|crp|esr|afp|antibody|serolog\w*|ana|anca|igm|igg|ige|glucose|hba1c|electrophoresis|mutation|genetic|molecular|sequencing|pcr|gene|variant)\b", text):
        return "lab_tests"

    if action == "ORDER_PROCEDURE":
        return "procedure_or_intervention"
    if action in {"ORDER_TEST", "REQUEST_RESULT"}:
        return "lab_tests"
    return "other_or_off_protocol"


def collapse_adjacent_duplicates(steps: list[SequenceStep]) -> list[SequenceStep]:
    collapsed: list[SequenceStep] = []
    for step in steps:
        if collapsed and collapsed[-1].category == step.category:
            continue
        collapsed.append(step)
    return collapsed


def first_unique_sequence(steps: list[SequenceStep]) -> list[SequenceStep]:
    seen: set[str] = set()
    out: list[SequenceStep] = []
    for step in collapse_adjacent_duplicates(steps):
        if step.category in seen:
            continue
        seen.add(step.category)
        out.append(step)
    return out


def gold_steps_from_case(case_data: dict[str, Any], *, exclude_initial_phase: bool = True) -> list[SequenceStep]:
    raw_items = case_data.get("evidence_items") or case_data.get("evidence_events") or []
    steps: list[SequenceStep] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        try:
            phase_index = int(item.get("phase_index", 0) or 0)
        except (TypeError, ValueError):
            phase_index = 0
        if exclude_initial_phase and phase_index <= 0:
            continue
        category = categorize_source_evidence(item)
        steps.append(
            SequenceStep(
                category=category,
                weight=weight_for_category(category),
                source_id=str(item.get("id") or item.get("event_id") or f"G{index + 1:03d}"),
                label=str(item.get("repeat_key") or item.get("source_section") or ""),
                raw_category=str(item.get("category") or item.get("evidence_type") or ""),
            )
        )
    return steps


def has_initial_presentation(case_data: dict[str, Any]) -> bool:
    initial = case_data.get("initial_observation")
    if isinstance(initial, dict) and any(bool(value) for value in initial.values()):
        return True
    for item in case_data.get("evidence_items") or case_data.get("evidence_events") or []:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("phase_index", 0) or 0) <= 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def initial_presentation_step() -> SequenceStep:
    return SequenceStep(
        category="initial_presentation",
        weight=weight_for_category("initial_presentation"),
        source_id="INITIAL",
        label="Initial presentation shown to LLM A",
        raw_category="initial_presentation",
    )


def request_from_clinician(clinician: dict[str, Any]) -> str:
    for field in ("request", "message", "question", "test", "test_name", "procedure", "procedure_name", "result", "result_name"):
        value = clinician.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            return ", ".join(str(item) for item in value if item)
        if isinstance(value, dict) and value:
            return json.dumps(value, ensure_ascii=False)
    return ""


def should_skip_prediction_turn(action: str, request: str, category: str) -> bool:
    if action in FINAL_TRAJECTORY_ACTIONS:
        return True
    # WAIT_REASSESS is a system-control action only when it does not ask for a
    # concrete documented evidence type, such as follow-up status or treatment response.
    if action == "WAIT_REASSESS" and category == "other_or_off_protocol":
        return True
    return False


def prediction_steps_from_episode(episode: dict[str, Any]) -> list[SequenceStep]:
    steps: list[SequenceStep] = []
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        clinician = turn.get("clinician") if isinstance(turn.get("clinician"), dict) else {}
        action = str(clinician.get("action") or "").upper()
        request = request_from_clinician(clinician)
        category = categorize_request_text(request, action)
        report = turn.get("report") if isinstance(turn.get("report"), dict) else {}
        report_status = normalize_text(report.get("status"))
        if report_status in {"too_broad", "invalid_request"}:
            category = "other_or_off_protocol"
        if should_skip_prediction_turn(action, request, category):
            continue
        steps.append(
            SequenceStep(
                category=category,
                weight=weight_for_category(category),
                turn_number=int(turn.get("turn_number") or len(steps) + 1),
                action=action,
                request=request,
            )
        )
    return steps


def weighted_tversky_alignment(
    gold: list[SequenceStep],
    pred: list[SequenceStep],
    *,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> dict[str, float]:
    gold_by_category: defaultdict[str, float] = defaultdict(float)
    pred_by_category: defaultdict[str, float] = defaultdict(float)
    for step in gold:
        gold_by_category[step.category] += step.weight
    for step in pred:
        pred_by_category[step.category] += step.weight
    categories = set(gold_by_category) | set(pred_by_category)
    matched = sum(min(gold_by_category[cat], pred_by_category[cat]) for cat in categories)
    missed = sum(max(gold_by_category[cat] - pred_by_category[cat], 0.0) for cat in categories)
    extra = sum(max(pred_by_category[cat] - gold_by_category[cat], 0.0) for cat in categories)
    denom = matched + alpha * extra + beta * missed
    alignment = matched / denom if denom else None
    return {
        "weighted_tversky_alignment": alignment,
        "weighted_tversky_distance": 1.0 - alignment if alignment is not None else None,
        "matched_weight": matched,
        "missed_weight": missed,
        "extra_weight": extra,
    }


def weighted_lcs_recall(gold: list[SequenceStep], pred: list[SequenceStep]) -> dict[str, float]:
    n = len(gold)
    m = len(pred)
    dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gold[i - 1].category == pred[j - 1].category:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + gold[i - 1].weight)
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    total_gold = sum(step.weight for step in gold)
    return {
        "weighted_lcs_weight": dp[n][m],
        "weighted_lcs_recall": dp[n][m] / total_gold if total_gold else None,
    }


def weighted_pairwise_order_accuracy(gold: list[SequenceStep], pred: list[SequenceStep]) -> dict[str, float | int]:
    first_pred_pos: dict[str, int] = {}
    for index, step in enumerate(pred):
        first_pred_pos.setdefault(step.category, index)

    numerator = 0.0
    denominator = 0.0
    comparable_pairs = 0
    correct_pairs = 0
    inversions = 0
    for i, left in enumerate(gold):
        left_pos = first_pred_pos.get(left.category)
        if left_pos is None:
            continue
        for right in gold[i + 1 :]:
            right_pos = first_pred_pos.get(right.category)
            if right_pos is None or right.category == left.category:
                continue
            pair_weight = math.sqrt(left.weight * right.weight)
            comparable_pairs += 1
            denominator += pair_weight
            if left_pos < right_pos:
                correct_pairs += 1
                numerator += pair_weight
            else:
                inversions += 1
    return {
        "weighted_pairwise_order_accuracy": numerator / denominator if denominator else None,
        "pairwise_comparable_pairs": comparable_pairs,
        "pairwise_correct_pairs": correct_pairs,
        "pairwise_inversions": inversions,
    }


def step_to_dict(step: SequenceStep) -> dict[str, Any]:
    return {
        "category": step.category,
        "weight": step.weight,
        "source_id": step.source_id,
        "label": step.label,
        "raw_category": step.raw_category,
        "turn_number": step.turn_number,
        "action": step.action,
        "request": step.request,
    }


def evaluate_case(
    *,
    case_id: str,
    case_data: dict[str, Any],
    episode: dict[str, Any],
    alpha: float = 2.0,
    beta: float = 1.0,
    include_initial_stage: bool = True,
) -> dict[str, Any]:
    gold_raw = gold_steps_from_case(case_data, exclude_initial_phase=include_initial_stage)
    pred_raw = prediction_steps_from_episode(episode)
    if include_initial_stage and has_initial_presentation(case_data):
        initial = initial_presentation_step()
        gold_raw = [initial, *gold_raw]
        pred_raw = [initial, *pred_raw]
    gold = first_unique_sequence(gold_raw)
    pred = first_unique_sequence(pred_raw)
    metrics = {
        **weighted_tversky_alignment(gold, pred, alpha=alpha, beta=beta),
        **weighted_lcs_recall(gold, pred),
        **weighted_pairwise_order_accuracy(gold, pred),
    }
    gold_counter = Counter(step.category for step in gold)
    pred_counter = Counter(step.category for step in pred)
    return {
        "case_id": case_id,
        "status": "completed",
        "num_gold_steps": len(gold),
        "num_pred_steps": len(pred),
        "gold_sequence": [step.category for step in gold],
        "prediction_sequence": [step.category for step in pred],
        "gold_category_counts": dict(sorted(gold_counter.items())),
        "prediction_category_counts": dict(sorted(pred_counter.items())),
        "metrics": metrics,
        "gold_steps": [step_to_dict(step) for step in gold],
        "prediction_steps": [step_to_dict(step) for step in pred],
        "gold_steps_raw": [step_to_dict(step) for step in gold_raw],
        "prediction_steps_raw": [step_to_dict(step) for step in pred_raw],
    }
