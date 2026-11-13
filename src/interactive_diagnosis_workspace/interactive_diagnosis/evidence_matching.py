from __future__ import annotations

import re
from typing import Any


RETURNABLE_POLICIES = {"return_if_requested", "return_if_specific_test_requested"}

STOPWORDS = {
    "please", "provide", "result", "results", "current", "initial", "patient", "test", "tests",
    "finding", "findings", "report", "including", "any", "during", "with", "from", "what", "were",
    "blood", "serum", "plasma", "laboratory", "screening", "assess", "evaluate", "obtain",
    "anti", "antibody", "antibodies", "assay", "level", "levels", "activity", "confirm", "exclude",
    "and", "for", "order", "evaluation", "panel", "marker", "markers", "the", "this", "that",
    "scan", "imaging", "procedure", "study", "analysis", "status", "await", "previously", "ordered",
    "to", "of", "a", "an", "in", "on", "by", "or", "as", "is", "are",
}

ACTION_CATEGORY_MAP = {
    "ASK_HISTORY": {"history", "symptom", "clinical_course"},
    "REQUEST_EXAM": {"physical_exam", "admission_observation"},
    "ORDER_TEST": {
        "routine_lab", "targeted_lab", "microbiology", "serology", "molecular", "genetic",
        "imaging_report", "pathology_report", "procedure", "raw_image",
    },
    "ORDER_PROCEDURE": {"pathology_report", "procedure"},
    "REQUEST_RESULT": {
        "routine_lab", "targeted_lab", "microbiology", "serology", "molecular", "genetic",
        "imaging_report", "pathology_report", "procedure", "treatment_response", "raw_image",
    },
    "REVIEW_IMAGE": {"raw_image", "raw_pathology_or_microscopy_image"},
    "WAIT_REASSESS": {"history", "symptom", "clinical_course", "physical_exam", "admission_observation", "treatment_response"},
}

MODALITY_PATTERNS = {
    "ct": r"\b(ct|computed tomography|pet[-/ ]?ct)\b",
    "mri": r"\b(mri|magnetic resonance)\b",
    "pet": r"\b(pet|pet[-/ ]?ct)\b",
    "ultrasound": r"\b(ultrasound|sonography|sonographic|ceus)\b",
    "xray": r"\b(x[- ]?ray|radiograph|radiography)\b",
}

PROCEDURE_TREATMENT_RE = re.compile(
    r"\b(resect|resection|surgery|surgical|hepatectomy|amputation|angioplasty|treat|treatment|"
    r"therapy|transplant|ablation|embolization|start|administer)\w*\b",
    re.IGNORECASE,
)


def source_items(case_data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in case_data.get("evidence_items") or []
        if isinstance(item, dict) and item.get("disclosure_policy") in RETURNABLE_POLICIES
    ]


def inferred_actions_for_item(item: dict[str, Any]) -> set[str]:
    category = str(item.get("category") or "")
    mode = str(item.get("acquisition_mode") or "")
    actions: set[str] = set()
    if category in {"history", "symptom"}:
        actions.add("ASK_HISTORY")
    if category == "clinical_course":
        actions.update({"ASK_HISTORY", "WAIT_REASSESS"})
    if category in {"physical_exam", "admission_observation"}:
        actions.add("REQUEST_EXAM")
    if category in {
        "routine_lab", "targeted_lab", "microbiology", "serology", "molecular",
        "genetic", "imaging_report", "pathology_report", "procedure", "treatment_response",
    }:
        actions.update({"ORDER_TEST", "REQUEST_RESULT"})
    if category in {"raw_image", "raw_pathology_or_microscopy_image"}:
        actions.add("REVIEW_IMAGE")
    if mode == "spontaneous_clinical_event":
        actions.add("WAIT_REASSESS")
    return actions or {"REQUEST_RESULT"}


def action_allowed_for_item(item: dict[str, Any], action: str) -> bool:
    allowed_actions = item.get("allowed_actions")
    if isinstance(allowed_actions, list) and allowed_actions:
        allowed = {str(value).strip().upper() for value in allowed_actions}
    else:
        allowed = inferred_actions_for_item(item)
    return action in allowed


def is_postdiagnostic_item(item: dict[str, Any]) -> bool:
    stage = str(item.get("diagnostic_stage") or "").lower()
    if stage in {"postdiagnostic", "postdiagnostic_treatment_or_followup", "treatment_or_outcome"}:
        return True
    phase = str(item.get("clinical_phase") or "")
    mode = str(item.get("acquisition_mode") or "")
    category = str(item.get("category") or "")
    return (
        phase == "treatment_or_outcome"
        or mode == "treatment_or_outcome_only"
        or category == "outcome_context"
    )


def item_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("canonical_target") or ""),
            str(item.get("repeat_key") or ""),
            str(item.get("answer_text") or ""),
            str(item.get("time_anchor") or ""),
            str(item.get("category") or ""),
            str((item.get("image_details") or {}).get("image_type") or ""),
            str((item.get("image_details") or {}).get("image_finding") or ""),
            str((item.get("image_details") or {}).get("report_finding") or ""),
        ]
    ).lower()


def normalize_match_text(text: str) -> str:
    normalized = text.lower()
    replacements = {
        "hba1c": "hba1c hemoglobin a1c glycated hemoglobin",
        "hb a1c": "hba1c hemoglobin a1c glycated hemoglobin",
        "hb electrophoresis": "hb hemoglobin electrophoresis hemoglobinopathy variant",
        "hbe": "hbe hemoglobin e",
        "hbf": "hbf hemoglobin f",
        "hba2": "hba2 hemoglobin a2",
        "glycated haemoglobin": "hba1c hemoglobin a1c glycated hemoglobin",
        "glycated hemoglobin": "hba1c hemoglobin a1c glycated hemoglobin",
        "egfr": "egfr estimated glomerular filtration rate",
        "acr": "acr albumin creatinine ratio",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_match_text(text))
        if len(token) > 1 and token not in STOPWORDS
    }


def requested_modalities(text: str) -> set[str]:
    lowered = text.lower()
    modalities: set[str] = set()
    for modality, pattern in MODALITY_PATTERNS.items():
        if re.search(pattern, lowered):
            modalities.add(modality)
    return modalities


def modality_matches(item: dict[str, Any], modalities: set[str]) -> bool:
    if not modalities:
        return True
    text = item_text(item)
    return any(re.search(MODALITY_PATTERNS[modality], text) for modality in modalities)


def allowed_categories_for_action(action: str, request: str) -> set[str]:
    allowed = set(ACTION_CATEGORY_MAP.get(action, set()))
    text = request.lower()
    if action in {"ORDER_TEST", "ORDER_PROCEDURE", "REQUEST_RESULT", "REVIEW_IMAGE"}:
        if re.search(r"\b(ct|mri|pet|pet-ct|ultrasound|sonography|ceus|x-ray|radiograph|imaging)\b", text):
            if action == "REVIEW_IMAGE":
                allowed.intersection_update({"raw_image"})
            else:
                allowed.intersection_update({"imaging_report"})
        elif re.search(r"\b(biopsy|histolog|patholog|immunohist|cytolog|marrow)\w*\b", text):
            if action == "REVIEW_IMAGE":
                allowed.intersection_update({"raw_pathology_or_microscopy_image"})
            else:
                allowed.intersection_update({"pathology_report", "procedure"})
        elif re.search(r"\b(gene|genetic|mutation|sequenc|molecular|ngs|fish|karyotype|uba1)\w*\b", text):
            allowed.intersection_update({"genetic", "molecular", "raw_image"})
    return allowed


def pathology_or_procedure_request(request: str) -> bool:
    return bool(re.search(r"\b(biopsy|histolog|patholog|immunohist|cytolog|marrow)\w*\b", request, re.I))


def match_score(item: dict[str, Any], request: str) -> int:
    request_tokens = tokens(request)
    if not request_tokens:
        return 0
    item_tokens = tokens(item_text(item))
    score = len(request_tokens.intersection(item_tokens))
    lowered = normalize_match_text(request)
    text = normalize_match_text(item_text(item))
    for phrase in ("uba1", "afp", "alpha fetoprotein", "bone marrow", "mri", "ceus", "ultrasound", "ct"):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lowered) and re.search(
            rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text
        ):
            score += 4
    if re.search(r"alpha[- ]fetoprotein", lowered) and "afp" in text:
        score += 6
    return score


def split_compound_request(request: str) -> list[str]:
    # Split only on strong conjunctions. "with" often binds a coherent study
    # such as "bone marrow biopsy with cytologic examination" and should remain
    # one component.
    parts = [
        part.strip(" ;.")
        for part in re.split(r"\s+(?:and|plus)\s+", request, flags=re.IGNORECASE)
        if part.strip(" ;.")
    ]
    return parts or [request.strip()]


def filter_candidates(
    case_data: dict[str, Any],
    action: str,
    request: str,
    current_phase_index: int | None = None,
    include_future: bool = False,
    include_postdiagnostic: bool = False,
) -> list[dict[str, Any]]:
    allowed_categories = allowed_categories_for_action(action, request)
    modalities = set() if pathology_or_procedure_request(request) else requested_modalities(request)
    candidates: list[dict[str, Any]] = []
    for item in source_items(case_data):
        if not include_postdiagnostic and is_postdiagnostic_item(item):
            continue
        if not action_allowed_for_item(item, action):
            continue
        if allowed_categories and item.get("category") not in allowed_categories:
            continue
        if not modality_matches(item, modalities):
            continue
        phase = int(item.get("phase_index", 0) or 0)
        if current_phase_index is not None and not include_future and phase > current_phase_index:
            continue
        score = match_score(item, request)
        if score <= 0:
            continue
        if score < 2 and not pathology_or_procedure_request(request):
            continue
        candidates.append(item)
    return candidates


def best_matches(
    case_data: dict[str, Any],
    action: str,
    request: str,
    current_phase_index: int,
    include_future: bool,
    max_items: int = 2,
) -> list[dict[str, Any]]:
    matches = filter_candidates(
        case_data,
        action,
        request,
        current_phase_index=current_phase_index,
        include_future=include_future,
    )
    if include_future:
        matches.sort(
            key=lambda item: (
                -match_score(item, request),
                int(item.get("phase_index", 0) or 0),
                str(item.get("evidence_id") or ""),
            )
        )
    else:
        matches.sort(
            key=lambda item: (
                int(item.get("phase_index", 0) or 0) > current_phase_index,
                int(item.get("phase_index", 0) or 0),
                -match_score(item, request),
                str(item.get("evidence_id") or ""),
            )
        )
    if not matches:
        return []
    best_phase = int(matches[0].get("phase_index", 0) or 0)
    best_score = match_score(matches[0], request)
    selected = [
        item
        for item in matches
        if int(item.get("phase_index", 0) or 0) == best_phase
        and match_score(item, request) >= max(1, best_score - 1)
    ]
    return selected[:max_items]


def public_evidence_view(item: dict[str, Any]) -> dict[str, Any]:
    view = {
        "category": item.get("category", ""),
        "time_anchor": item.get("time_anchor", ""),
        "finding": item.get("answer_text", ""),
        "source_status": item.get("availability_status", ""),
    }
    image_type = (item.get("image_details") or {}).get("image_type")
    if image_type:
        view["image_type"] = image_type
    return view
