from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


SCHEMA_VERSION = "2026-07-12-source-grounded-clinical-replay-v7-initial-scope"

CLINICAL_PHASES = {
    "initial_presentation": 0,
    "early_workup": 1,
    "targeted_workup": 2,
    "clinical_progression": 3,
    "advanced_workup": 4,
    "confirmatory_testing": 5,
    "treatment_or_outcome": 6,
}

ACQUISITION_MODES = {
    "history_recall",
    "bedside_observation",
    "requestable_test",
    "requestable_report",
    "requestable_raw_image",
    "spontaneous_clinical_event",
    "treatment_or_outcome_only",
}

QUERY_TYPES = {
    "ASK_HISTORY",
    "REQUEST_EXAM",
    "ORDER_TEST",
    "ORDER_PROCEDURE",
    "REQUEST_RESULT",
    "WAIT_REASSESS",
    "REVIEW_IMAGE",
}

TIME_SCOPES = {"initial", "current", "latest", "trend", "repeat", "unspecified"}

EVIDENCE_CATEGORIES = {
    "history",
    "symptom",
    "clinical_course",
    "admission_observation",
    "physical_exam",
    "routine_lab",
    "targeted_lab",
    "imaging_report",
    "raw_image",
    "pathology_report",
    "raw_pathology_or_microscopy_image",
    "microbiology",
    "serology",
    "molecular",
    "genetic",
    "procedure",
    "treatment_response",
    "outcome_context",
}

AVAILABILITY_STATUSES = {
    "reported",
    "negative",
    "not_performed",
    "not_reported",
    "unavailable",
    "inconclusive",
}

SOURCE_TYPES = {
    "case_narrative",
    "contemporaneous_report",
    "objective_figure_caption",
    "retrospective_figure_caption",
    "discussion_or_retrospective_interpretation",
}

DISCLOSURE_POLICIES = {
    "return_if_requested",
    "return_if_specific_test_requested",
    "withhold_final_diagnosis_leakage",
    "hidden_reference_only",
}

DIAGNOSTIC_STAGES = {
    "initial",
    "diagnostic_workup",
    "confirmatory",
    "postdiagnostic_treatment_or_followup",
}

TEMPORAL_ROLES = {
    "first_contact_visible",
    "early_diagnostic_workup",
    "interval_progression",
    "admission_observation",
    "targeted_result",
    "confirmatory_result",
    "postdiagnostic_intervention_or_followup",
}

INITIAL_SCOPE_WARNING_PATTERNS = {
    "later relapse/recurrence language": r"\b(relaps(?:e|es|ed|ing)|recurr(?:ed|ence|ences|ent|ing)|recurrent)\b",
    "treatment response/dependence language": r"\b(steroid taper|tapering|withdraw(?:al|n)|drug withdrawal|rapid improvement|treatment initiated|treatment response|dependen(?:ce|t))\b",
    "later time anchor language": r"\b(subsequent months|over the next|by the end of|postoperative|post-operative|after surgery)\b",
    "late intervention/outcome language": r"\b(resection|hepatectomy|chemotherapy|transplant|outcome|discharged|died)\b",
    "confirmatory advanced-result language": r"\b(UBA1|somatic mutation|genetic testing|molecular testing|bone marrow|histology confirmed|immunohistochemistry)\b",
}

QUERY_STATUSES = {
    "answered",
    "partially_answered",
    "not_reported",
    "not_performed",
    "unavailable",
    "inconclusive",
    "irrelevant",
    "ambiguous",
    "already_disclosed",
    "answered_partial",
    "not_yet_available",
    "ordered",
    "no_result_currently_available",
    "no_new_clinical_update",
    "invalid_action",
}

CLINICIAN_ACTIONS = {"ASK_HISTORY", "REQUEST_EXAM", "ORDER_TEST", "ORDER_PROCEDURE", "REQUEST_RESULT", "WAIT_REASSESS", "REVIEW_IMAGE", "FINAL"}
EVIDENCE_ACTIONS = CLINICIAN_ACTIONS - {"FINAL"}
EPISODE_TERMINATIONS = {"final_submitted", "final_submitted_incorrect", "budget_exhausted", "error"}


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str]
    errors: list[str]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _flatten_initial_text(initial: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "chief_complaint",
        "first_contact_summary",
        "pre_referral_summary",
        "initial_context",
        "encounter_context",
    ):
        pieces.append(_text(initial.get(key)))
    for key in ("presenting_symptoms", "known_prior_history", "initial_available_data"):
        value = initial.get(key)
        if isinstance(value, list):
            pieces.extend(_text(item) for item in value)
    return "\n".join(piece for piece in pieces if piece)


def validate_case(data: dict[str, Any]) -> ValidationResult:
    warnings: list[str] = []
    errors: list[str] = []

    if not isinstance(data, dict):
        return ValidationResult(False, [], ["case must be a JSON object"])

    if data.get("schema_version") != SCHEMA_VERSION:
        warnings.append("schema_version is missing or differs from the current version")
    if not _text(data.get("case_id")):
        errors.append("case_id is required")

    initial = data.get("initial_observation")
    if not isinstance(initial, dict):
        errors.append("initial_observation must be an object")
    else:
        if not _text(initial.get("demographics")):
            warnings.append("initial_observation.demographics is empty")
        if not _text(initial.get("chief_complaint")):
            warnings.append("initial_observation.chief_complaint is empty")
        forbidden = {"final_diagnosis", "pmc_id", "article_title"}
        leaked = sorted(key for key in forbidden if initial.get(key))
        if leaked:
            errors.append(f"initial_observation contains forbidden leakage fields: {', '.join(leaked)}")
        initial_text = _flatten_initial_text(initial)
        scope_flags = [
            label
            for label, pattern in INITIAL_SCOPE_WARNING_PATTERNS.items()
            if re.search(pattern, initial_text, re.IGNORECASE)
        ]
        if scope_flags:
            warnings.append(
                "initial_observation may include post-first-contact information: "
                + "; ".join(scope_flags)
            )

    hidden = data.get("hidden_reference")
    if not isinstance(hidden, dict):
        errors.append("hidden_reference must be an object")
    elif not _text(hidden.get("final_diagnosis_verbatim")):
        warnings.append("hidden_reference.final_diagnosis_verbatim is empty")

    budget = data.get("budget")
    if not isinstance(budget, dict):
        errors.append("budget must be an object")
    else:
        rounds = budget.get("max_actions", budget.get("max_query_rounds"))
        if not isinstance(rounds, int) or rounds <= 0:
            errors.append("budget.max_actions must be a positive integer")
        if rounds < 15:
            warnings.append("pilot protocol recommends at least 15 total dialogue turns")
        if rounds not in {15, 25, 35}:
            warnings.append("recommended budget-curve settings are 15, 25, or 35 total dialogue turns")

    evidence = data.get("evidence_items")
    if not isinstance(evidence, list):
        errors.append("evidence_items must be a list")
        evidence = []

    seen_ids: set[str] = set()
    for index, item in enumerate(evidence):
        context = f"evidence_items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{context} must be an object")
            continue

        evidence_id = _text(item.get("evidence_id"))
        if not evidence_id:
            errors.append(f"{context}.evidence_id is required")
        elif evidence_id in seen_ids:
            errors.append(f"duplicate evidence_id: {evidence_id}")
        seen_ids.add(evidence_id)

        category = _text(item.get("category"))
        if category not in EVIDENCE_CATEGORIES:
            errors.append(f"{evidence_id or context}: unknown category '{category}'")

        status = _text(item.get("availability_status"))
        if status not in AVAILABILITY_STATUSES:
            errors.append(f"{evidence_id or context}: unknown availability_status '{status}'")

        source_type = _text(item.get("source_type"))
        if source_type not in SOURCE_TYPES:
            warnings.append(f"{evidence_id or context}: unknown or missing source_type '{source_type}'")

        policy = _text(item.get("disclosure_policy"))
        if policy not in DISCLOSURE_POLICIES:
            errors.append(f"{evidence_id or context}: unknown disclosure_policy '{policy}'")

        phase = _text(item.get("clinical_phase"))
        if phase not in CLINICAL_PHASES:
            errors.append(f"{evidence_id or context}: unknown clinical_phase '{phase}'")
        phase_index = item.get("phase_index")
        if not isinstance(phase_index, int) or phase_index < 0:
            errors.append(f"{evidence_id or context}: phase_index must be a non-negative integer")
        elif phase in CLINICAL_PHASES and phase_index != CLINICAL_PHASES[phase]:
            warnings.append(f"{evidence_id or context}: phase_index does not match clinical_phase")

        acquisition_mode = _text(item.get("acquisition_mode"))
        if acquisition_mode not in ACQUISITION_MODES:
            errors.append(f"{evidence_id or context}: unknown acquisition_mode '{acquisition_mode}'")

        diagnostic_stage = item.get("diagnostic_stage")
        if diagnostic_stage is not None and _text(diagnostic_stage) not in DIAGNOSTIC_STAGES:
            errors.append(f"{evidence_id or context}: unknown diagnostic_stage '{diagnostic_stage}'")

        temporal_role = item.get("temporal_role")
        if temporal_role is not None and _text(temporal_role) not in TEMPORAL_ROLES:
            errors.append(f"{evidence_id or context}: unknown temporal_role '{temporal_role}'")
        if _text(temporal_role) == "first_contact_visible" and category not in {
            "history",
            "symptom",
            "physical_exam",
            "admission_observation",
        }:
            warnings.append(
                f"{evidence_id or context}: first_contact_visible temporal_role on category '{category}' should be audited"
            )
        if _text(temporal_role) == "first_contact_visible":
            item_text = " ".join(
                [
                    _text(item.get("answer_text")),
                    _text(item.get("source_quote")),
                    _text(item.get("time_anchor")),
                    _text(item.get("canonical_target")),
                ]
            )
            item_scope_flags = [
                label
                for label, pattern in INITIAL_SCOPE_WARNING_PATTERNS.items()
                if re.search(pattern, item_text, re.IGNORECASE)
            ]
            if item_scope_flags:
                warnings.append(
                    f"{evidence_id or context}: first_contact_visible evidence may mix later course: "
                    + "; ".join(item_scope_flags)
                )

        allowed_actions = item.get("allowed_actions")
        if allowed_actions is not None:
            if not isinstance(allowed_actions, list):
                errors.append(f"{evidence_id or context}: allowed_actions must be a list")
            else:
                unknown_actions = sorted(
                    str(value) for value in allowed_actions if str(value) not in EVIDENCE_ACTIONS
                )
                if unknown_actions:
                    errors.append(
                        f"{evidence_id or context}: unknown allowed_actions: {', '.join(unknown_actions)}"
                    )

        if not _text(item.get("repeat_key")):
            warnings.append(f"{evidence_id or context}: repeat_key is empty")

        image_details = item.get("image_details")
        if image_details is not None and not isinstance(image_details, dict):
            warnings.append(f"{evidence_id or context}: image_details is not an object")
        elif isinstance(image_details, dict):
            hidden_interpretation = _text(image_details.get("diagnostic_significance_hidden"))
            if hidden_interpretation and hidden_interpretation in _text(item.get("answer_text")):
                warnings.append(f"{evidence_id or context}: retrospective image interpretation leaks into answer_text")

        source_quote = _text(item.get("source_quote"))
        if status in {"reported", "negative", "not_performed", "unavailable", "inconclusive"}:
            if not source_quote:
                warnings.append(f"{evidence_id or context}: source-supported status has no source_quote")

        if status == "not_reported" and _text(item.get("answer_text")):
            warnings.append(f"{evidence_id or context}: not_reported item should not invent answer_text")

        if source_type in {
            "retrospective_figure_caption",
            "discussion_or_retrospective_interpretation",
        } and policy == "return_if_requested":
            warnings.append(
                f"{evidence_id or context}: retrospective content is directly disclosable; audit for leakage"
            )

        image_paths = item.get("image_paths")
        if category in {"raw_image", "raw_pathology_or_microscopy_image"}:
            if not isinstance(image_paths, list) or not any(_text(value) for value in image_paths):
                warnings.append(f"{evidence_id or context}: raw-image item has no image_paths")

    return ValidationResult(not errors, warnings, errors)
