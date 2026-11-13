from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "2026-07-06-diagnostic-narrowing-v1"

STAGE_LABELS = {
    "initial_presentation",
    "initial_workup",
    "major_visual_evidence",
    "targeted_investigation",
    "pathology_or_microscopy",
    "confirmatory_evidence",
    "treatment_or_outcome_if_relevant",
    "final_synthesis",
}

ORDER_BASIS_VALUES = {
    "explicit_temporal_cue",
    "case_narrative_order",
    "clinical_stage_normalization",
    "figure_order",
    "uncertain",
}

ORDER_CONFIDENCE_VALUES = {"high", "medium", "low"}

DIAGNOSTIC_ROLE_VALUES = {
    "initial_context",
    "broad_narrowing",
    "major_narrowing",
    "rule_out_alternative",
    "confirmatory",
    "treatment_response",
    "outcome",
    "clinical_complication",
    "care_transition",
    "management_context",
    "final_synthesis_support",
}

EXPECTED_NEXT_EVIDENCE_CATEGORY_VALUES = {
    "imaging",
    "pathology",
    "microbiology",
    "genetic",
    "serology",
    "molecular",
    "routine_lab",
    "additional_clinical_context",
    "final_synthesis",
    "none",
    "uncertain",
}

EXPECTED_READINESS_VALUES = {"not_ready", "possibly_ready", "ready"}

MILESTONE_FUNCTION_VALUES = {
    "broaden_context",
    "narrow_differential",
    "localize_disease",
    "characterize_abnormality",
    "rule_out_alternative",
    "confirm_diagnosis",
    "support_final_synthesis",
    "treatment_response",
    "outcome_context",
    "clinical_complication",
    "care_transition",
    "management_context",
}

QUESTION_TYPE_VALUES = {
    "clinical_diagnostic_update",
    "image_interpretation_and_diagnostic_update",
    "pathology_or_microscopy_update",
    "confirmatory_diagnostic_update",
    "treatment_or_outcome_update",
}

MISSING_TEST_STATUS_VALUES = {
    "unavailable",
    "not_performed",
    "not_reported",
    "inconclusive",
    "unknown",
}

GUARDRAIL_FLAG_VALUES = {
    "source_statement_conflict",
    "unverified_reported_abnormality",
    "unsupported_confirmation_language",
    "premature_finalization",
    "missing_confirmatory_evidence",
    "image_text_inconsistency",
}


@dataclass
class ValidationResult:
    ok: bool
    warnings: list[str]
    errors: list[str]


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    value = (text or "").strip()
    if not value:
        raise ValueError("empty response")
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = value.find("{")
    if start < 0:
        raise ValueError("response does not contain a JSON object")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(value[start : index + 1])
    raise ValueError("could not extract complete JSON object")


def normalize_whitespace(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_diagnosis_text(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_guardrail_flags(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        flag = normalize_whitespace(value).lower()
        if flag in GUARDRAIL_FLAG_VALUES and flag not in seen:
            seen.add(flag)
            normalized.append(flag)
    return normalized


def validate_guardrail_flags(values: Any, context: str, warnings: list[str]) -> None:
    if values is None:
        return
    if not isinstance(values, list):
        warnings.append(f"{context}.guardrail_flags is not a list")
        return
    for flag in values:
        normalized = normalize_whitespace(flag).lower()
        if normalized not in GUARDRAIL_FLAG_VALUES:
            warnings.append(f"{context}.guardrail_flags contains unknown value '{flag}'")


def get_diagnosis_variants(hidden_reference: dict[str, Any]) -> list[str]:
    if not isinstance(hidden_reference, dict):
        return []
    variants = []
    final_diagnosis = hidden_reference.get("final_diagnosis_verbatim") or hidden_reference.get("final_diagnosis", "")
    if final_diagnosis:
        variants.append(final_diagnosis)
    for item in hidden_reference.get("diagnosis_synonyms_or_variants") or []:
        if item:
            variants.append(str(item))
    normalized = []
    seen = set()
    for value in variants:
        norm = normalize_diagnosis_text(value)
        if norm and norm not in seen:
            seen.add(norm)
            normalized.append(norm)
    return normalized


def validate_milestone_case(data: dict[str, Any]) -> ValidationResult:
    warnings: list[str] = []
    errors: list[str] = []

    if not isinstance(data, dict):
        return ValidationResult(False, [], ["case data is not an object"])

    milestones = data.get("evidence_milestones")
    if not isinstance(milestones, list):
        errors.append("evidence_milestones must be a list")
        milestones = []

    if not data.get("case_id"):
        warnings.append("missing case_id")

    validate_guardrail_flags(data.get("guardrail_flags"), "case", warnings)

    events = data.get("evidence_events", [])
    event_ids: set[str] = set()
    if events and not isinstance(events, list):
        errors.append("evidence_events must be a list when present")
        events = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"evidence_events[{index}] is not an object")
            continue
        event_id = normalize_whitespace(event.get("event_id"))
        if not event_id:
            warnings.append(f"evidence_events[{index}] missing event_id")
        elif event_id in event_ids:
            warnings.append(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)
        if event.get("source_order") is None:
            warnings.append(f"{event_id or index}: missing source_order")
        if not normalize_whitespace(event.get("source_span")):
            warnings.append(f"{event_id or index}: missing source_span")

    hidden_reference = data.get("hidden_reference")
    if not isinstance(hidden_reference, dict):
        errors.append("hidden_reference must be an object")
    elif not (
        normalize_whitespace(hidden_reference.get("final_diagnosis_verbatim"))
        or normalize_whitespace(hidden_reference.get("final_diagnosis"))
    ):
        warnings.append("hidden_reference.final_diagnosis_verbatim/final_diagnosis is empty")
    else:
        if "diagnostic_reasoning_verbatim_spans" in hidden_reference and not isinstance(
            hidden_reference.get("diagnostic_reasoning_verbatim_spans"), list
        ):
            warnings.append("hidden_reference.diagnostic_reasoning_verbatim_spans is not a list")
        if "diagnostic_reasoning_items" in hidden_reference and not isinstance(
            hidden_reference.get("diagnostic_reasoning_items"), list
        ):
            warnings.append("hidden_reference.diagnostic_reasoning_items is not a list")
        if "missing_confirmatory_evidence" in hidden_reference and not isinstance(
            hidden_reference.get("missing_confirmatory_evidence"), list
        ):
            warnings.append("hidden_reference.missing_confirmatory_evidence is not a list")
        if "evidence_limitations" in hidden_reference and not isinstance(
            hidden_reference.get("evidence_limitations"), list
        ):
            warnings.append("hidden_reference.evidence_limitations is not a list")
        if (
            hidden_reference.get("confirmatory_evidence_present") is not None
            and not isinstance(hidden_reference.get("confirmatory_evidence_present"), bool)
        ):
            warnings.append("hidden_reference.confirmatory_evidence_present is not a boolean")

    included = [m for m in milestones if isinstance(m, dict) and m.get("include_in_dialogue", True)]
    if len(included) < 3:
        warnings.append("fewer than 3 included milestones")
    if len(included) > 8:
        warnings.append("more than 8 included milestones; consider regrouping")

    seen_turns: set[str] = set()
    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            errors.append(f"evidence_milestones[{index}] is not an object")
            continue
        turn_id = normalize_whitespace(milestone.get("turn_id"))
        if not turn_id:
            errors.append(f"evidence_milestones[{index}] missing turn_id")
        elif turn_id in seen_turns:
            errors.append(f"duplicate turn_id: {turn_id}")
        seen_turns.add(turn_id)

        stage = normalize_whitespace(milestone.get("stage"))
        if stage and stage not in STAGE_LABELS:
            warnings.append(f"{turn_id or index}: unknown stage '{stage}'")

        order_basis = normalize_whitespace(milestone.get("order_basis"))
        if order_basis and order_basis not in ORDER_BASIS_VALUES:
            warnings.append(f"{turn_id or index}: unknown order_basis '{order_basis}'")

        order_confidence = normalize_whitespace(milestone.get("order_confidence"))
        if order_confidence and order_confidence not in ORDER_CONFIDENCE_VALUES:
            warnings.append(f"{turn_id or index}: unknown order_confidence '{order_confidence}'")

        diagnostic_role = normalize_whitespace(milestone.get("diagnostic_role"))
        if diagnostic_role and diagnostic_role not in DIAGNOSTIC_ROLE_VALUES:
            warnings.append(f"{turn_id or index}: unknown diagnostic_role '{diagnostic_role}'")

        expected_next = normalize_whitespace(milestone.get("expected_next_evidence_category"))
        if not expected_next:
            warnings.append(f"{turn_id or index}: missing expected_next_evidence_category")
        elif expected_next not in EXPECTED_NEXT_EVIDENCE_CATEGORY_VALUES:
            warnings.append(f"{turn_id or index}: unknown expected_next_evidence_category '{expected_next}'")

        expected_readiness = normalize_whitespace(milestone.get("expected_readiness"))
        if not expected_readiness:
            warnings.append(f"{turn_id or index}: missing expected_readiness")
        elif expected_readiness not in EXPECTED_READINESS_VALUES:
            warnings.append(f"{turn_id or index}: unknown expected_readiness '{expected_readiness}'")

        milestone_function = normalize_whitespace(milestone.get("milestone_function"))
        if not milestone_function:
            warnings.append(f"{turn_id or index}: missing milestone_function")
        elif milestone_function not in MILESTONE_FUNCTION_VALUES:
            warnings.append(f"{turn_id or index}: unknown milestone_function '{milestone_function}'")

        if "include_in_core_narrowing_metrics" not in milestone:
            warnings.append(f"{turn_id or index}: missing include_in_core_narrowing_metrics")

        if milestone.get("diagnosis_revealing") is True and diagnostic_role not in {
            "confirmatory",
            "final_synthesis_support",
            "major_narrowing",
        }:
            warnings.append(f"{turn_id or index}: diagnosis_revealing true with diagnostic_role '{diagnostic_role}'")

        hidden = milestone.get("hidden_reference")
        if hidden is not None and not isinstance(hidden, dict):
            errors.append(f"{turn_id or index}: hidden_reference must be an object")

        missing_tests = milestone.get("missing_or_unavailable_tests")
        if missing_tests is not None:
            if not isinstance(missing_tests, list):
                warnings.append(f"{turn_id or index}: missing_or_unavailable_tests is not a list")
            else:
                for test_index, item in enumerate(missing_tests):
                    if not isinstance(item, dict):
                        warnings.append(f"{turn_id or index}: missing_or_unavailable_tests[{test_index}] is not an object")
                        continue
                    status = normalize_whitespace(item.get("status"))
                    if status and status not in MISSING_TEST_STATUS_VALUES:
                        warnings.append(
                            f"{turn_id or index}: missing_or_unavailable_tests[{test_index}] unknown status '{status}'"
                        )
                    if not normalize_whitespace(item.get("test")):
                        warnings.append(f"{turn_id or index}: missing_or_unavailable_tests[{test_index}] missing test")

        if milestone.get("evidence_limitations") is not None and not isinstance(
            milestone.get("evidence_limitations"), list
        ):
            warnings.append(f"{turn_id or index}: evidence_limitations is not a list")

        validate_guardrail_flags(milestone.get("guardrail_flags"), turn_id or str(index), warnings)

        included_event_ids = milestone.get("included_event_ids")
        if included_event_ids is None:
            warnings.append(f"{turn_id or index}: missing included_event_ids")
        elif not isinstance(included_event_ids, list):
            warnings.append(f"{turn_id or index}: included_event_ids is not a list")
        elif event_ids:
            missing = [str(event_id) for event_id in included_event_ids if str(event_id) not in event_ids]
            if missing:
                warnings.append(f"{turn_id or index}: included_event_ids not found in evidence_events: {', '.join(missing)}")

        if not normalize_whitespace(milestone.get("grouping_reason")):
            warnings.append(f"{turn_id or index}: missing grouping_reason")

        has_image = bool(milestone.get("linked_figures") or milestone.get("image_paths") or milestone.get("image_metadata"))
        if has_image and not (
            normalize_whitespace(milestone.get("dialogue_visible_evidence"))
            or normalize_whitespace(milestone.get("test_visible_evidence"))
        ):
            warnings.append(f"{turn_id or index}: image milestone missing dialogue_visible_evidence/test_visible_evidence")

        if not normalize_whitespace(milestone.get("question")):
            warnings.append(f"{turn_id or index}: missing question")

        question_type = normalize_whitespace(milestone.get("question_type"))
        if not question_type:
            warnings.append(f"{turn_id or index}: missing question_type")
        elif question_type not in QUESTION_TYPE_VALUES:
            warnings.append(f"{turn_id or index}: unknown question_type '{question_type}'")

    return ValidationResult(ok=not errors, warnings=warnings, errors=errors)


def empty_token_usage_summary() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cached_tokens": 0,
        "num_successful_api_calls": 0,
    }


def add_usage(summary: dict[str, int], response: Any) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    if not isinstance(usage, dict):
        return
    summary["prompt_tokens"] += usage.get("prompt_tokens") or 0
    summary["completion_tokens"] += usage.get("completion_tokens") or 0
    summary["total_tokens"] += usage.get("total_tokens") or 0
    summary["num_successful_api_calls"] += 1
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    summary["reasoning_tokens"] += completion_details.get("reasoning_tokens") or 0
    summary["cached_tokens"] += prompt_details.get("cached_tokens") or 0
