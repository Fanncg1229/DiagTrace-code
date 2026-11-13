from __future__ import annotations

import json
import re
from typing import Any

from .common import azure_client, default_model, extract_json_object, token_usage
from .prompts import INFORMATION_OUTPUT_INSTRUCTIONS, INFORMATION_SYSTEM_PROMPT


RETURNABLE_POLICIES = {
    "return_if_requested",
    "return_if_specific_test_requested",
}


QUERY_CATEGORY_MAP = {
    "ASK_HISTORY": {"history", "symptom"},
    "REQUEST_EXAM": {"physical_exam"},
    "ORDER_TEST": {
        "routine_lab", "targeted_lab", "microbiology", "serology", "molecular", "genetic",
        "imaging_report", "pathology_report", "procedure",
    },
    "REQUEST_RESULT": {
        "imaging_report", "pathology_report", "procedure", "microbiology", "serology", "molecular", "genetic",
        "routine_lab", "targeted_lab", "treatment_response",
    },
    "REVIEW_IMAGE": {"raw_image", "raw_pathology_or_microscopy_image"},
    "WAIT_REASSESS": {"history", "symptom", "physical_exam", "treatment_response", "outcome_context"},
}

STOPWORDS = {
    "please", "provide", "result", "results", "current", "initial", "patient", "test", "tests",
    "finding", "findings", "report", "including", "any", "during", "with", "from", "what", "were",
    "blood", "serum", "plasma", "laboratory", "screening", "assess", "evaluate", "obtain",
    "anti", "antibody", "antibodies", "assay", "level", "levels", "activity", "confirm", "exclude",
    "and", "for", "order", "evaluation", "panel", "marker", "markers",
    "start", "date", "new", "other", "medication", "medications", "history", "timeline",
}

GENERIC_TARGETS = {"", "laboratory", "lab", "pathology", "imaging", "imaging report", "report", "history"}


def inferred_actions_for_item(item: dict[str, Any]) -> set[str]:
    category = str(item.get("category") or "")
    mode = str(item.get("acquisition_mode") or "")
    actions: set[str] = set()
    if category in {"history", "symptom"}:
        actions.add("ASK_HISTORY")
    if category == "physical_exam":
        actions.add("REQUEST_EXAM")
    if category in {
        "routine_lab",
        "targeted_lab",
        "microbiology",
        "serology",
        "molecular",
        "genetic",
        "imaging_report",
        "pathology_report",
        "procedure",
        "treatment_response",
        "outcome_context",
    }:
        actions.update({"ORDER_TEST", "REQUEST_RESULT"})
    if category in {"raw_image", "raw_pathology_or_microscopy_image"}:
        actions.add("REVIEW_IMAGE")
    if mode in {"spontaneous_clinical_event", "treatment_or_outcome_only"}:
        actions.add("WAIT_REASSESS")
    return actions or {"REQUEST_RESULT"}


def action_allowed_for_item(item: dict[str, Any], query_type: str) -> bool:
    allowed_actions = item.get("allowed_actions")
    if isinstance(allowed_actions, list) and allowed_actions:
        allowed = {str(value).strip().upper() for value in allowed_actions}
    else:
        allowed = inferred_actions_for_item(item)
    return query_type in allowed


def allowed_categories_for_query(query_type: str, question: str) -> set[str]:
    allowed = set(QUERY_CATEGORY_MAP.get(query_type, set()))
    text = question.lower()
    if query_type in {"REQUEST_RESULT", "ORDER_TEST", "REVIEW_IMAGE"}:
        if re.search(r"\b(ct|mri|pet|pet-ct|ultrasound|sonography|x-ray|radiograph|imaging)\b", text):
            if query_type in {"REQUEST_RESULT", "ORDER_TEST"}:
                allowed.intersection_update({"imaging_report"})
            else:
                allowed.intersection_update({"raw_image"})
        elif re.search(r"\b(biopsy|histolog|patholog|immunohist|cytolog)\w*\b", text):
            if query_type in {"REQUEST_RESULT", "ORDER_TEST"}:
                allowed.intersection_update({"pathology_report"})
            else:
                allowed.intersection_update({"raw_pathology_or_microscopy_image"})
        elif re.search(r"\b(gene|genetic|mutation|sequenc|molecular|ngs|fish|karyotype)\w*\b", text):
            allowed.intersection_update({"genetic", "molecular"})
    return allowed


MODALITY_PATTERNS = {
    "ct": r"\b(ct|computed tomography|pet[-/ ]?ct)\b",
    "mri": r"\b(mri|magnetic resonance)\b",
    "pet": r"\b(pet|pet[-/ ]?ct)\b",
    "ultrasound": r"\b(ultrasound|sonography|sonographic)\b",
    "xray": r"\b(x[- ]?ray|radiograph|radiography)\b",
}


def requested_modalities(text: str) -> set[str]:
    lowered = text.lower()
    modalities: set[str] = set()
    if re.search(r"\b(ct|computed tomography|pet-ct|pet/ct)\b", lowered):
        modalities.add("ct")
    if re.search(r"\b(mri|magnetic resonance)\b", lowered):
        modalities.add("mri")
    if re.search(r"\b(pet|pet-ct|pet/ct)\b", lowered):
        modalities.add("pet")
    if re.search(r"\b(ultrasound|sonography|sonographic)\b", lowered):
        modalities.add("ultrasound")
    if re.search(r"\b(x-ray|xray|radiograph|radiography)\b", lowered):
        modalities.add("xray")
    return modalities


def item_text(item: dict[str, Any]) -> str:
    return " ".join(
        [
            str(item.get("canonical_target") or ""),
            str(item.get("repeat_key") or ""),
            str(item.get("answer_text") or ""),
            str(item.get("time_anchor") or ""),
            str((item.get("image_details") or {}).get("image_type") or ""),
            str((item.get("image_details") or {}).get("report_finding") or ""),
        ]
    ).lower()


def modality_matches(item: dict[str, Any], modalities: set[str]) -> bool:
    if not modalities:
        return True
    text = item_text(item)
    for modality in modalities:
        if re.search(MODALITY_PATTERNS[modality], text):
            return True
    return False


def query_tokens(question: str, evidence_request: dict[str, Any]) -> set[str]:
    text = query_text(question, evidence_request)
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in STOPWORDS}


def query_text(question: str, evidence_request: dict[str, Any]) -> str:
    return " ".join(
        [
            question,
            str(evidence_request.get("target") or ""),
            " ".join(str(value) for value in evidence_request.get("specific_tests_or_findings") or []),
            str(evidence_request.get("specimen_or_anatomic_site") or ""),
        ]
    ).lower()


def evidence_match_score(item: dict[str, Any], tokens: set[str]) -> int:
    item_tokens = set(re.findall(r"[a-z0-9]+", item_text(item)))
    return len(tokens.intersection(item_tokens))


def specific_request_tokens(evidence_request: dict[str, Any]) -> set[str]:
    target = str(evidence_request.get("target") or "").strip().lower()
    values = list(evidence_request.get("specific_tests_or_findings") or [])
    if target not in GENERIC_TARGETS:
        values.append(target)
    text = " ".join(str(value) for value in values).lower()
    return {token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2 and token not in STOPWORDS}


def validated_routed_ids(
    case_data: dict[str, Any],
    matched_ids: list[Any],
    query_type: str,
    question: str,
    evidence_request: dict[str, Any],
) -> list[str]:
    by_id = {
        str(item.get("evidence_id")): item
        for item in case_data.get("evidence_items") or []
        if isinstance(item, dict) and item.get("evidence_id")
    }
    full_query_text = query_text(question, evidence_request)
    allowed_categories = allowed_categories_for_query(query_type, full_query_text)
    tokens = query_tokens(question, evidence_request)
    specific_tokens = specific_request_tokens(evidence_request)
    modalities = requested_modalities(full_query_text)
    validated: list[str] = []
    for raw_id in matched_ids:
        evidence_id = str(raw_id)
        item = by_id.get(evidence_id)
        if not item or item.get("disclosure_policy") not in RETURNABLE_POLICIES:
            continue
        if not action_allowed_for_item(item, query_type):
            continue
        if allowed_categories and item.get("category") not in allowed_categories:
            continue
        if not modality_matches(item, modalities):
            continue
        score = evidence_match_score(item, tokens)
        specific_score = evidence_match_score(item, specific_tokens) if specific_tokens else score
        if specific_tokens and specific_score < 1 and not modalities:
            continue
        if score < 1:
            continue
        validated.append(evidence_id)
    return list(dict.fromkeys(validated))


def deterministic_specific_ids(
    case_data: dict[str, Any],
    query_type: str,
    evidence_request: dict[str, Any],
) -> list[str]:
    tokens = specific_request_tokens(evidence_request)
    if not tokens:
        return []
    request_text = query_text("", evidence_request)
    allowed_categories = allowed_categories_for_query(query_type, request_text)
    modalities = requested_modalities(request_text)
    scored: list[tuple[int, int, str]] = []
    for item in case_data.get("evidence_items") or []:
        if not isinstance(item, dict) or item.get("disclosure_policy") not in RETURNABLE_POLICIES:
            continue
        if not action_allowed_for_item(item, query_type):
            continue
        if allowed_categories and item.get("category") not in allowed_categories:
            continue
        if not modality_matches(item, modalities):
            continue
        score = evidence_match_score(item, tokens)
        if score > 0:
            scored.append((score, int(item.get("phase_index", 0) or 0), str(item.get("evidence_id"))))
    if not scored:
        return []
    time_scope = str(evidence_request.get("time_scope") or "unspecified").lower()
    if time_scope in {"initial", "current"}:
        earliest_phase = min(phase for _, phase, _ in scored)
        scored = [(score, phase, evidence_id) for score, phase, evidence_id in scored if phase == earliest_phase]
    best_score = max(score for score, _, _ in scored)
    best = [(phase, evidence_id) for score, phase, evidence_id in scored if score == best_score]
    return [evidence_id for _, evidence_id in sorted(best)[:2]]


def deterministic_course_ids(case_data: dict[str, Any], current_phase_index: int) -> list[str]:
    candidates = [
        item
        for item in case_data.get("evidence_items") or []
        if isinstance(item, dict)
        and item.get("disclosure_policy") in RETURNABLE_POLICIES
        and action_allowed_for_item(item, "WAIT_REASSESS")
        and int(item.get("phase_index", 0) or 0) > current_phase_index
        and (
            item.get("acquisition_mode") == "spontaneous_clinical_event"
            or item.get("category") in {"history", "symptom", "physical_exam"}
        )
    ]
    if not candidates:
        return []
    next_phase = min(int(item.get("phase_index", 0) or 0) for item in candidates)
    return [
        str(item.get("evidence_id"))
        for item in candidates
        if int(item.get("phase_index", 0) or 0) == next_phase
    ][:3]


def evidence_public_view(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id", ""),
        "category": item.get("category", ""),
        "availability_status": item.get("availability_status", ""),
        "clinical_phase": item.get("clinical_phase", ""),
        "time_anchor": item.get("time_anchor", ""),
        "finding": item.get("answer_text", ""),
        "safe_image_metadata": {
            "image_type": (item.get("image_details") or {}).get("image_type", ""),
            "figure_ids": item.get("figure_ids") or [],
        },
        "figure_ids": item.get("figure_ids") or [],
    }


def build_wait_reassess_observation(
    case_data: dict[str, Any],
    disclosed_ids: set[str],
    pending_order_ids: set[str],
    current_phase_index: int,
) -> dict[str, Any]:
    returnable = [
        item
        for item in case_data.get("evidence_items") or []
        if isinstance(item, dict)
        and item.get("disclosure_policy") in RETURNABLE_POLICIES
        and item.get("evidence_id") not in disclosed_ids
        and (
            action_allowed_for_item(item, "WAIT_REASSESS")
            or str(item.get("evidence_id")) in pending_order_ids
        )
        and int(item.get("phase_index", 0) or 0) > current_phase_index
    ]
    if not returnable:
        return {
            "query_status": "unavailable",
            "message": "No additional source-grounded clinical event is available in the case record.",
            "phase_index_before": current_phase_index,
            "phase_index_after": current_phase_index,
            "evidence": [],
            "image_paths": [],
            "disclosed_evidence_ids": [],
        }

    next_phase = min(int(item.get("phase_index", 0) or 0) for item in returnable)
    phase_items = [item for item in returnable if int(item.get("phase_index", 0) or 0) == next_phase]
    wait_reassess_items = [
        item
        for item in phase_items
        if action_allowed_for_item(item, "WAIT_REASSESS")
    ]
    ordered = [item for item in phase_items if str(item.get("evidence_id")) in pending_order_ids]
    new_items = list({str(item.get("evidence_id")): item for item in wait_reassess_items + ordered}.values())
    for item in new_items:
        disclosed_ids.add(str(item.get("evidence_id")))
        pending_order_ids.discard(str(item.get("evidence_id")))
    return {
        "query_status": "answered" if new_items else "unavailable",
        "message": (
            "Source-grounded clinical course information for the next event is provided below."
            if new_items
            else "The replay advanced, but no source-grounded spontaneous finding or ordered result is available at this event."
        ),
        "phase_index_before": current_phase_index,
        "phase_index_after": next_phase,
        "evidence": [evidence_public_view(item) for item in new_items],
        "image_paths": [],
        "disclosed_evidence_ids": [item.get("evidence_id", "") for item in new_items],
    }


def build_order_observation(
    routing: dict[str, Any],
    pending_order_ids: set[str],
    current_phase_index: int,
) -> dict[str, Any]:
    matched_ids = [str(value) for value in routing.get("matched_evidence_ids") or [] if str(value)]
    pending_order_ids.update(matched_ids)
    return {
        "query_status": "ordered",
        "normalized_query_target": routing.get("normalized_query_target", ""),
        "message": "The order is recorded in the source-grounded replay. No result is released by the order action alone.",
        "phase_index_before": current_phase_index,
        "phase_index_after": current_phase_index,
        "evidence": [],
        "image_paths": [],
        "disclosed_evidence_ids": [],
    }


def phase_limited_items(
    items: list[dict[str, Any]],
    current_phase_index: int,
    query_type: str,
    time_scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not items:
        return [], []
    course_request = query_type == "WAIT_REASSESS"
    serial_scope = time_scope in {"latest", "trend", "repeat"}
    blocked: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for item in items:
        phase_index = int(item.get("phase_index", 0) or 0)
        mode = str(item.get("acquisition_mode") or "")
        future_only = mode in {"spontaneous_clinical_event", "treatment_or_outcome_only"}
        if time_scope == "initial" and phase_index > 1:
            blocked.append(item)
            continue
        if future_only and phase_index > current_phase_index and not course_request:
            blocked.append(item)
            continue
        if phase_index > current_phase_index + 1 and not (course_request or serial_scope):
            blocked.append(item)
            continue
        if (
            phase_index > current_phase_index + 1
            and serial_scope
            and not course_request
            and mode != "requestable_test"
            and mode != "requestable_report"
            and mode != "requestable_raw_image"
        ):
            blocked.append(item)
            continue
        candidates.append(item)
    if not candidates:
        return [], blocked
    target_phase = min(int(item.get("phase_index", 0) or 0) for item in candidates)
    allowed = [item for item in candidates if int(item.get("phase_index", 0) or 0) == target_phase]
    blocked.extend(item for item in candidates if int(item.get("phase_index", 0) or 0) != target_phase)
    return allowed, blocked


def router_inventory(
    case_data: dict[str, Any],
    query_type: str,
    question: str,
    evidence_request: dict[str, Any],
    current_phase_index: int,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    tokens = query_tokens(question, evidence_request)
    full_query_text = query_text(question, evidence_request)
    allowed_categories = allowed_categories_for_query(query_type, full_query_text)
    modalities = requested_modalities(full_query_text)
    inventory = []
    for item in case_data.get("evidence_items") or []:
        if not isinstance(item, dict) or item.get("disclosure_policy") not in RETURNABLE_POLICIES:
            continue
        if not action_allowed_for_item(item, query_type):
            continue
        if allowed_categories and item.get("category") not in allowed_categories:
            continue
        if not modality_matches(item, modalities):
            continue
        score = evidence_match_score(item, tokens)
        inventory.append(
            {
                "evidence_id": item.get("evidence_id", ""),
                "category": item.get("category", ""),
                "clinical_phase": item.get("clinical_phase", ""),
                "phase_index": item.get("phase_index", 0),
                "time_anchor": item.get("time_anchor", ""),
                "acquisition_mode": item.get("acquisition_mode", ""),
                "repeat_key": item.get("repeat_key", ""),
                "canonical_target": item.get("canonical_target", ""),
                "availability_status": item.get("availability_status", ""),
                "disclosure_policy": item.get("disclosure_policy", ""),
                "answer_text": str(item.get("answer_text", ""))[:300],
                "figure_ids": item.get("figure_ids") or [],
                "has_raw_image": bool(item.get("image_paths")),
                "_match_score": score,
            }
        )
    inventory.sort(
        key=lambda value: (
            -int(value.get("_match_score", 0)),
            abs(int(value.get("phase_index", 0) or 0) - current_phase_index),
            str(value.get("evidence_id", "")),
        )
    )
    selected = inventory[:max_items]
    for item in selected:
        item.pop("_match_score", None)
    return selected


def build_observation(
    case_data: dict[str, Any],
    routing: dict[str, Any],
    disclosed_ids: set[str],
    current_phase_index: int = 0,
    query_type: str = "",
    evidence_request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = {
        item.get("evidence_id"): item
        for item in case_data.get("evidence_items") or []
        if isinstance(item, dict) and item.get("disclosure_policy") in RETURNABLE_POLICIES
    }
    requested_ids = routing.get("matched_evidence_ids")
    if not isinstance(requested_ids, list):
        requested_ids = []
    approved = [
        by_id[value]
        for value in requested_ids
        if value in by_id and action_allowed_for_item(by_id[value], query_type)
    ]
    request = evidence_request if isinstance(evidence_request, dict) else {}
    time_scope = str(request.get("time_scope") or "unspecified").lower()
    serial_scope = time_scope in {"latest", "trend", "repeat"}

    # For a generic request for a serial test, return the earliest matching
    # result at the current state rather than a more diagnostic later result.
    if not serial_scope:
        replacements: list[dict[str, Any]] = []
        for item in approved:
            repeat_key = str(item.get("repeat_key") or "")
            if not repeat_key:
                replacements.append(item)
                continue
            serial_items = [
                candidate
                for candidate in by_id.values()
                if candidate.get("repeat_key") == repeat_key
                and action_allowed_for_item(candidate, query_type)
                and int(candidate.get("phase_index", 0) or 0) >= current_phase_index
            ]
            if serial_items:
                replacements.append(min(serial_items, key=lambda value: int(value.get("phase_index", 0) or 0)))
            else:
                replacements.append(item)
        approved = list({item.get("evidence_id"): item for item in replacements}.values())

    approved, blocked = phase_limited_items(approved, current_phase_index, query_type, time_scope)
    new_items = [item for item in approved if item.get("evidence_id") not in disclosed_ids]

    if approved and not new_items:
        status = "already_disclosed"
    elif new_items:
        statuses = {item.get("availability_status") for item in new_items}
        if statuses == {"not_performed"}:
            status = "not_performed"
        elif statuses == {"unavailable"}:
            status = "unavailable"
        elif statuses == {"inconclusive"}:
            status = "inconclusive"
        elif routing.get("query_status") == "partially_answered" or blocked:
            status = "partially_answered"
        else:
            status = "answered" if len(new_items) == len(approved) else "partially_answered"
    else:
        status = routing.get("query_status")
        if blocked:
            status = "not_yet_available"
        elif status not in {"not_reported", "irrelevant", "ambiguous", "already_disclosed"}:
            status = "not_reported"

    evidence = []
    image_paths: list[str] = []
    for item in new_items:
        evidence.append(evidence_public_view(item))
        image_paths.extend(item.get("image_paths") or [])
        disclosed_ids.add(str(item.get("evidence_id")))

    if status == "not_reported":
        message = "The requested information is not reported in the available case record. This is not a negative result."
    elif status == "already_disclosed":
        message = "The directly matching information has already been disclosed."
    elif status == "irrelevant":
        message = "The request does not map to diagnostically relevant information in the available case record."
    elif status == "ambiguous":
        message = "The request is too ambiguous to map safely to one source-supported evidence item."
    elif status == "not_yet_available":
        status = "unavailable"
        message = "No source-grounded answer is available from the currently accessible replay state. This is not a negative clinical result."
    elif status == "partially_answered":
        message = "The available source-grounded portion of the request is provided; other requested information was not reported or is not yet available."
    else:
        message = "Source-grounded information matching the request is provided below."

    next_phase_index = max(
        [current_phase_index] + [int(item.get("phase_index", 0) or 0) for item in new_items]
    )

    result = {
        "query_status": status,
        "normalized_query_target": routing.get("normalized_query_target", ""),
        "message": message,
        "phase_index_before": current_phase_index,
        "phase_index_after": next_phase_index,
        "evidence": evidence,
        "image_paths": list(dict.fromkeys(image_paths)),
        "disclosed_evidence_ids": [item.get("evidence_id", "") for item in new_items],
    }
    return result


class InformationAgent:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "low",
        max_tokens: int = 1500,
        timeout: int = 600,
    ) -> None:
        self.client = azure_client(timeout)
        self.model = model or default_model()
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    async def answer(
        self,
        case_data: dict[str, Any],
        question: str,
        query_type: str,
        evidence_request: dict[str, Any],
        disclosed_ids: set[str],
        current_phase_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
        routing, usage = await self.route(
            case_data=case_data,
            question=question,
            query_type=query_type,
            evidence_request=evidence_request,
            disclosed_ids=disclosed_ids,
            current_phase_index=current_phase_index,
        )
        if query_type == "WAIT_REASSESS":
            observation = build_wait_reassess_observation(
                case_data,
                disclosed_ids,
                set(),
                current_phase_index,
            )
        else:
            observation = build_observation(
                case_data,
                routing,
                disclosed_ids,
                current_phase_index=current_phase_index,
                query_type=query_type,
                evidence_request=evidence_request,
            )
        return observation, routing, usage

    async def route(
        self,
        case_data: dict[str, Any],
        question: str,
        query_type: str,
        evidence_request: dict[str, Any],
        disclosed_ids: set[str],
        current_phase_index: int,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        prompt = (
            f"Clinician question:\n{question}\n\n"
            f"Clinician action:\n{query_type}\n\n"
            f"Structured evidence request:\n{json.dumps(evidence_request, ensure_ascii=False, indent=2)}\n\n"
            f"Current clinical phase index:\n{current_phase_index}\n\n"
            f"Previously disclosed evidence IDs:\n{json.dumps(sorted(disclosed_ids), ensure_ascii=False)}\n\n"
            "Returnable evidence inventory:\n"
            f"{json.dumps(router_inventory(case_data, query_type, question, evidence_request, current_phase_index), ensure_ascii=False, indent=2)}\n\n"
            f"{INFORMATION_OUTPUT_INSTRUCTIONS}"
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": INFORMATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            response_format={"type": "json_object"},
        )
        raw = (response.choices[0].message.content or "").strip()
        routing = extract_json_object(raw)
        matched = routing.get("matched_evidence_ids")
        if not isinstance(matched, list):
            matched = []
        routing["matched_evidence_ids"] = validated_routed_ids(
            case_data, matched, query_type, question, evidence_request
        )
        if query_type == "WAIT_REASSESS":
            course_ids = deterministic_course_ids(case_data, current_phase_index)
            if course_ids:
                routing["matched_evidence_ids"] = course_ids
                routing["query_status"] = "answered"
                routing["brief_rationale"] = "deterministic next clinical-event transition"
        else:
            specific_ids = deterministic_specific_ids(case_data, query_type, evidence_request)
            if specific_ids:
                routing["matched_evidence_ids"] = specific_ids
                routing["query_status"] = "answered"
                routing["brief_rationale"] = "deterministic specific-test routing"
        if not routing["matched_evidence_ids"]:
            candidates = router_inventory(
                case_data, query_type, question, evidence_request, current_phase_index, max_items=3
            )
            tokens = specific_request_tokens(evidence_request) or query_tokens(question, evidence_request)
            scored = [
                (evidence_match_score(item, tokens), item.get("evidence_id"))
                for item in case_data.get("evidence_items") or []
                if isinstance(item, dict)
                and item.get("evidence_id") in {candidate.get("evidence_id") for candidate in candidates}
                and action_allowed_for_item(item, query_type)
                and modality_matches(item, requested_modalities(query_text(question, evidence_request)))
            ]
            if scored and max(score for score, _ in scored) >= 2:
                best_score = max(score for score, _ in scored)
                routing["matched_evidence_ids"] = [
                    evidence_id for score, evidence_id in scored if score == best_score and evidence_id
                ][:2]
                routing["query_status"] = "answered"
                routing["brief_rationale"] = "deterministic lexical fallback after empty model routing"
        return routing, token_usage(response)
