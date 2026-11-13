from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI


load_dotenv(find_dotenv(usecwd=True))


VALID_CLINICIAN_ACTIONS = frozenset(
    {
        "ASK_HISTORY",
        "REQUEST_EXAM",
        "ORDER_TEST",
        "REQUEST_RESULT",
        "ORDER_PROCEDURE",
        "WAIT_REASSESS",
        "FINAL",
    }
)

PHASE_LABELS = (
    "initial_presentation",
    "early_workup",
    "targeted_workup",
    "clinical_progression",
    "advanced_workup",
    "confirmatory_testing",
    "treatment_or_outcome",
)


def normalize_evidence_category(raw: dict[str, Any]) -> str:
    category = str(raw.get("category") or raw.get("evidence_type") or "other").lower()
    aliases = {
        "lab": "routine_lab",
        "laboratory": "routine_lab",
        "lab_result": "routine_lab",
        "laboratory_result": "routine_lab",
        "imaging": "imaging_report",
        "radiology": "imaging_report",
        "pathology": "pathology_report",
        "molecular": "molecular_test",
        "genetic": "molecular_test",
        "exam": "physical_exam",
        "course": "clinical_course",
        "treatment": "treatment_response",
    }
    category = aliases.get(category, category)
    known = {
        "history",
        "symptom",
        "physical_exam",
        "routine_lab",
        "targeted_lab",
        "microbiology",
        "serology",
        "molecular_test",
        "imaging_report",
        "pathology_report",
        "procedure",
        "clinical_course",
        "treatment_response",
        "image_finding",
    }
    if category in known:
        return category
    if raw.get("image_finding") or raw.get("is_image_finding"):
        return "image_finding"
    searchable = " ".join(
        str(raw.get(field) or "")
        for field in ("repeat_key", "source_text", "source_quote", "source_span", "answer_text")
    ).lower()
    if any(word in searchable for word in ("biopsy", "histolog", "bone marrow", "cytolog", "immunohist")):
        return "pathology_report"
    if any(
        word in searchable
        for word in (" ct ", "mri", "ultrasound", "x-ray", "xray", "pet scan", "ceus", "fibroscan", "elastography", "echocardi", "doppler")
    ):
        return "imaging_report"
    if any(word in searchable for word in ("mutation", "genetic", "sequencing", "ngs", "jak-2", "uba1")):
        return "molecular_test"
    if any(word in searchable for word in ("antibod", "serolog", "immunoglobulin")):
        return "serology"
    if any(
        word in searchable
        for word in ("hemoglobin", "platelet", "leukocyte", "cbc", "crp", "bilirubin", "glucose", "insulin", "laboratory", "blood analysis")
    ):
        return "routine_lab"
    return "other"


CLINICIAN_SYSTEM_PROMPT = """
You are the clinician being evaluated in an interactive medical diagnosis benchmark.
Return valid JSON only.

You initially receive an incomplete presentation. On each turn, ask one focused,
clinically natural question that would best narrow your differential diagnosis, or
submit a final diagnosis when you are sufficiently confident.

Valid actions:
- ASK_HISTORY: ask about symptoms, exposures, medications, or past history
- REQUEST_EXAM: ask for physical examination or bedside findings
- ORDER_TEST: order one laboratory, imaging, microbiology, or molecular test
- REQUEST_RESULT: request one available laboratory, imaging-report, pathology,
  molecular, microbiology, or source-extracted image finding
- ORDER_PROCEDURE: order one diagnostic procedure such as biopsy or endoscopy
- WAIT_REASSESS: wait for a pending result or ask for the next source-grounded clinical
  course update, such as hospitalization, ICU transfer, surgery/procedure, new symptoms,
  complications, or treatment response/failure
- FINAL: submit one final diagnosis and a concise evidence-based explanation

Rules:
- Ask only one focused question per turn.
- Use natural language as a clinician would.
- Maintain a short ranked differential diagnosis and confidence after each observation.
- Do not request raw image pixels; only textual findings are available.
- You see the complete prior dialogue, including your earlier DDx states and every report reply.
- If the diagnostic context is stalled or the patient's course may have evolved, use
  WAIT_REASSESS to obtain the next documented clinical update rather than repeatedly
  ordering unavailable tests.
- If you receive repeated not_reported, ordered_but_not_documented, too_broad, or
  invalid_request replies, use WAIT_REASSESS to learn whether the documented clinical
  course evolved before asking more unavailable tests.
- Do not submit FINAL immediately after one local or intermediate diagnosis, such as a
  biopsy diagnosis, if the longitudinal course, recurrence, systemic involvement,
  treatment response, or complications remain unknown. In that situation, use
  WAIT_REASSESS to check for later source-grounded clinical updates.
- FINAL ends the episode and receives no correctness feedback.
- On the final available turn you must submit FINAL using your current most likely diagnosis.

For an information request, return:
{
  "action": "one valid non-final action",
  "request": "one focused clinical question",
  "current_ddx": ["ranked diagnosis", "..."],
  "confidence": 0.0,
  "brief_rationale": "one short sentence"
}

For the final answer, return:
{
  "action": "FINAL",
  "diagnosis": "one diagnosis",
  "current_ddx": ["ranked diagnosis", "..."],
  "confidence": 0.0,
  "supporting_evidence_ids": ["IDs from disclosed report replies"],
  "reasoning": "concise explanation grounded only in disclosed evidence"
}
""".strip()


REPORT_SYSTEM_PROMPT = """
You are the Report Agent in a source-grounded medical information-seeking benchmark.
Return valid JSON only.

You simulate a longitudinal hospital information and reporting platform. You see all
source-extracted PMC evidence, but the clinician sees only evidence that you release.
You decide the current clinical phase, whether an ordered test is pending, and which
timepoint directly answers the request. The application validates and returns either
the selected record's complete source text or a contiguous verbatim excerpt that you
select. You must not write or paraphrase the medical answer yourself.

Rules:
1. Select only from the supplied evidence records.
2. Do not diagnose, interpret significance, recommend a next step, or infer missing facts.
3. Return at most one evidence_id. Never reveal an ID or medical content in explanation.
4. Keep phase_index_after monotonic. ORDER_TEST, ORDER_PROCEDURE, pending, and
   ordered_but_not_documented never advance the phase. A phase advances only when you return
   found for a later record; set phase_index_after to that record's phase.
5. When ORDER_TEST or ORDER_PROCEDURE matches a source record, return ordered and include
   that evidence_id internally. A later matching REQUEST_RESULT or WAIT_REASSESS must return
   found for that source-backed result; do not keep an already ordered source-backed result
   pending again. If the source has no result for the order, return
   ordered_but_not_documented without an evidence_id; this means the article never documents
   the result, not that the result is pending. Do not imply that waiting will eventually
   produce a result. Do not invent turnaround time.
6. REQUEST_RESULT does not require a prior order. If a directly responsive source record
   exists, you may return it and advance to its phase. WAIT_REASSESS should release the
   earliest pending result first. If no pending result exists, release the earliest
   undisclosed source-grounded clinical course update at or after the current phase.
   Clinical course updates include hospitalization, ICU transfer, surgery/procedure,
   new symptoms, new organ involvement, thrombosis, complications, treatment response,
   treatment failure, or other timeline events the clinician would normally know.
7. For a vague request about a repeated measurement, select the earliest undisclosed
   relevant record available in the current phase. Later requests may reveal later records
   after the clinical phase has advanced.
8. If the clinician explicitly names a future timepoint, select that record only if the
   phase is available; otherwise you may mark it pending once, with that future record's
   evidence_id and a source-grounded prerequisite/timepoint in explanation. The next matching
   result request or WAIT_REASSESS must release the result if it is source-backed.
9. A question about a CT image finding must be answered only from an extracted radiology
   report or textual image finding. You cannot inspect image pixels.
10. If any supplied evidence record is clinically related to a reasonable request,
    return one related source-backed record. Your job is only to map the request to
    related article evidence, not to infer or synthesize an answer. Do not return
    not_reported merely because the clinician used a different test name, asked for a
    panel, or requested more detail than the source contains. Prefer the earliest
    available related record and let the verbatim source text define the answer.
11. Absence from the source means not_reported, never a negative clinical result.
12. If the request contains multiple independent questions, but one component has a
    related source-backed record, return that one component. Return too_broad only when no
    single related source record can be selected without guessing the clinician's
    priority.
13. Do not expose article metadata, retrospective interpretation, author narration of the
    diagnostic process, final-diagnosis prose, or evidence not responsive to the request.
14. A record marked requires_safe_excerpt mixes usable clinical evidence with diagnosis-
    revealing narration. Select it only when its contemporaneous test/report result directly
    answers the request, and return a strict contiguous verbatim_excerpt containing only the
    result (measurements, positive/negative result, mutation, morphology, imaging or pathology
    finding). Exclude clauses such as "thus establishing the diagnosis". Never return the
    complete text of such a record. A formal contemporaneous pathology or radiology report
    result may be returned; retrospective author interpretation may not.
15. Legal clinical results include laboratory values, imaging report findings, pathology
    morphology, microbiology findings, and genetic variants. Diagnostic-process text such as
    "this confirmed", "diagnosed as", "consistent with the final diagnosis", or the authors'
    causal reasoning must be excluded even when it appears in the same source record.

Return exactly one of:
{
  "status": "found | pending | ordered | ordered_but_not_documented | not_reported | too_broad | invalid_request | no_new_update",
  "evidence_id": "one ID only when grounded in a source record, otherwise empty",
  "verbatim_excerpt": "optional contiguous source substring; required for requires_safe_excerpt",
  "phase_index_after": 0,
  "phase_label_after": "one supplied phase label",
  "pending_order": "concise ordered test/procedure name or empty",
  "explanation": "operational status only; no medical content"
}
""".strip()


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    category: str
    source_text: str
    temporal_cue: str = ""
    timeline_order: int = 0
    phase_index: int = 0
    phase_label: str = "initial_presentation"
    repeat_key: str = ""
    source_section: str = ""
    image_finding: bool = False
    diagnosis_revealing: bool = False

    @classmethod
    def from_raw(cls, raw: dict[str, Any], fallback_order: int) -> "EvidenceItem":
        source_text = (
            raw.get("source_text")
            or raw.get("source_quote")
            or raw.get("source_span")
            or raw.get("answer_text")
            or ""
        )
        order = raw.get("timeline_order")
        if order is None:
            order = raw.get("source_order")
        if order is None:
            order = raw.get("phase_index")
        if order is None:
            order = fallback_order
        phase_index = int(raw.get("phase_index") or 0)
        if phase_index < 0 or phase_index >= len(PHASE_LABELS):
            phase_index = 0
        return cls(
            id=str(raw.get("id") or raw.get("evidence_id") or f"E{fallback_order}"),
            category=normalize_evidence_category(raw),
            source_text=str(source_text),
            temporal_cue=str(raw.get("temporal_cue") or raw.get("time_anchor") or ""),
            timeline_order=int(order),
            phase_index=phase_index,
            phase_label=str(
                raw.get("phase_label")
                or raw.get("clinical_phase")
                or PHASE_LABELS[phase_index]
            ),
            repeat_key=str(raw.get("repeat_key") or ""),
            source_section=str(raw.get("source_section") or ""),
            image_finding=bool(raw.get("image_finding") or raw.get("is_image_finding")),
            diagnosis_revealing=bool(
                raw.get("diagnosis_revealing")
                or raw.get("contains_final_diagnosis")
                or raw.get("withhold_from_dialogue")
            ),
        )


@dataclass(frozen=True)
class DialogueTurn:
    turn_number: int
    clinician: dict[str, Any]
    report: dict[str, Any] | None


def _build_client() -> AzureOpenAI:
    required = {
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return AzureOpenAI(
        api_key=required["AZURE_OPENAI_API_KEY"],
        azure_endpoint=required["AZURE_OPENAI_ENDPOINT"],
        api_version=required["AZURE_OPENAI_API_VERSION"],
        timeout=600,
    )


def _call_json_model(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    usage_total: dict[str, int] | None = None,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = getattr(response.choices[0].message, "content", "{}") or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response was not a JSON object")
    if usage_total is not None:
        usage = getattr(response, "usage", None)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage_total[key] = usage_total.get(key, 0) + int(getattr(usage, key, 0) or 0)
    return parsed


class ReportAgent:
    """LLM evidence selector whose public answers are exact stored source text."""

    def __init__(self, evidence_items: list[EvidenceItem | dict[str, Any]], client: Any, model: str) -> None:
        normalized = [
            item if isinstance(item, EvidenceItem) else EvidenceItem.from_raw(item, index)
            for index, item in enumerate(evidence_items)
        ]
        self.evidence_items = sorted(normalized, key=lambda item: (item.timeline_order, item.id))
        self.evidence_by_id = {item.id: item for item in self.evidence_items if item.source_text}
        self.client = client
        self.model = model
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.current_phase_index = 0
        self.pending_orders: list[dict[str, str]] = []

    @staticmethod
    def _requires_safe_excerpt(item: EvidenceItem) -> bool:
        return item.diagnosis_revealing

    def answer(
        self,
        action: str,
        request: str,
        disclosed_ids: set[str] | None = None,
        *,
        _retry_reason: str = "",
        _can_retry: bool = True,
        _repair_errors: list[str] | None = None,
    ) -> dict[str, Any]:
        disclosed_ids = set(disclosed_ids or set())
        repair_errors = list(_repair_errors or [])
        records = [
            {
                "evidence_id": item.id,
                "category": item.category,
                "temporal_cue": item.temporal_cue,
                "timeline_order": item.timeline_order,
                "phase_index": item.phase_index,
                "phase_label": item.phase_label,
                "repeat_key": item.repeat_key,
                "source_section": item.source_section,
                "image_finding": item.image_finding,
                "source_text": item.source_text,
                "requires_safe_excerpt": self._requires_safe_excerpt(item),
                "already_disclosed": item.id in disclosed_ids,
            }
            for item in self.evidence_items
            if item.id in self.evidence_by_id
        ]
        prompt = (
            f"Clinician action: {action}\n"
            f"Clinician request: {request}\n"
            f"Current phase index: {self.current_phase_index}\n"
            f"Current phase label: {PHASE_LABELS[self.current_phase_index]}\n"
            f"Pending orders: {json.dumps(self.pending_orders, ensure_ascii=False)}\n"
            f"Previously disclosed evidence IDs: {sorted(disclosed_ids)}\n\n"
            f"Evidence records in chronological order:\n{json.dumps(records, ensure_ascii=False, indent=2)}"
        )
        if _retry_reason:
            prompt += (
                "\n\nYour previous output was rejected by the protocol validator: "
                f"{_retry_reason}\n"
                "Return one corrected JSON response. You may select the same evidence with a "
                "strict contiguous verbatim_excerpt, select another valid source record, or return "
                "a legal unavailable status such as not_reported or ordered_but_not_documented when "
                "the source does not support a valid answer. Do not fabricate or combine excerpts."
            )
        selected = _call_json_model(
            self.client,
            self.model,
            REPORT_SYSTEM_PROMPT,
            prompt,
            self.usage,
        )
        status = str(selected.get("status") or "invalid_request").lower()
        if status == "ordered_no_source_result":
            status = "ordered_but_not_documented"
        evidence_id = str(selected.get("evidence_id") or "")
        requested_phase = selected.get("phase_index_after", self.current_phase_index)
        try:
            phase_index_after = int(requested_phase)
        except (TypeError, ValueError):
            phase_index_after = self.current_phase_index
        phase_index_after = min(max(phase_index_after, self.current_phase_index), len(PHASE_LABELS) - 1)
        pending_order = str(selected.get("pending_order") or "").strip()
        explanation = str(selected.get("explanation") or "").strip()

        if action in {"ORDER_TEST", "ORDER_PROCEDURE"} and status == "pending":
            status = "ordered"
        if action in {"ORDER_TEST", "ORDER_PROCEDURE"} and status == "ordered" and not evidence_id:
            status = "ordered_but_not_documented"
        if action == "REQUEST_RESULT" and status == "ordered":
            status = "pending"
        if action == "REQUEST_RESULT" and status == "pending" and not evidence_id:
            status = "not_reported"
        if action == "WAIT_REASSESS" and status == "pending" and not evidence_id:
            status = "no_new_update"

        def finish(
            payload: dict[str, Any],
            *,
            validation_passed: bool = True,
            final_validation_error: str = "",
        ) -> dict[str, Any]:
            if repair_errors:
                errors = list(repair_errors)
                if final_validation_error:
                    errors.append(final_validation_error)
                payload = dict(payload)
                payload.update(
                    {
                        "repair_attempted": True,
                        "repair_count": len(repair_errors),
                        "original_validation_error": repair_errors[0],
                        "validation_errors": errors,
                        "final_validation_passed": validation_passed,
                        "repair_failed": not validation_passed,
                    }
                )
            return payload

        def invalid_selection(reason: str) -> dict[str, Any]:
            if _can_retry:
                return self.answer(
                    action,
                    request,
                    disclosed_ids,
                    _retry_reason=reason,
                    _can_retry=False,
                    _repair_errors=repair_errors + [reason],
                )
            return finish(
                {
                    "status": "invalid_report_selection",
                    "evidence_id": "",
                    "source_text": "",
                    "phase_index": self.current_phase_index,
                    "phase_label": PHASE_LABELS[self.current_phase_index],
                    "pending_orders": [item["request"] for item in self.pending_orders],
                    "explanation": reason,
                    "verbatim": True,
                },
                validation_passed=False,
                final_validation_error=reason,
            )

        allowed_statuses = {
            "ASK_HISTORY": {"found", "not_reported", "too_broad", "invalid_request"},
            "REQUEST_EXAM": {"found", "not_reported", "too_broad", "invalid_request"},
            "ORDER_TEST": {"ordered", "ordered_but_not_documented", "too_broad", "invalid_request"},
            "ORDER_PROCEDURE": {"ordered", "ordered_but_not_documented", "too_broad", "invalid_request"},
            "REQUEST_RESULT": {"found", "pending", "not_reported", "too_broad", "invalid_request"},
            "WAIT_REASSESS": {
                "found",
                "pending",
                "not_reported",
                "no_new_update",
                "invalid_request",
            },
        }
        if status not in allowed_statuses.get(action, set()):
            return invalid_selection(f"Status {status} is not valid for clinician action {action}.")

        if status == "ordered_but_not_documented":
            return finish(
                {
                    "status": status,
                    "evidence_id": "",
                    "source_text": "",
                    "phase_index": self.current_phase_index,
                    "phase_label": PHASE_LABELS[self.current_phase_index],
                    "pending_orders": [item["request"] for item in self.pending_orders],
                    "explanation": explanation,
                    "verbatim": True,
                }
            )

        if status in {"pending", "ordered"}:
            if evidence_id not in self.evidence_by_id:
                return invalid_selection("A source-backed pending order requires a valid evidence ID.")
            if any(entry.get("evidence_id") == evidence_id for entry in self.pending_orders):
                return invalid_selection(
                    "A source-backed order/result is already pending; the next matching result request "
                    "or reassessment must return found rather than pending again."
                )
            if not pending_order:
                pending_order = request
            pending = {"request": pending_order, "evidence_id": evidence_id}
            if pending not in self.pending_orders:
                self.pending_orders.append(pending)
            return finish(
                {
                    "status": status,
                    "evidence_id": "",
                    "source_text": "",
                    "phase_index": self.current_phase_index,
                    "phase_label": PHASE_LABELS[self.current_phase_index],
                    "pending_orders": [item["request"] for item in self.pending_orders],
                    "explanation": explanation,
                    "verbatim": True,
                }
            )

        if status != "found":
            if status == "no_new_update" and self.pending_orders:
                return invalid_selection(
                    "A pending source-backed result exists; WAIT_REASSESS must release it or select a "
                    "different source-grounded update."
                )
            if status not in {"not_reported", "too_broad", "invalid_request", "no_new_update"}:
                status = "invalid_request"
            return finish(
                {
                    "status": status,
                    "evidence_id": "",
                    "source_text": "",
                    "phase_index": self.current_phase_index,
                    "phase_label": PHASE_LABELS[self.current_phase_index],
                    "pending_orders": [item["request"] for item in self.pending_orders],
                    "explanation": explanation,
                    "verbatim": True,
                }
            )

        item = self.evidence_by_id.get(evidence_id)
        if item is None:
            return invalid_selection("found requires a valid evidence ID.")
        expected_phase = max(self.current_phase_index, item.phase_index)
        if phase_index_after != expected_phase:
            return invalid_selection("phase_index_after must equal the selected evidence phase.")
        excerpt = str(selected.get("verbatim_excerpt") or "").strip()
        source_text = item.source_text
        if excerpt:
            normalized_excerpt = " ".join(excerpt.split())
            normalized_source = " ".join(item.source_text.split())
            if normalized_excerpt not in normalized_source:
                return invalid_selection("verbatim_excerpt is not a contiguous source subspan.")
            source_text = excerpt
        if self._requires_safe_excerpt(item) and (
            not excerpt or normalized_excerpt == " ".join(item.source_text.split())
        ):
            return invalid_selection("A diagnosis-revealing record requires a strict safe excerpt.")
        self.current_phase_index = expected_phase
        self.pending_orders = [
            entry for entry in self.pending_orders if entry.get("evidence_id") != item.id
        ]
        if item.id in disclosed_ids:
            return finish(
                {
                    "status": "already_disclosed",
                    "evidence_id": item.id,
                    "source_text": source_text,
                    "category": item.category,
                    "temporal_cue": item.temporal_cue,
                    "phase_index": self.current_phase_index,
                    "phase_label": PHASE_LABELS[self.current_phase_index],
                    "pending_orders": [entry["request"] for entry in self.pending_orders],
                    "verbatim": True,
                }
            )
        return finish(
            {
                "status": "found",
                "evidence_id": item.id,
                "source_text": source_text,
                "category": item.category,
                "temporal_cue": item.temporal_cue,
                "phase_index": self.current_phase_index,
                "phase_label": PHASE_LABELS[self.current_phase_index],
                "pending_orders": [entry["request"] for entry in self.pending_orders],
                "verbatim": True,
            }
        )


class DialogueWorkflow:
    """Thin episode loop for a clinician LLM talking to a source-grounded report LLM."""

    def __init__(
        self,
        case_data: dict[str, Any],
        clinician_client: Any | None = None,
        report_client: Any | None = None,
        clinician_model: str | None = None,
        report_model: str | None = None,
        max_turns: int = 25,
        resume_episode: dict[str, Any] | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.case_data = case_data
        self.clinician_client = clinician_client or _build_client()
        self.report_client = report_client or _build_client()
        self.clinician_model = (
            clinician_model
            or os.getenv("AZURE_OPENAI_CLINICIAN_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or "o4-mini"
        )
        self.report_model = (
            report_model
            or os.getenv("AZURE_OPENAI_REPORT_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT")
            or "o4-mini"
        )
        self.max_turns = max_turns
        self.initial_observation = case_data.get("initial_observation", {})
        self.report_agent = ReportAgent(
            evidence_items=case_data.get("evidence_items") or case_data.get("evidence_events") or [],
            client=self.report_client,
            model=self.report_model,
        )
        self.disclosed_ids: set[str] = set()
        self.turns: list[DialogueTurn] = []
        self.clinician_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if resume_episode:
            self._restore_episode(resume_episode)

    def _restore_episode(self, episode: dict[str, Any]) -> None:
        if episode.get("case_id") and episode.get("case_id") != self.case_data.get("case_id"):
            raise ValueError("Resume episode case_id does not match case_data")
        for raw_turn in episode.get("turns", []):
            if not isinstance(raw_turn, dict):
                continue
            clinician = raw_turn.get("clinician") or {}
            report = raw_turn.get("report")
            self.turns.append(
                DialogueTurn(
                    turn_number=int(raw_turn.get("turn_number") or len(self.turns) + 1),
                    clinician=dict(clinician),
                    report=dict(report) if isinstance(report, dict) else None,
                )
            )
        self.disclosed_ids = set(episode.get("disclosed_evidence_ids") or [])
        model_usage = episode.get("model_usage") or {}
        self.clinician_usage.update(model_usage.get("clinician") or {})
        self.report_agent.usage.update(model_usage.get("report") or {})
        reports = [turn.report for turn in self.turns if turn.report]
        if reports:
            last_report = reports[-1]
            phase = last_report.get("phase_index")
            if isinstance(phase, int) and 0 <= phase < len(PHASE_LABELS):
                self.report_agent.current_phase_index = phase
            self.report_agent.pending_orders = [
                {"request": str(request), "evidence_id": ""}
                for request in last_report.get("pending_orders", [])
                if request
            ]

    def _clinician_prompt(self, turn_number: int) -> str:
        dialogue = []
        for turn in self.turns:
            dialogue.append({"turn": turn.turn_number, "clinician": turn.clinician, "report": turn.report})
        final_instruction = (
            "This is the final available turn. You must submit FINAL."
            if turn_number == self.max_turns
            else "Ask one focused question or submit FINAL if sufficiently confident."
        )
        return (
            f"Initial presentation:\n{json.dumps(self.initial_observation, ensure_ascii=False, indent=2)}\n\n"
            f"Dialogue so far:\n{json.dumps(dialogue, ensure_ascii=False, indent=2)}\n\n"
            f"Turn: {turn_number}/{self.max_turns}\n{final_instruction}"
        )

    @staticmethod
    def _request_from_clinician(clinician: dict[str, Any], action: str) -> str:
        field_names = ["request", "message", "question"]
        if action == "ORDER_TEST":
            field_names.extend(["test_name", "test", "order"])
        elif action == "ORDER_PROCEDURE":
            field_names.extend(["procedure_name", "procedure", "order"])
        elif action == "REQUEST_RESULT":
            field_names.extend(["result_name", "test_name", "result"])
        for field_name in field_names:
            value = clinician.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                return ", ".join(str(item) for item in value if item)
            if isinstance(value, dict) and value:
                return json.dumps(value, ensure_ascii=False)
        if action == "WAIT_REASSESS":
            return "Wait and reassess for pending results or the next clinical update."
        return ""

    def step(self) -> dict[str, Any]:
        turn_number = len(self.turns) + 1
        if turn_number > self.max_turns:
            raise RuntimeError("Episode budget is exhausted")
        clinician = _call_json_model(
            self.clinician_client,
            self.clinician_model,
            CLINICIAN_SYSTEM_PROMPT,
            self._clinician_prompt(turn_number),
            self.clinician_usage,
        )
        action = str(clinician.get("action") or "").upper()
        if turn_number == self.max_turns and action != "FINAL":
            forced_prompt = (
                self._clinician_prompt(turn_number)
                + "\n\nYour previous response did not submit FINAL. Return FINAL now with your current "
                "most likely diagnosis, confidence, and concise diagnostic reasoning."
            )
            clinician = _call_json_model(
                self.clinician_client,
                self.clinician_model,
                CLINICIAN_SYSTEM_PROMPT,
                forced_prompt,
                self.clinician_usage,
            )
            action = str(clinician.get("action") or "").upper()
            if action != "FINAL":
                ddx = clinician.get("current_ddx") or ["unknown"]
                clinician = {
                    "action": "FINAL",
                    "diagnosis": str(ddx[0]),
                    "current_ddx": ddx,
                    "confidence": float(clinician.get("confidence") or 0.0),
                    "reasoning": "Maximum turn budget reached; submitting the current most likely diagnosis.",
                }
                action = "FINAL"
        if action not in VALID_CLINICIAN_ACTIONS:
            raise ValueError(f"Invalid clinician action: {action or '(missing)'}")

        if action == "FINAL":
            turn = DialogueTurn(turn_number=turn_number, clinician=clinician, report=None)
            self.turns.append(turn)
            return {"turn_number": turn_number, "is_final": True, "clinician": clinician, "report": None}

        request = self._request_from_clinician(clinician, action)
        if not request:
            correction_prompt = (
                self._clinician_prompt(turn_number)
                + f"\n\nYour previous {action} response omitted its clinical request. Return the same "
                "decision as valid JSON with one non-empty `request` string."
            )
            clinician = _call_json_model(
                self.clinician_client,
                self.clinician_model,
                CLINICIAN_SYSTEM_PROMPT,
                correction_prompt,
                self.clinician_usage,
            )
            action = str(clinician.get("action") or "").upper()
            if action == "FINAL":
                turn = DialogueTurn(turn_number=turn_number, clinician=clinician, report=None)
                self.turns.append(turn)
                return {"turn_number": turn_number, "is_final": True, "clinician": clinician, "report": None}
            if action not in VALID_CLINICIAN_ACTIONS:
                raise ValueError(f"Invalid corrected clinician action: {action or '(missing)'}")
            request = self._request_from_clinician(clinician, action)
            if not request:
                raise ValueError(f"{action} requires a non-empty request after schema correction")
        clinician["request"] = request
        report = self.report_agent.answer(action, request, self.disclosed_ids)
        if report["status"] == "found":
            self.disclosed_ids.add(report["evidence_id"])
        turn = DialogueTurn(turn_number=turn_number, clinician=clinician, report=report)
        self.turns.append(turn)
        return {"turn_number": turn_number, "is_final": False, "clinician": clinician, "report": report}

    def episode_payload(self, final: dict[str, Any] | None = None) -> dict[str, Any]:
        if final is None and self.turns and self.turns[-1].clinician.get("action") == "FINAL":
            final = self.turns[-1].clinician
        return {
            "case_id": self.case_data.get("case_id", ""),
            "max_turns": self.max_turns,
            "turns_used": len(self.turns),
            "termination": "final" if final is not None else "in_progress",
            "disclosed_evidence_ids": sorted(self.disclosed_ids),
            "turns": [
                {"turn_number": turn.turn_number, "clinician": turn.clinician, "report": turn.report}
                for turn in self.turns
            ],
            "final": final,
            "model_usage": {
                "clinician": dict(self.clinician_usage),
                "report": dict(self.report_agent.usage),
            },
        }

    def run_episode(
        self,
        on_turn: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        final: dict[str, Any] | None = None
        if self.turns and self.turns[-1].clinician.get("action") == "FINAL":
            final = self.turns[-1].clinician
        if final is not None:
            return self.episode_payload(final)
        while len(self.turns) < self.max_turns:
            result = self.step()
            if result["is_final"]:
                final = result["clinician"]
            if on_turn:
                on_turn(self.episode_payload(final))
            if final is not None:
                break
        episode = self.episode_payload(final)
        if final is None:
            episode["termination"] = "budget_exhausted"
        return episode
