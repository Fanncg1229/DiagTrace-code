from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

from interactive_diagnosis_workspace.dialogue_workflow.workflow import (
    PHASE_LABELS,
    EvidenceItem,
    _build_client,
    _call_json_model,
)


VALID_TRAJECTORY_ACTIONS = frozenset(
    {
        "ASK_HISTORY",
        "REQUEST_EXAM",
        "ORDER_TEST",
        "REQUEST_RESULT",
        "ORDER_PROCEDURE",
        "WAIT_REASSESS",
        "FINAL_TRAJECTORY",
    }
)


ACTION_ALIASES = {
    "REQUEST_HISTORY": "ASK_HISTORY",
    "ASK_CLINICAL_HISTORY": "ASK_HISTORY",
    "REQUEST_CLINICAL_HISTORY": "ASK_HISTORY",
    "REQUEST_PHYSICAL_EXAM": "REQUEST_EXAM",
    "REQUEST_PHYSICAL_EXAMINATION": "REQUEST_EXAM",
    "REQUEST_TEST": "REQUEST_RESULT",
    "REQUEST_LAB": "REQUEST_RESULT",
    "REQUEST_LABS": "REQUEST_RESULT",
    "REQUEST_LAB_TEST": "REQUEST_RESULT",
    "REQUEST_LAB_TESTS": "REQUEST_RESULT",
    "REQUEST_IMAGING": "REQUEST_RESULT",
    "REQUEST_PATHOLOGY": "REQUEST_RESULT",
    "REQUEST_MICROBIOLOGY": "REQUEST_RESULT",
    "REQUEST_PROCEDURE": "ORDER_PROCEDURE",
}


def _normalize_action(action: Any) -> str:
    normalized = str(action or "").strip().upper()
    return ACTION_ALIASES.get(normalized, normalized)


def _infer_missing_action(clinician: dict[str, Any]) -> str:
    if any(key in clinician for key in ("trajectory_summary", "key_supporting_evidence", "remaining_uncertainties")):
        return "FINAL_TRAJECTORY"
    if any(key in clinician for key in ("request", "message", "question", "test_name", "procedure_name", "result_name")):
        return "REQUEST_RESULT"
    return ""


def _is_claude_model(model: str | None) -> bool:
    return bool(model and str(model).strip().lower().startswith("claude"))


class ClaudeFoundryChatClient:
    """Tiny Anthropic Foundry adapter exposing the OpenAI-style method used here."""

    def __init__(self, *, endpoint: str, api_key: str, timeout: int = 600) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _messages_url(self) -> str:
        if self.endpoint.endswith("/messages"):
            return self.endpoint
        return f"{self.endpoint}/messages"

    def _create(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> Any:
        system = ""
        user_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "system":
                system = content if not system else f"{system}\n\n{content}"
            elif role in {"user", "assistant"}:
                user_messages.append({"role": role, "content": content})
        payload = {
            "model": model,
            "max_tokens": 4096,
            "temperature": 0,
            "system": system,
            "messages": user_messages,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._messages_url(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "api-key": self.api_key,
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude Foundry request failed with HTTP {exc.code}: {body}") from exc
        content_blocks = parsed.get("content") or []
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text") or ""))
        usage = parsed.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="\n".join(text_parts)))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


class OpenAICompatibleChatClient:
    """Small chat-completions adapter for OpenAI-compatible model endpoints."""

    def __init__(self, *, base_url: str, api_key: str, timeout: int = 600) -> None:
        self.base_url = self._normalize_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        parsed = urllib.parse.urlsplit(base_url.strip())
        if "/api/projects/" in parsed.path:
            parsed = parsed._replace(path="/openai/v1", query="", fragment="")
        return urllib.parse.urlunsplit(parsed).rstrip("/")

    def _chat_completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _create(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> Any:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(),
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                parsed = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
        choice = (parsed.get("choices") or [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = str((message or {}).get("content") or "")
        usage = parsed.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )


def _build_claude_foundry_client() -> ClaudeFoundryChatClient:
    required = {
        "AZURE_FOUNDRY_CLAUDE_API_KEY": os.getenv("AZURE_FOUNDRY_CLAUDE_API_KEY"),
        "AZURE_FOUNDRY_CLAUDE_ENDPOINT": os.getenv("AZURE_FOUNDRY_CLAUDE_ENDPOINT"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    return ClaudeFoundryChatClient(
        endpoint=str(required["AZURE_FOUNDRY_CLAUDE_ENDPOINT"]),
        api_key=str(required["AZURE_FOUNDRY_CLAUDE_API_KEY"]),
    )


def _is_kimi_model(model: str | None) -> bool:
    return bool(model and str(model).strip().lower().startswith("kimi"))


def _is_gemini_model(model: str | None) -> bool:
    return bool(model and str(model).strip().lower().startswith("gemini"))


def _build_kimi_client() -> OpenAICompatibleChatClient:
    provider = (os.getenv("KIMI_PROVIDER") or "").strip().casefold()
    official_api_key = os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
    official_base_url = os.getenv("KIMI_CHAT_COMPLETIONS_URL") or os.getenv("KIMI_BASE_URL")
    azure_api_key = os.getenv("AZURE_FOUNDRY_KIMI_API_KEY")
    azure_base_url = os.getenv("AZURE_FOUNDRY_KIMI_ENDPOINT") or os.getenv("AZURE_FOUNDRY_KIMI_BASE_URL")
    if provider in {"moonshot", "official", "kimi"}:
        api_key = official_api_key
        base_url = official_base_url
    elif provider in {"azure", "foundry"}:
        api_key = azure_api_key
        base_url = azure_base_url
    elif official_api_key and official_base_url:
        api_key = official_api_key
        base_url = official_base_url
    else:
        api_key = azure_api_key
        base_url = azure_base_url
    missing = []
    if not api_key:
        missing.append("KIMI_API_KEY/MOONSHOT_API_KEY or AZURE_FOUNDRY_KIMI_API_KEY")
    if not base_url:
        missing.append("KIMI_BASE_URL/KIMI_CHAT_COMPLETIONS_URL or AZURE_FOUNDRY_KIMI_ENDPOINT")
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    timeout = int(os.getenv("KIMI_TIMEOUT_SECONDS") or os.getenv("OPENAI_COMPATIBLE_TIMEOUT_SECONDS") or 600)
    return OpenAICompatibleChatClient(base_url=str(base_url), api_key=str(api_key), timeout=timeout)


class GeminiChatClient:
    """Google Gen AI adapter exposing the OpenAI-style method used here."""

    def __init__(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("Install google-genai before using Gemini models: python -m pip install google-genai") from exc
        self._types = types
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").casefold() in {"1", "true", "yes"}:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
            if not project:
                raise RuntimeError("Missing GOOGLE_CLOUD_PROJECT for Vertex AI Gemini")
            self.client = genai.Client(vertexai=True, project=project, location=location)
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY, or set GOOGLE_GENAI_USE_VERTEXAI=True for Vertex AI")
            self.client = genai.Client(api_key=api_key)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]], **_: Any) -> Any:
        system_parts: list[str] = []
        content_parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                content_parts.append(content)
        response = self.client.models.generate_content(
            model=model,
            contents="\n\n".join(content_parts),
            config=self._types.GenerateContentConfig(
                system_instruction="\n\n".join(system_parts) or None,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage, "total_token_count", 0) or prompt_tokens + completion_tokens)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=getattr(response, "text", "") or "{}"))],
            usage=SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )


def _build_gemini_client() -> GeminiChatClient:
    return GeminiChatClient()


def _build_client_for_model(model: str | None) -> Any:
    """Pick the provider client that serves `model`; defaults to Azure OpenAI."""
    if _is_claude_model(model):
        return _build_claude_foundry_client()
    if _is_kimi_model(model):
        return _build_kimi_client()
    if _is_gemini_model(model):
        return _build_gemini_client()
    return _build_client()


TRAJECTORY_CLINICIAN_SYSTEM_PROMPT = """
You are the clinician-auditor being evaluated in an interactive clinical trajectory reconstruction benchmark.
Return valid JSON only.

You are given an initial presentation and the known final diagnosis. Your task is not
to re-diagnose the patient. Your task is to actively reconstruct and verify the
source-grounded clinical evidence trajectory that supports the known final diagnosis.

Valid actions:
- ASK_HISTORY: ask about symptoms, exposures, medications, or past history
- REQUEST_EXAM: ask for physical examination or bedside findings
- ORDER_TEST: order one laboratory, imaging, microbiology, molecular, or genetic test
- REQUEST_RESULT: request one available lab, imaging-report, pathology, molecular,
  microbiology, treatment-response, clinical-course, or source-extracted image finding
- ORDER_PROCEDURE: order one diagnostic procedure such as biopsy, endoscopy, surgery, or sampling
- WAIT_REASSESS: wait for a pending result or ask for the next source-grounded clinical
  course update, such as hospitalization, procedure, complication, treatment response,
  treatment failure, or outcome
- FINAL_TRAJECTORY: submit the reconstructed clinical trajectory and evidence support

Rules:
- Ask only one focused clinical request per turn.
- Use the known final diagnosis to guide information seeking, but do not treat textbook
  expectations as observed case evidence.
- Only information returned by the report agent is observed case evidence.
- Prefer concrete clinical evidence requests over meta-questions such as "why was this
  diagnosis made?" or "what confirmed the diagnosis?"
- Reconstruct temporal order: presentation -> workup -> key findings -> confirmation ->
  course/treatment/outcome when available.
- Do not request raw image pixels; only textual image findings/reports are available.
- If stalled, use WAIT_REASSESS to obtain the next documented clinical update.
- Before FINAL_TRAJECTORY, try to verify the trajectory across the clinically relevant
  parts of the case: initial manifestation, key diagnostic test/report, temporal
  progression or procedure, treatment response/outcome when relevant, and documented
  rule-out evidence when it materially supports the known diagnosis.
- Do not finalize after only one confirmatory result if major parts of the source-grounded
  trajectory remain unverified and could be requested within the turn budget.
- Do not ask for every evidence category mechanically. Prioritize evidence that is
  clinically relevant to the known final diagnosis and trajectory.
- On the final available turn you must submit FINAL_TRAJECTORY.

For an information request, return:
{
  "action": "one valid non-final action",
  "request": "one focused clinical request",
  "current_trajectory_hypothesis": "brief current source-grounded trajectory hypothesis",
  "confirmed_evidence": ["brief claims supported by disclosed report replies"],
  "missing_key_evidence": ["brief claims or evidence types still needed"],
  "confidence_that_diagnosis_is_supported": 0.0
}

For the final answer, return:
{
  "action": "FINAL_TRAJECTORY",
  "final_diagnosis": "the known final diagnosis",
  "trajectory_summary": "concise temporal summary grounded only in disclosed evidence",
  "key_supporting_evidence": [
    {"claim": "source-grounded claim", "evidence_ids": ["IDs from disclosed report replies"]}
  ],
  "ruled_out_or_less_likely_alternatives": [
    {"alternative": "condition", "basis": "source-grounded basis or empty"}
  ],
  "remaining_uncertainties": ["missing or not documented evidence, if any"],
  "confidence_that_diagnosis_is_supported": 0.0
}
""".strip()


TRAJECTORY_REPORT_SYSTEM_PROMPT = """
You are the Report Agent in a source-grounded diagnosis-aware clinical trajectory benchmark.
Return valid JSON only.

The clinician already knows the final diagnosis. You simulate a longitudinal hospital
information and reporting platform. You see all source-extracted PMC evidence, but the
clinician sees only evidence that you release. Your job is to provide requested clinical
evidence from the source so the clinician can reconstruct the trajectory. You must not
write or paraphrase the medical answer yourself.

Rules:
1. Select only from the supplied evidence records.
2. Do not diagnose, interpret significance, recommend a next step, or infer missing facts.
3. Return at most one evidence_id. Never reveal an ID or medical content in explanation.
4. Keep phase_index_after monotonic. ORDER_TEST, ORDER_PROCEDURE, pending, and
   ordered_but_not_documented never advance phase. A phase advances only when you return
   found for a later record; set phase_index_after to that record's phase.
5. REQUEST_RESULT does not require a prior order. If a directly responsive source record
   exists, you may return it and advance to its phase.
6. WAIT_REASSESS has strict priority order. If Pending orders is non-empty and the first
   pending order has a source-backed evidence_id, you must return found for that pending
   evidence_id before releasing any other update. If no pending order exists, release the
   earliest undisclosed source-grounded clinical course update at or after the current
   phase. Clinical course updates include hospitalization, ICU transfer, procedure,
   surgery, new symptoms or organ involvement, complications, treatment response/failure,
   or outcome. Return no_new_update only when there is no pending result and no undisclosed
   source-grounded clinical course update.
7. If ORDER_TEST or ORDER_PROCEDURE matches a source record, return ordered and include
   that evidence_id internally. A later matching REQUEST_RESULT or WAIT_REASSESS must
   return found for that result; do not keep source-backed results pending repeatedly.
8. If the source has no result for an order, return ordered_but_not_documented without an
   evidence_id. This means the article never documents the result, not that it may appear later.
9. If the clinician asks a meta-diagnostic question such as "what confirmed the diagnosis",
   "why was this diagnosed", "give the diagnostic reasoning", or "what evidence supports
   [known diagnosis]" without naming a specific clinical evidence type, return
   request_too_diagnostic and ask operationally for a specific evidence type. Do not provide
   medical content in the explanation.
10. Legal clinical results include symptoms, exam findings, laboratory values, imaging
    report findings, pathology morphology/IHC, microbiology, genetic variants, procedures,
    clinical course, treatment response, and outcome.
11. Do not expose article metadata, retrospective author narration, diagnostic-process
    prose, final-diagnosis prose, or evidence not responsive to the request.
12. A record marked requires_safe_excerpt mixes usable clinical evidence with diagnostic
    narration. Select it only when its contemporaneous result directly answers the request,
    and return a strict contiguous verbatim_excerpt containing only the result. Exclude
    clauses such as "thus establishing the diagnosis", "confirmed", "diagnosed as", or
    "supported the diagnosis". Never return the complete text of such a record.
13. If a request is reasonable and any supplied record is clinically related, return one
    related source-backed record. Prefer the earliest available related record and let the
    verbatim source text define the answer. Do not combine non-contiguous text spans or
    multiple records into a single verbatim_excerpt; this version supports one evidence
    record per turn.
14. Absence from the source means not_reported, never a negative clinical result.
15. If the request contains multiple independent questions but one component has a related
    source-backed record, return that component. Return too_broad only when no single related
    record can be selected without guessing priority.

Return exactly one of:
{
  "status": "found | pending | ordered | ordered_but_not_documented | not_reported | too_broad | invalid_request | no_new_update | request_too_diagnostic",
  "evidence_id": "one ID only when grounded in a source record, otherwise empty",
  "verbatim_excerpt": "optional contiguous source substring; required for requires_safe_excerpt",
  "phase_index_after": 0,
  "phase_label_after": "one supplied phase label",
  "pending_order": "concise ordered test/procedure name or empty",
  "explanation": "operational status only; no medical content"
}
""".strip()


@dataclass(frozen=True)
class TrajectoryTurn:
    turn_number: int
    clinician: dict[str, Any]
    report: dict[str, Any] | None


class TrajectoryReportAgent:
    """Source selector for diagnosis-aware trajectory reconstruction."""

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
        if action == "WAIT_REASSESS" and self.pending_orders:
            prompt += (
                "\n\nProtocol reminder: WAIT_REASSESS must release the first pending "
                "source-backed result before any other clinical update. Return found for "
                f"evidence_id={self.pending_orders[0].get('evidence_id', '')} unless that "
                "record cannot be safely excerpted under the rules."
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
            TRAJECTORY_REPORT_SYSTEM_PROMPT,
            prompt,
            self.usage,
        )
        status = str(selected.get("status") or "invalid_request").lower()
        if status == "ordered_no_source_result":
            status = "ordered_but_not_documented"
        evidence_id = str(selected.get("evidence_id") or "")
        try:
            phase_index_after = int(selected.get("phase_index_after", self.current_phase_index))
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
            "ASK_HISTORY": {"found", "not_reported", "too_broad", "invalid_request", "request_too_diagnostic"},
            "REQUEST_EXAM": {"found", "not_reported", "too_broad", "invalid_request", "request_too_diagnostic"},
            "ORDER_TEST": {"ordered", "ordered_but_not_documented", "too_broad", "invalid_request", "request_too_diagnostic"},
            "ORDER_PROCEDURE": {"ordered", "ordered_but_not_documented", "too_broad", "invalid_request", "request_too_diagnostic"},
            "REQUEST_RESULT": {"found", "pending", "not_reported", "too_broad", "invalid_request", "request_too_diagnostic"},
            "WAIT_REASSESS": {"found", "pending", "not_reported", "no_new_update", "invalid_request", "request_too_diagnostic"},
        }
        if status not in allowed_statuses.get(action, set()):
            return invalid_selection(f"Status {status} is not valid for clinician action {action}.")

        if status in {"ordered_but_not_documented", "request_too_diagnostic"}:
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
                return invalid_selection("WAIT_REASSESS must release a pending source-backed result.")
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
        self.pending_orders = [entry for entry in self.pending_orders if entry.get("evidence_id") != item.id]
        return finish(
            {
                "status": "already_disclosed" if item.id in disclosed_ids else "found",
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


class TrajectoryWorkflow:
    """Known-diagnosis, source-grounded clinical trajectory reconstruction loop."""

    def __init__(
        self,
        case_data: dict[str, Any],
        clinician_client: Any | None = None,
        report_client: Any | None = None,
        clinician_model: str | None = None,
        report_model: str | None = None,
        max_turns: int = 15,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.case_data = case_data
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
        self.clinician_client = clinician_client or _build_client_for_model(self.clinician_model)
        self.report_client = report_client or _build_client_for_model(self.report_model)
        self.max_turns = max_turns
        self.initial_observation = case_data.get("initial_observation", {})
        hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
        self.final_diagnosis = str(hidden.get("final_diagnosis_verbatim") or hidden.get("final_diagnosis") or "")
        self.report_agent = TrajectoryReportAgent(
            evidence_items=case_data.get("evidence_items") or case_data.get("evidence_events") or [],
            client=self.report_client,
            model=self.report_model,
        )
        self.disclosed_ids: set[str] = set()
        self.turns: list[TrajectoryTurn] = []
        self.clinician_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _clinician_prompt(self, turn_number: int) -> str:
        dialogue = [
            {"turn": turn.turn_number, "clinician": turn.clinician, "report": turn.report}
            for turn in self.turns
        ]
        final_instruction = (
            "This is the final available turn. You must submit FINAL_TRAJECTORY."
            if turn_number == self.max_turns
            else "Ask one focused request or submit FINAL_TRAJECTORY if the trajectory is sufficiently reconstructed."
        )
        return (
            f"Initial presentation:\n{json.dumps(self.initial_observation, ensure_ascii=False, indent=2)}\n\n"
            f"Known final diagnosis:\n{self.final_diagnosis}\n\n"
            f"Dialogue so far:\n{json.dumps(dialogue, ensure_ascii=False, indent=2)}\n\n"
            f"Turn: {turn_number}/{self.max_turns}\n{final_instruction}"
        )

    @staticmethod
    def _request_from_clinician(clinician: dict[str, Any], action: str) -> str:
        if action == "WAIT_REASSESS":
            return "Wait and reassess for pending results or the next clinical trajectory update."
        for field_name in ("request", "message", "question", "test_name", "procedure_name", "result_name"):
            value = clinician.get(field_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list) and value:
                return ", ".join(str(item) for item in value if item)
            if isinstance(value, dict) and value:
                return json.dumps(value, ensure_ascii=False)
        return ""

    @staticmethod
    def _cited_evidence_ids(final: dict[str, Any]) -> set[str]:
        cited: set[str] = set()
        key_supporting = final.get("key_supporting_evidence")
        if isinstance(key_supporting, list):
            for item in key_supporting:
                if isinstance(item, dict):
                    cited.update(str(value) for value in item.get("evidence_ids", []) if value)
        return cited

    def _normalize_final_citations(self, final: dict[str, Any]) -> dict[str, Any]:
        key_supporting = final.get("key_supporting_evidence")
        if not isinstance(key_supporting, list):
            return final
        normalized_final = dict(final)
        normalized_supporting: list[Any] = []
        for item in key_supporting:
            if not isinstance(item, dict):
                normalized_supporting.append(item)
                continue
            normalized_item = dict(item)
            normalized_ids: list[str] = []
            for value in item.get("evidence_ids", []) or []:
                raw = str(value or "").strip()
                if not raw:
                    continue
                if raw in self.disclosed_ids:
                    candidate = raw
                else:
                    match = re.search(r"\bE\d{3}\b", raw)
                    candidate = match.group(0) if match else ""
                if candidate and candidate in self.disclosed_ids and candidate not in normalized_ids:
                    normalized_ids.append(candidate)
            normalized_item["evidence_ids"] = normalized_ids
            normalized_supporting.append(normalized_item)
        normalized_final["key_supporting_evidence"] = normalized_supporting
        return normalized_final

    def _repair_final_if_needed(self, final: dict[str, Any], turn_number: int) -> dict[str, Any]:
        final = self._normalize_final_citations(final)
        invalid_ids = sorted(self._cited_evidence_ids(final) - self.disclosed_ids)
        if not invalid_ids:
            return final
        disclosed_reports = [
            turn.report
            for turn in self.turns
            if isinstance(turn.report, dict) and turn.report.get("status") in {"found", "already_disclosed"}
        ]
        repair_prompt = (
            self._clinician_prompt(turn_number)
            + "\n\nYour previous FINAL_TRAJECTORY cited evidence IDs that were not disclosed: "
            + json.dumps(invalid_ids, ensure_ascii=False)
            + "\nRewrite FINAL_TRAJECTORY now. Use only these disclosed report replies as observed evidence:\n"
            + json.dumps(disclosed_reports, ensure_ascii=False, indent=2)
        )
        repaired = _call_json_model(
            self.clinician_client,
            self.clinician_model,
            TRAJECTORY_CLINICIAN_SYSTEM_PROMPT,
            repair_prompt,
            self.clinician_usage,
        )
        if str(repaired.get("action") or "").upper() == "FINAL_TRAJECTORY":
            return self._normalize_final_citations(repaired)
        return final

    def step(self) -> dict[str, Any]:
        turn_number = len(self.turns) + 1
        clinician = _call_json_model(
            self.clinician_client,
            self.clinician_model,
            TRAJECTORY_CLINICIAN_SYSTEM_PROMPT,
            self._clinician_prompt(turn_number),
            self.clinician_usage,
        )
        action = _normalize_action(clinician.get("action")) or _infer_missing_action(clinician)
        if action:
            clinician["action"] = action
        if turn_number == self.max_turns and action != "FINAL_TRAJECTORY":
            clinician = _call_json_model(
                self.clinician_client,
                self.clinician_model,
                TRAJECTORY_CLINICIAN_SYSTEM_PROMPT,
                self._clinician_prompt(turn_number)
                + "\n\nYour previous response did not submit FINAL_TRAJECTORY. Return FINAL_TRAJECTORY now.",
                self.clinician_usage,
            )
            action = _normalize_action(clinician.get("action")) or _infer_missing_action(clinician)
            if action:
                clinician["action"] = action
        if action not in VALID_TRAJECTORY_ACTIONS:
            raise ValueError(f"Invalid clinician action: {action or '(missing)'}")
        if action == "FINAL_TRAJECTORY":
            clinician = self._repair_final_if_needed(clinician, turn_number)
            turn = TrajectoryTurn(turn_number=turn_number, clinician=clinician, report=None)
            self.turns.append(turn)
            return {"turn_number": turn_number, "is_final": True, "clinician": clinician, "report": None}

        request = self._request_from_clinician(clinician, action)
        if not request:
            raise ValueError(f"{action} requires a non-empty request")
        clinician["request"] = request
        report = self.report_agent.answer(action, request, self.disclosed_ids)
        if report["status"] == "found":
            self.disclosed_ids.add(report["evidence_id"])
        turn = TrajectoryTurn(turn_number=turn_number, clinician=clinician, report=report)
        self.turns.append(turn)
        return {"turn_number": turn_number, "is_final": False, "clinician": clinician, "report": report}

    def episode_payload(self, final: dict[str, Any] | None = None) -> dict[str, Any]:
        if final is None and self.turns and self.turns[-1].clinician.get("action") == "FINAL_TRAJECTORY":
            final = self.turns[-1].clinician
        return {
            "case_id": self.case_data.get("case_id", ""),
            "setting": "trajectory_reconstruction",
            "max_turns": self.max_turns,
            "turns_used": len(self.turns),
            "termination": "final" if final is not None else "in_progress",
            "known_final_diagnosis": self.final_diagnosis,
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

    def run_episode(self, on_turn: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        final: dict[str, Any] | None = None
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
