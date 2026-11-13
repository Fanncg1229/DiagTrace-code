from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import AzureOpenAI

from .workflow import PHASE_LABELS, normalize_evidence_category


load_dotenv(find_dotenv(usecwd=True))


SCHEMA_VERSION = "2026-07-dialogue-workflow-extraction-v2-workup-anchor"

EVIDENCE_CATEGORIES = frozenset(
    {
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
        "other",
    }
)


EXTRACTION_SYSTEM_PROMPT = """
You extract a source-grounded longitudinal medical case from a PMC case report for
a two-LLM interactive diagnosis benchmark. Return valid JSON only.

The online benchmark works as follows: a Clinician LLM sees only an incomplete
initial presentation and asks a Hospital/Report LLM for information. The Report LLM
may return only the verbatim evidence records produced here. Therefore source fidelity,
clinical chronology, and diagnosis-leakage labels are critical.

Extraction rules:
1. Use only facts explicitly present in the supplied article text.
2. Every source_text must be a contiguous verbatim quote from the supplied text.
   Do not paraphrase, summarize, combine sentences from different locations, correct
   grammar, normalize units, or add medical knowledge.
3. Extract one coherent clinical event or contemporaneous result panel per evidence item.
4. Put events in clinical chronological order, which may differ from document order.
5. Repeated measurements must be separate records. Give them the same stable repeat_key
   (for example cbc, crp, chest_ct) and distinct temporal_cue values.
6. initial_observation must represent the first encounter at which diagnostic workup
   begins for the still-undiagnosed condition being evaluated. Include demographics,
   presenting symptoms, known prior history (including relevant antecedent events), and
   facts already available at that encounter. Do not start from an earlier incidental
   event merely because it is first chronologically, and do not compress later results
   into the initial view.
7. Mark diagnosis_revealing=true for any record that states the final diagnosis, says a
   diagnosis was confirmed/established, retrospectively explains why the diagnosis is
   correct, or otherwise directly gives away the hidden answer. Retain the entire verbatim
   record even when one sentence mixes a legitimate test result with diagnostic narration;
   do not move that sentence only into hidden_reference and do not split it upstream. The
   online workflow may later expose only a validated verbatim subspan containing the test
   result.
8. A named intermediate diagnosis or documented contemporaneous impression may remain
   usable when it is part of the actual workup and is not the final hidden diagnosis.
9. Do not create a negative result from missing information. Extract negative tests only
   when the article explicitly states them.
10. For figures, extract only textual findings explicitly stated in the case narrative,
    radiology/pathology report, or figure caption. Set image_finding=true. Do not infer
    anything from pixels and do not output image paths.
11. Article title, abstract conclusions, discussion, and conclusion may help identify the
    hidden reference but must not be copied into initial_observation or usable evidence if
    they reveal the final diagnosis.
12. source_section names where the quote appeared. timeline_order starts at 0 and increases.

Use phase_index and phase_label pairs:
0 initial_presentation
1 early_workup
2 targeted_workup
3 clinical_progression
4 advanced_workup
5 confirmatory_testing
6 treatment_or_outcome

Return this schema:
{
  "initial_observation": {
    "demographics": "source-grounded demographics",
    "encounter_context": "where/why the first encounter occurred",
    "chief_complaint": "initial presenting problem",
    "first_contact_summary": "incomplete first-contact presentation only",
    "presenting_symptoms": ["..."],
    "known_prior_history": ["..."]
  },
  "evidence_items": [
    {
      "id": "E001",
      "category": "one allowed category",
      "source_text": "contiguous verbatim source quote",
      "source_section": "Case presentation or Figure caption",
      "temporal_cue": "before admission, on admission, ICU day 2, or empty",
      "timeline_order": 0,
      "phase_index": 0,
      "phase_label": "initial_presentation",
      "repeat_key": "stable normalized key or empty",
      "image_finding": false,
      "diagnosis_revealing": false
    }
  ],
  "hidden_reference": {
    "final_diagnosis_verbatim": "final diagnosis as stated by the article",
    "accepted_aliases": ["explicit aliases only"],
    "diagnostic_reasoning_verbatim_spans": ["contiguous source quote"]
  }
}
""".strip()


@dataclass(frozen=True)
class ExtractionValidation:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


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


def _call_extractor(
    client: Any,
    model: str,
    article_text: str,
    retry_instruction: str = "",
) -> tuple[dict[str, Any], dict[str, int]]:
    user_prompt = f"PMC article text:\n\n{article_text}"
    if retry_instruction:
        user_prompt += (
            "\n\nPrevious extraction failed validation. Return a corrected full JSON extraction.\n"
            f"Validation errors:\n{retry_instruction}\n\n"
            "Critical repair rule: every evidence_items[].source_text and every hidden_reference "
            "verbatim span must be copied exactly as a contiguous substring from the supplied "
            "article text after whitespace normalization. Do not paraphrase, merge distant "
            "phrases, rewrite grammar, or normalize units."
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=16000,
        response_format={"type": "json_object"},
    )
    content = getattr(response.choices[0].message, "content", "{}") or "{}"
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Extraction model response was not a JSON object")
    usage = getattr(response, "usage", None)
    usage_dict = {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return parsed, usage_dict


def _normalized_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _match_key(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    value = value.replace("‐", "-").replace("–", "-").replace("—", "-")
    value = value.replace("’", "'").replace("“", '"').replace("”", '"')
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _is_verbatim_span(value: str, article_text: str) -> bool:
    return bool(value and _normalized_space(value) in _normalized_space(article_text))


def _candidate_article_spans(article_text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for block in article_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        body = block.split("\n", 1)[1].strip() if "\n" in block else block
        if not body:
            continue
        sentence_matches = list(re.finditer(r"[^.!?]+(?:[.!?]+|$)", body))
        spans = [body]
        sentences = [match.group(0).strip() for match in sentence_matches if match.group(0).strip()]
        spans.extend(sentences)
        for window in (2, 3):
            for start in range(0, max(0, len(sentences) - window + 1)):
                spans.append(" ".join(sentences[start : start + window]).strip())
        for span in spans:
            key = _normalized_space(span)
            if key and key not in seen:
                seen.add(key)
                candidates.append(key)
    return candidates


def _repair_verbatim_span(source_text: str, article_text: str) -> tuple[str, float] | None:
    source_key = _match_key(source_text)
    source_tokens = set(source_key.split())
    if len(source_key) < 20 or not source_tokens:
        return None

    best_span = ""
    best_score = 0.0
    for candidate in _candidate_article_spans(article_text):
        candidate_key = _match_key(candidate)
        if not candidate_key:
            continue
        candidate_tokens = set(candidate_key.split())
        overlap = len(source_tokens & candidate_tokens) / max(len(source_tokens), 1)
        if overlap < 0.35:
            continue
        ratio = difflib.SequenceMatcher(None, source_key, candidate_key).ratio()
        score = (0.65 * ratio) + (0.35 * overlap)
        if score > best_score:
            best_score = score
            best_span = candidate

    if best_span and best_score >= 0.72 and _is_verbatim_span(best_span, article_text):
        return best_span, best_score
    return None


def repair_source_text_spans(case_data: dict[str, Any], article_text: str) -> list[dict[str, Any]]:
    repairs: list[dict[str, Any]] = []
    for item in case_data.get("evidence_items") or []:
        if not isinstance(item, dict):
            continue
        source_text = str(item.get("source_text") or "").strip()
        if not source_text or item.get("verbatim_verified"):
            continue
        repaired = _repair_verbatim_span(source_text, article_text)
        if repaired is None:
            continue
        repaired_text, score = repaired
        item["source_text"] = repaired_text
        item["verbatim_verified"] = _is_verbatim_span(repaired_text, article_text)
        repairs.append(
            {
                "evidence_id": str(item.get("id") or ""),
                "score": round(score, 4),
                "original_length": len(source_text),
                "repaired_length": len(repaired_text),
            }
        )
    return repairs


def _normalize_initial(raw: Any) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "demographics": str(raw.get("demographics") or ""),
        "encounter_context": str(raw.get("encounter_context") or ""),
        "chief_complaint": str(raw.get("chief_complaint") or ""),
        "first_contact_summary": str(raw.get("first_contact_summary") or ""),
        "presenting_symptoms": [str(value) for value in raw.get("presenting_symptoms", []) if value],
        "known_prior_history": [str(value) for value in raw.get("known_prior_history", []) if value],
    }


def normalize_extraction(
    extracted: dict[str, Any],
    *,
    case_id: str,
    article_text: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    evidence_items: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(extracted.get("evidence_items", [])):
        if not isinstance(raw, dict):
            continue
        evidence_id = str(raw.get("id") or raw.get("evidence_id") or f"E{index + 1:03d}")
        if evidence_id in used_ids:
            evidence_id = f"E{index + 1:03d}"
        used_ids.add(evidence_id)
        phase_index = int(raw.get("phase_index") or 0)
        if phase_index < 0 or phase_index >= len(PHASE_LABELS):
            phase_index = 0
        category = normalize_evidence_category(raw)
        if category not in EVIDENCE_CATEGORIES:
            category = "other"
        source_text = str(raw.get("source_text") or raw.get("source_quote") or "").strip()
        evidence_items.append(
            {
                "id": evidence_id,
                "category": category,
                "source_text": source_text,
                "source_section": str(raw.get("source_section") or ""),
                "temporal_cue": str(raw.get("temporal_cue") or ""),
                "timeline_order": int(raw.get("timeline_order") if raw.get("timeline_order") is not None else index),
                "phase_index": phase_index,
                "phase_label": PHASE_LABELS[phase_index],
                "repeat_key": str(raw.get("repeat_key") or ""),
                "image_finding": bool(raw.get("image_finding") or False),
                "diagnosis_revealing": bool(raw.get("diagnosis_revealing") or False),
                "verbatim_verified": _is_verbatim_span(source_text, article_text),
            }
        )
    evidence_items.sort(key=lambda item: (item["timeline_order"], item["id"]))

    hidden = extracted.get("hidden_reference", {})
    hidden = hidden if isinstance(hidden, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "source_metadata": {
            "extraction_model": model,
            "usage": dict(usage or {}),
        },
        "initial_observation": _normalize_initial(extracted.get("initial_observation")),
        "evidence_items": evidence_items,
        "hidden_reference": {
            "final_diagnosis_verbatim": str(hidden.get("final_diagnosis_verbatim") or ""),
            "accepted_aliases": [str(value) for value in hidden.get("accepted_aliases", []) if value],
            "diagnostic_reasoning_verbatim_spans": [
                str(value) for value in hidden.get("diagnostic_reasoning_verbatim_spans", []) if value
            ],
        },
    }


def validate_extraction(case_data: dict[str, Any]) -> ExtractionValidation:
    errors: list[str] = []
    warnings: list[str] = []
    initial = case_data.get("initial_observation") or {}
    if not initial.get("first_contact_summary"):
        errors.append("initial_observation.first_contact_summary is required")
    evidence_items = case_data.get("evidence_items") or []
    if not evidence_items:
        errors.append("at least one evidence item is required")
    for item in evidence_items:
        evidence_id = item.get("id", "(missing)")
        if not item.get("source_text"):
            errors.append(f"{evidence_id}: source_text is empty")
        elif not item.get("verbatim_verified"):
            errors.append(f"{evidence_id}: source_text is not a verbatim span of the supplied article")
        if item.get("phase_label") != PHASE_LABELS[item.get("phase_index", 0)]:
            errors.append(f"{evidence_id}: phase label/index mismatch")

    hidden = case_data.get("hidden_reference") or {}
    diagnosis = str(hidden.get("final_diagnosis_verbatim") or "").strip()
    if not diagnosis:
        errors.append("hidden_reference.final_diagnosis_verbatim is required")
    elif diagnosis.lower() in json.dumps(initial, ensure_ascii=False).lower():
        errors.append("final diagnosis appears in initial_observation")

    usable = [item for item in evidence_items if not item.get("diagnosis_revealing")]
    if not usable:
        errors.append("no non-diagnosis-revealing evidence is available to the Report Agent")
    if not any(item.get("repeat_key") for item in evidence_items):
        warnings.append("no repeat_key values were extracted; verify whether the case has repeated measurements")
    return ExtractionValidation(tuple(errors), tuple(warnings))


def extract_from_case_text(
    article_text: str,
    *,
    case_id: str,
    client: Any | None = None,
    model: str | None = None,
    max_validation_retries: int = 1,
) -> dict[str, Any]:
    if not article_text.strip():
        raise ValueError("article_text must not be empty")
    selected_model = (
        model
        or os.getenv("AZURE_OPENAI_EXTRACTION_DEPLOYMENT")
        or os.getenv("AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4")
        or "gpt-4.1"
    )
    extraction_client = client or _build_client()
    attempts: list[dict[str, Any]] = []
    retry_instruction = ""
    case_data: dict[str, Any] | None = None
    validation: ExtractionValidation | None = None
    for attempt_index in range(max(0, max_validation_retries) + 1):
        extracted, usage = _call_extractor(
            extraction_client,
            selected_model,
            article_text,
            retry_instruction=retry_instruction,
        )
        case_data = normalize_extraction(
            extracted,
            case_id=case_id,
            article_text=article_text,
            model=selected_model,
            usage=usage,
        )
        repairs = repair_source_text_spans(case_data, article_text)
        validation = validate_extraction(case_data)
        attempts.append(
            {
                "attempt": attempt_index + 1,
                "usage": usage,
                "source_text_repairs": repairs,
                "errors": list(validation.errors),
                "warnings": list(validation.warnings),
            }
        )
        if validation.ok:
            break
        retry_instruction = "\n".join(validation.errors)
    if case_data is None or validation is None:
        raise ValueError("Extraction did not run")
    case_data["source_metadata"]["validation_attempts"] = attempts
    if not validation.ok:
        raise ValueError("Invalid extraction: " + "; ".join(validation.errors))
    case_data["extraction_warnings"] = list(validation.warnings)
    return case_data


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def nxml_to_article_text(path: str | Path) -> str:
    root = ET.parse(path).getroot()
    chunks: list[str] = []
    article_title = next((node for node in root.iter() if _local_name(node.tag) == "article-title"), None)
    if article_title is not None:
        title = _normalized_space("".join(article_title.itertext()))
        if title:
            chunks.append(f"[ARTICLE TITLE — METADATA ONLY]\n{title}")

    body = next((node for node in root.iter() if _local_name(node.tag) == "body"), None)
    if body is not None:
        for section in [node for node in body.iter() if _local_name(node.tag) == "sec"]:
            title_node = next(
                (child for child in section if _local_name(child.tag) == "title"),
                None,
            )
            section_title = _normalized_space("".join(title_node.itertext())) if title_node is not None else "Untitled"
            paragraphs = [
                _normalized_space("".join(node.itertext()))
                for node in section
                if _local_name(node.tag) == "p"
            ]
            section_text = "\n".join(value for value in paragraphs if value)
            if section_text:
                chunks.append(f"[SECTION: {section_title}]\n{section_text}")

    captions: list[str] = []
    for figure in [node for node in root.iter() if _local_name(node.tag) == "fig"]:
        figure_id = figure.attrib.get("id", "figure")
        caption = next((node for node in figure if _local_name(node.tag) == "caption"), None)
        if caption is not None:
            caption_text = _normalized_space("".join(caption.itertext()))
            if caption_text:
                captions.append(f"[FIGURE CAPTION: {figure_id}]\n{caption_text}")
    chunks.extend(captions)
    article_text = "\n\n".join(chunks)
    if not article_text:
        raise ValueError(f"No article text found in NXML: {path}")
    return article_text


def extract_from_nxml(
    path: str | Path,
    *,
    case_id: str | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    nxml_path = Path(path)
    article_text = nxml_to_article_text(nxml_path)
    case_data = extract_from_case_text(
        article_text,
        case_id=case_id or nxml_path.parent.name,
        client=client,
        model=model,
    )
    case_data["source_metadata"]["nxml_path"] = str(nxml_path)
    return case_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a dialogue-workflow case from PMC NXML")
    parser.add_argument("input", type=Path, help="PMC .nxml input path")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--model", default=None, help="Azure deployment; defaults to env or gpt-4.1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = extract_from_nxml(
        args.input,
        case_id=args.case_id,
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
