from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import (
    azure_client,
    extract_json_object,
    load_json,
    save_json,
    token_usage,
)
from .nxml_utils import find_nxmls, parse_nxml
from .prompts import EXTRACTION_OUTPUT_INSTRUCTIONS, EXTRACTION_SYSTEM_PROMPT
from .schemas import (
    ACQUISITION_MODES,
    AVAILABILITY_STATUSES,
    CLINICAL_PHASES,
    DISCLOSURE_POLICIES,
    EVIDENCE_ACTIONS,
    EVIDENCE_CATEGORIES,
    SCHEMA_VERSION,
    SOURCE_TYPES,
    TEMPORAL_ROLES,
    validate_case,
)


def extraction_source(parsed_nxml: dict[str, Any], max_chars: int) -> str:
    payload = {
        "article_title_internal_only": parsed_nxml.get("article_title", ""),
        "case_id": parsed_nxml.get("pmcid", ""),
        "figures": parsed_nxml.get("figures", []),
        "case_narrative_body_sections": parsed_nxml.get("body_sections", []),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED_FOR_MODEL_CONTEXT]"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def patient_specific_figure(figure: dict[str, Any]) -> bool:
    """Exclude literature diagrams/algorithms from patient-image coverage."""
    caption = str(figure.get("caption") or "").lower()
    non_patient_pattern = re.compile(
        r"\b(algorithm|flowchart|flow chart|schematic|adapted from|diagnostic pathway|"
        r"therapeutic pathway|review of|classification system)\b"
    )
    return not bool(non_patient_pattern.search(caption))


def inferred_acquisition_mode(category: str) -> str:
    if category in {"history", "symptom"}:
        return "history_recall"
    if category == "clinical_course":
        return "spontaneous_clinical_event"
    if category in {"physical_exam", "admission_observation"}:
        return "bedside_observation"
    if category in {"imaging_report", "pathology_report"}:
        return "requestable_report"
    if category in {"raw_image", "raw_pathology_or_microscopy_image"}:
        return "requestable_raw_image"
    if category in {"treatment_response", "outcome_context"}:
        return "treatment_or_outcome_only"
    return "requestable_test"


def inferred_allowed_actions(category: str, acquisition_mode: str) -> list[str]:
    actions: set[str] = set()
    if category in {"history", "symptom"}:
        actions.add("ASK_HISTORY")
    if category == "clinical_course":
        actions.update({"ASK_HISTORY", "WAIT_REASSESS"})
    if category in {"physical_exam", "admission_observation"}:
        actions.add("REQUEST_EXAM")
    if category == "admission_observation":
        actions.add("WAIT_REASSESS")
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
    if category in {"pathology_report", "procedure"}:
        actions.add("ORDER_PROCEDURE")
    if category in {"raw_image", "raw_pathology_or_microscopy_image"}:
        actions.add("REVIEW_IMAGE")
    if acquisition_mode in {"spontaneous_clinical_event", "treatment_or_outcome_only"}:
        actions.add("WAIT_REASSESS")
    if not actions:
        actions.add("REQUEST_RESULT")
    return sorted(actions)


def is_postdiagnostic_like(item: dict[str, Any]) -> bool:
    return (
        str(item.get("diagnostic_stage") or "") == "postdiagnostic_treatment_or_followup"
        or str(item.get("clinical_phase") or "") == "treatment_or_outcome"
        or str(item.get("acquisition_mode") or "") == "treatment_or_outcome_only"
        or str(item.get("category") or "") in {"treatment_response", "outcome_context"}
    )


def looks_like_molecular_raw_result(item: dict[str, Any]) -> bool:
    text = " ".join(
        [
            str(item.get("canonical_target") or ""),
            str(item.get("repeat_key") or ""),
            str(item.get("answer_text") or ""),
            str(item.get("source_quote") or ""),
            str((item.get("image_details") or {}).get("image_type") or ""),
            str((item.get("image_details") or {}).get("image_finding") or ""),
            str((item.get("image_details") or {}).get("report_finding") or ""),
        ]
    ).lower()
    return bool(re.search(r"\b(uba1|mutation|genetic|molecular|sequenc|genotype)\w*\b", text))


def normalized_repeat_key(item: dict[str, Any]) -> str:
    value = str(item.get("repeat_key") or item.get("canonical_target") or item.get("category") or "evidence")
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value[:80] or "evidence"


def inferred_temporal_role(item: dict[str, Any]) -> str:
    category = str(item.get("category") or "")
    phase = str(item.get("clinical_phase") or "")
    diagnostic_stage = str(item.get("diagnostic_stage") or "")
    if diagnostic_stage == "postdiagnostic_treatment_or_followup" or phase == "treatment_or_outcome":
        return "postdiagnostic_intervention_or_followup"
    if diagnostic_stage == "confirmatory" or phase == "confirmatory_testing":
        return "confirmatory_result"
    if category in {
        "routine_lab",
        "targeted_lab",
        "microbiology",
        "serology",
        "molecular",
        "genetic",
        "imaging_report",
        "pathology_report",
        "raw_image",
        "raw_pathology_or_microscopy_image",
        "procedure",
    }:
        return "targeted_result"
    if category == "clinical_course" or phase == "clinical_progression":
        return "interval_progression"
    if category == "admission_observation":
        return "admission_observation"
    if phase == "initial_presentation" and category in {"history", "symptom", "physical_exam"}:
        return "first_contact_visible"
    return "early_diagnostic_workup"


def postprocess_extraction(
    extracted: dict[str, Any],
    parsed_nxml: dict[str, Any],
    model: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    figure_by_id = {
        str(figure.get("figure_id") or ""): figure
        for figure in parsed_nxml.get("figures", [])
        if figure.get("figure_id")
    }

    evidence_items: list[dict[str, Any]] = []
    for index, raw in enumerate(_list(extracted.get("evidence_items")), start=1):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        item["evidence_id"] = f"E{index:03d}"

        if item.get("category") not in EVIDENCE_CATEGORIES:
            item["category"] = "history"
        if item.get("availability_status") not in AVAILABILITY_STATUSES:
            item["availability_status"] = "reported"
        if item.get("source_type") not in SOURCE_TYPES:
            item["source_type"] = "case_narrative"
        if item.get("disclosure_policy") not in DISCLOSURE_POLICIES:
            item["disclosure_policy"] = "return_if_requested"

        phase = str(item.get("clinical_phase") or "early_workup").strip()
        if phase not in CLINICAL_PHASES:
            phase = "early_workup"
        item["clinical_phase"] = phase
        item["phase_index"] = CLINICAL_PHASES[phase]
        temporal_role = str(item.get("temporal_role") or "").strip()
        if temporal_role == "first_contact_visible" and item["category"] not in {
            "history",
            "symptom",
            "physical_exam",
            "admission_observation",
        }:
            temporal_role = ""
        item["temporal_role"] = temporal_role if temporal_role in TEMPORAL_ROLES else inferred_temporal_role(item)
        item["time_anchor"] = str(item.get("time_anchor") or "").strip()
        mode = str(item.get("acquisition_mode") or "").strip()
        item["acquisition_mode"] = mode if mode in ACQUISITION_MODES else inferred_acquisition_mode(item["category"])
        allowed_actions = [
            str(value).strip().upper()
            for value in _list(item.get("allowed_actions"))
            if str(value).strip().upper() in EVIDENCE_ACTIONS
        ]
        item["repeat_key"] = normalized_repeat_key(item)
        confidence = str(item.get("order_confidence") or "medium").strip().lower()
        item["order_confidence"] = confidence if confidence in {"high", "medium", "low"} else "medium"

        item["figure_ids"] = [
            str(value).strip() for value in _list(item.get("figure_ids")) if str(value).strip()
        ]
        item.setdefault("canonical_target", "")
        item.setdefault("answer_text", "")
        item.setdefault("source_quote", "")
        item.setdefault("source_section", "")
        if item["category"] in {"raw_image", "raw_pathology_or_microscopy_image"}:
            image_details = item.get("image_details")
            if not isinstance(image_details, dict):
                image_details = {}
            item["image_details"] = {
                "image_type": str(image_details.get("image_type") or "").strip(),
                "image_finding": str(image_details.get("image_finding") or "").strip(),
                "report_finding": str(image_details.get("report_finding") or "").strip(),
                "diagnostic_significance_hidden": str(
                    image_details.get("diagnostic_significance_hidden") or ""
                ).strip(),
            }
            if not str(item.get("answer_text") or "").strip():
                item["answer_text"] = (
                    item["image_details"]["image_finding"]
                    or item["image_details"]["report_finding"]
                )
        else:
            item.pop("image_details", None)

        if not is_postdiagnostic_like(item):
            allowed_actions = sorted(
                set(allowed_actions).union(
                    inferred_allowed_actions(item["category"], item["acquisition_mode"])
                )
            )
        if item["category"] in {"raw_image", "raw_pathology_or_microscopy_image"} and looks_like_molecular_raw_result(item):
            allowed_actions = sorted(set(allowed_actions).union({"ORDER_TEST", "REQUEST_RESULT", "REVIEW_IMAGE"}))
        if (
            item["category"] == "imaging_report"
            and "ceus" in " ".join(
                [
                    str(item.get("canonical_target") or ""),
                    str(item.get("repeat_key") or ""),
                    str(item.get("answer_text") or ""),
                ]
            ).lower()
            and not is_postdiagnostic_like(item)
        ):
            allowed_actions = sorted(set(allowed_actions).union({"WAIT_REASSESS"}))
        item["allowed_actions"] = allowed_actions or inferred_allowed_actions(
            item["category"], item["acquisition_mode"]
        )
        if (
            item["disclosure_policy"] == "withhold_final_diagnosis_leakage"
            and item["category"] in {"raw_image", "raw_pathology_or_microscopy_image"}
            and str(item.get("answer_text") or "").strip()
            and not re.search(
                r"\b(diagnos(?:is|ed)|syndrome|establish(?:es|ed)|confirms definitive)\b",
                str(item.get("answer_text") or ""),
                re.IGNORECASE,
            )
        ):
            item["disclosure_policy"] = "return_if_specific_test_requested"

        image_paths: list[str] = []
        for figure_id in item["figure_ids"]:
            figure = figure_by_id.get(figure_id)
            if figure:
                image_paths.extend(figure.get("image_paths") or [])
        item["image_paths"] = list(dict.fromkeys(image_paths))
        if item["category"] not in {"raw_image", "raw_pathology_or_microscopy_image"}:
            item["image_paths"] = []
        evidence_items.append(item)

    represented_figures_before_fill = {
        figure_id
        for item in evidence_items
        if item.get("category") in {"raw_image", "raw_pathology_or_microscopy_image"}
        for figure_id in item.get("figure_ids") or []
    }
    for figure_id, figure in figure_by_id.items():
        if (
            figure_id in represented_figures_before_fill
            or not figure.get("image_paths")
            or not patient_specific_figure(figure)
        ):
            continue
        linked = next(
            (item for item in evidence_items if figure_id in (item.get("figure_ids") or [])),
            None,
        )
        if not linked:
            continue
        is_pathology = linked.get("category") == "pathology_report"
        category = "raw_pathology_or_microscopy_image" if is_pathology else "raw_image"
        repeat_base = str(linked.get("repeat_key") or figure_id.lower())
        caption = str(figure.get("caption") or "").strip()
        evidence_items.append(
            {
                "evidence_id": f"E{len(evidence_items) + 1:03d}",
                "category": category,
                "clinical_phase": linked.get("clinical_phase", "early_workup"),
                "phase_index": int(linked.get("phase_index", 1) or 1),
                "temporal_role": linked.get("temporal_role", inferred_temporal_role(linked)),
                "time_anchor": linked.get("time_anchor", ""),
                "acquisition_mode": "requestable_raw_image",
                "allowed_actions": ["REVIEW_IMAGE"],
                "repeat_key": f"{repeat_base}_raw_image",
                "order_confidence": linked.get("order_confidence", "medium"),
                "canonical_target": f"{linked.get('canonical_target') or figure_id} raw image",
                "availability_status": "reported",
                "answer_text": caption,
                "source_quote": caption,
                "source_section": "Figure caption",
                "source_type": "objective_figure_caption",
                "disclosure_policy": linked.get("disclosure_policy", "return_if_requested"),
                "figure_ids": [figure_id],
                "image_details": {
                    "image_type": "",
                    "image_finding": caption,
                    "report_finding": linked.get("answer_text", "") if is_pathology else "",
                    "diagnostic_significance_hidden": "",
                },
                "image_paths": list(figure.get("image_paths") or []),
            }
        )

    source_text = json.dumps(parsed_nxml.get("body_sections", []), ensure_ascii=False)
    biopsy_not_option = re.search(
        r"Because the nodule was subcapsular, the biopsy was not an option due to the high risk of hemorrhage",
        source_text,
        flags=re.IGNORECASE,
    )
    has_biopsy_not_performed = any(
        "biopsy" in str(item.get("canonical_target") or item.get("answer_text") or "").lower()
        and item.get("availability_status") == "not_performed"
        for item in evidence_items
    )
    if biopsy_not_option and not has_biopsy_not_performed:
        evidence_items.append(
            {
                "evidence_id": f"E{len(evidence_items) + 1:03d}",
                "category": "procedure",
                "clinical_phase": "targeted_workup",
                "phase_index": CLINICAL_PHASES["targeted_workup"],
                "time_anchor": "after CEUS assessment",
                "temporal_role": "targeted_result",
                "acquisition_mode": "requestable_report",
                "allowed_actions": ["ORDER_PROCEDURE", "REQUEST_RESULT"],
                "diagnostic_stage": "diagnostic_workup",
                "repeat_key": "liver_biopsy",
                "order_confidence": "high",
                "canonical_target": "liver biopsy feasibility",
                "availability_status": "not_performed",
                "answer_text": "Biopsy was not performed because the nodule was subcapsular and hemorrhage risk was high.",
                "source_quote": "Because the nodule was subcapsular, the biopsy was not an option due to the high risk of hemorrhage.",
                "source_section": "Imaging examinations",
                "source_type": "case_narrative",
                "disclosure_policy": "return_if_requested",
                "figure_ids": [],
                "image_paths": [],
            }
        )

    initial = extracted.get("initial_observation")
    if not isinstance(initial, dict):
        initial = {}
    initial = {
        "demographics": str(initial.get("demographics") or "").strip(),
        "encounter_context": str(initial.get("encounter_context") or "").strip(),
        "chief_complaint": str(initial.get("chief_complaint") or "").strip(),
        "first_contact_summary": str(
            initial.get("first_contact_summary") or initial.get("pre_referral_summary") or ""
        ).strip(),
        "pre_referral_summary": str(
            initial.get("pre_referral_summary") or initial.get("first_contact_summary") or ""
        ).strip(),
        "known_prior_history": [
            str(x).strip() for x in _list(initial.get("known_prior_history")) if str(x).strip()
        ],
        "presenting_symptoms": [str(x).strip() for x in _list(initial.get("presenting_symptoms")) if str(x).strip()],
        "initial_available_data": [
            str(x).strip() for x in _list(initial.get("initial_available_data")) if str(x).strip()
        ],
        "initial_context": str(initial.get("initial_context") or "").strip(),
        "source_quotes": [str(x).strip() for x in _list(initial.get("source_quotes")) if str(x).strip()],
    }
    if not initial["chief_complaint"] and initial["presenting_symptoms"]:
        initial["chief_complaint"] = "; ".join(initial["presenting_symptoms"][:2])
    if not initial["chief_complaint"] and initial["initial_context"]:
        initial["chief_complaint"] = initial["initial_context"].split(".", 1)[0].strip()

    hidden = extracted.get("hidden_reference")
    if not isinstance(hidden, dict):
        hidden = {}
    hidden.setdefault("case_level_diagnosis", "")
    for key in (
        "diagnosis_synonyms_or_variants",
        "diagnostic_reasoning_verbatim_spans",
        "confirmatory_evidence_targets",
        "missing_confirmatory_evidence",
        "evidence_limitations",
    ):
        hidden[key] = _list(hidden.get(key))

    result = {
        "schema_version": SCHEMA_VERSION,
        "case_id": parsed_nxml.get("pmcid") or extracted.get("case_id") or "",
        "source_metadata": {
            "pmcid": parsed_nxml.get("pmcid", ""),
            "article_title_hidden": parsed_nxml.get("article_title", ""),
            "nxml_path": parsed_nxml.get("nxml_path", ""),
            "image_folder": parsed_nxml.get("image_folder", ""),
            "extraction_model": model,
        },
        "initial_observation": initial,
        "budget": {"max_actions": 25},
        "figure_inventory": parsed_nxml.get("figures", []),
        "evidence_items": evidence_items,
        "hidden_reference": hidden,
        "source_audit": extracted.get("source_audit") if isinstance(extracted.get("source_audit"), dict) else {},
        "_extraction_diagnostics": {
            "created_at": datetime.now().isoformat(),
            "model": model,
            "token_usage": usage,
            "num_body_sections": len(parsed_nxml.get("body_sections", [])),
            "num_figures": len(parsed_nxml.get("figures", [])),
            "num_evidence_items": len(evidence_items),
        },
    }
    validation = validate_case(result)
    retrievable_figures = {
        figure.get("figure_id")
        for figure in parsed_nxml.get("figures", [])
        if figure.get("figure_id") and figure.get("image_paths") and patient_specific_figure(figure)
    }
    represented_figures = {
        figure_id
        for item in evidence_items
        if item.get("category") in {"raw_image", "raw_pathology_or_microscopy_image"}
        for figure_id in item.get("figure_ids") or []
    }
    confirmatory_targets = hidden.get("confirmatory_evidence_targets") or []
    completeness_warnings: list[str] = []
    if len(evidence_items) < 10:
        completeness_warnings.append("fewer than 10 evidence items; likely incomplete case extraction")
    missing_figures = sorted(retrievable_figures - represented_figures)
    if missing_figures:
        completeness_warnings.append(f"retrievable figures missing raw-image evidence: {', '.join(missing_figures)}")
    if hidden.get("final_diagnosis_verbatim") and not confirmatory_targets:
        completeness_warnings.append("final diagnosis exists but confirmatory_evidence_targets is empty")
    completeness_ok = not completeness_warnings
    result["_extraction_diagnostics"].update(
        {
            "validation_ok": validation.ok,
            "validation_warnings": validation.warnings + completeness_warnings,
            "validation_errors": validation.errors,
            "completeness_ok": completeness_ok,
            "completeness_warnings": completeness_warnings,
        }
    )
    return result


class InteractiveCaseExtractor:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int = 16000,
        timeout: int = 600,
        max_source_chars: int = 100000,
    ) -> None:
        self.client = azure_client(timeout)
        self.model = (
            model
            or os.getenv("AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT")
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4")
            or "gpt-4.1"
        )
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.max_source_chars = max_source_chars

    async def convert(self, nxml_path: Path) -> dict[str, Any]:
        parsed_nxml = parse_nxml(nxml_path)
        prompt = (
            f"{EXTRACTION_OUTPUT_INSTRUCTIONS}\n\n"
            "Parsed PMC source follows. Article title is internal extraction context only and must never appear "
            "in initial_observation.\n\n"
            f"{extraction_source(parsed_nxml, self.max_source_chars)}"
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = await self.client.chat.completions.create(**kwargs)
        raw = (response.choices[0].message.content or "").strip()
        extracted = extract_json_object(raw)
        return postprocess_extraction(extracted, parsed_nxml, self.model, token_usage(response))


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Extract interactive diagnosis evidence stores from raw PMC NXML.")
    parser.add_argument("--input_dir", default="../internalmed")
    parser.add_argument("--output_dir", default="data/pmc_interactive_extraction/internalmed")
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--max_tokens", type=int, default=16000)
    parser.add_argument("--max_source_chars", type=int, default=100000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh_postprocess", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    nxmls = find_nxmls(input_dir)
    if args.case_ids:
        selected = {value.strip() for item in args.case_ids for value in item.split(",") if value.strip()}
        nxmls = [path for path in nxmls if path.parent.name in selected]
    if args.limit is not None:
        nxmls = nxmls[: args.limit]

    extractor = InteractiveCaseExtractor(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_source_chars=args.max_source_chars,
    )
    cases = []
    for path in nxmls:
        output_path = output_dir / f"{path.parent.name}.json"
        if args.refresh_postprocess:
            if not output_path.exists():
                cases.append({"case_id": path.parent.name, "status": "missing_existing_output"})
                continue
            existing = load_json(output_path)
            parsed_nxml = parse_nxml(path)
            old_diagnostics = existing.get("_extraction_diagnostics") or {}
            result = postprocess_extraction(
                existing,
                parsed_nxml,
                str((existing.get("source_metadata") or {}).get("extraction_model") or extractor.model),
                old_diagnostics.get("token_usage") or {},
            )
            save_json(output_path, result)
            diagnostics = result["_extraction_diagnostics"]
            status = "postprocess_refreshed" if diagnostics["validation_ok"] and diagnostics["completeness_ok"] else "audit_required"
            print(f"[extract] {path.parent.name}: {status}", flush=True)
            cases.append({"case_id": path.parent.name, "status": status, "output_path": str(output_path)})
            continue
        if output_path.exists() and not args.overwrite:
            print(f"[extract] {path.parent.name}: skipped existing", flush=True)
            cases.append({"case_id": path.parent.name, "status": "skipped_existing", "output_path": str(output_path)})
            continue
        print(f"[extract] {path.parent.name}: extracting", flush=True)
        try:
            result = await extractor.convert(path)
            save_json(output_path, result)
            diagnostics = result["_extraction_diagnostics"]
            status = "completed" if diagnostics["validation_ok"] and diagnostics["completeness_ok"] else "audit_required"
        except Exception as exc:
            status = "failed"
            save_json(output_path.with_suffix(".failed.json"), {"case_id": path.parent.name, "error": str(exc)})
        print(f"[extract] {path.parent.name}: {status}", flush=True)
        cases.append({"case_id": path.parent.name, "status": status, "output_path": str(output_path)})

    case_index = {str(item.get("case_id")): item for item in cases}
    for existing in sorted(output_dir.glob("*.json")):
        if existing.name.startswith("_") or existing.name.endswith(".failed.json"):
            continue
        case_index.setdefault(
            existing.stem,
            {
                "case_id": existing.stem,
                "status": "existing_completed",
                "output_path": str(existing),
            },
        )
    manifest_cases = [case_index[key] for key in sorted(case_index)]

    save_json(
        output_dir / "_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "schema_version": SCHEMA_VERSION,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "model": extractor.model,
            "num_nxml_cases_found": len(find_nxmls(input_dir)),
            "cases": manifest_cases,
        },
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
