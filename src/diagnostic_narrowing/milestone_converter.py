from __future__ import annotations

import argparse
import json
import os
import re
from html import unescape
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI

from diagnostic_narrowing.schemas import (
    SCHEMA_VERSION,
    EXPECTED_NEXT_EVIDENCE_CATEGORY_VALUES,
    EXPECTED_READINESS_VALUES,
    MILESTONE_FUNCTION_VALUES,
    extract_json_object,
    normalize_whitespace,
    validate_milestone_case,
    normalize_guardrail_flags,
)


load_dotenv()


MILESTONE_CONVERSION_PROMPT = """
You are converting a PMC case report into a source-grounded diagnostic evidence
trajectory for a multimodal diagnostic narrowing benchmark.

Return only valid JSON. Do not include markdown.

Core task:
Construct source-order diagnostic narrowing dialogue turns from the case report.
Do this in two steps:

Step 1: Extract source-ordered evidence events from the case narrative.
An evidence event is a verbatim or minimally reorganized source-supported span
that reports a clinically relevant piece or group of evidence, such as symptoms,
physical findings, grouped routine labs, imaging findings, pathology findings,
microbiology, IHC, genetics, treatment response, or outcome.
Keep evidence_events moderately granular. Usually extract 8-16 high-signal
events, not every minor lab value, date, treatment detail, or sentence. One event may
represent a compact group of related source findings, such as routine labs
together, imaging findings together, or pathology/IHC findings together. Each
source_span must be short enough for manual audit, preferably one compact source
sentence or phrase. Do not copy whole paragraphs into source_span.

Step 2: Group adjacent or diagnostically related evidence events into
clinically meaningful evidence milestones. A milestone should not be one turn
per individual test, lab, or image. It is a multi-evidence bundle that can
contain different labs, physical findings, images, procedures, pathology, and
special tests when they appear together in the article's evidence progression.
Aim for 6-9 included milestones when possible and about 10 maximum. Prefer more
turns when a case has long evolving evidence. Do not collapse a long clinical
course, major systemic progression, ICU course, vascular/pathology findings,
bone marrow findings, and confirmatory tests into one or two dense turns.

Granularity rule:
Split dense milestones so each turn represents one clinically interpretable
step in the narrowing process. For example, a VEXAS-like case should usually be
split into steps such as:
- initial presentation;
- Sweet syndrome / early inflammatory workup / negative infection or malignancy
  screen;
- multisystem progression such as ear/larynx/eye/neurologic/muscle symptoms;
- recurrent thrombosis or ICU transition when diagnostically relevant;
- skin/vascular findings and cryoglobulinemia;
- cytopenias and bone marrow examination with marrow figures attached;
- confirmatory genetic or molecular result;
- treatment/outcome only if relevant.

Design note:
This is DDX-TRACE-inspired in evaluating diagnostic trajectory and premature
closure, MediQ-inspired in requiring abstention/uncertainty when evidence is
insufficient, and AMIE-inspired in maintaining a state-aware diagnostic
dialogue. It is adapted to fixed source-derived PMC milestones, not free patient
interviews or a true clinical workup timeline.

Image placement rule:
Do not place all figures into one generic early image milestone. Images are not
a fixed second-stage evidence block. Assign each figure or image group to the
milestone where it functions diagnostically in the source-derived evidence
trajectory. For example:
- CT, MRI, ultrasound, endoscopy, or gross visual findings that reveal an
  important abnormality may belong in major_visual_evidence or
  targeted_investigation.
- Pathology, cytology, microscopy, bone marrow, or histology figures may belong
  in pathology_or_microscopy.
- IHC, sequencing chromatograms, microbiology images, disease-specific tests, or
  molecular figures may belong in confirmatory_evidence if they confirm or
  strongly reveal the final diagnosis.
- Follow-up images belong in treatment_or_outcome_if_relevant only when they
  contribute to diagnostic reasoning.
Group multiple images only when they share the same modality/stage and
diagnostic role. Split them across milestones when they represent different
diagnostic stages or confirmation levels.

Do not claim to reconstruct the true clinical timeline. Use source-derived
diagnostic progression, evidence-disclosure sequence, or diagnostic evidence
trajectory. Record order_basis and order_confidence.

Ordering rule:
Do not use section headings alone as turns. Use evidence events inside case
narrative sections. Preserve article narrative order unless explicit
temporal/procedural cues justify another local ordering. If the article lists
many tests without clear chronology, group them by source adjacency and
diagnostic function, then mark order_basis=clinical_stage_normalization and
order_confidence=low or medium. Figure order can support image matching but
should not force all images into one turn.

Source-grounding rules:
- Every non-empty field must be supported by source text.
- Do not infer unsupported medical content.
- Do not generate new final diagnosis or diagnostic reasoning in extraction.
- Keep exact source spans or minimally reorganized source-supported text.
- Preserve temporal/procedural cues such as initially, on admission,
  subsequently, after biopsy, hospital day, or months earlier.
- Prefer case narrative sections: Case Presentation, Case Report, Clinical
  Course, Patient Presentation, Investigations, Diagnostic Assessment,
  Treatment and Outcome.
- Avoid abstract, introduction, discussion, and conclusion as the main source
  for visible evidence ordering because they may summarize retrospectively.
- Separate the evidence shown in dialogue user turns from hidden image references.
  visible_evidence is the complete source-derived evidence bundle for audit and
  internal reference generation. dialogue_visible_evidence is the
  leakage-controlled evidence shown in both test-model and saved reference
  dialogue user turns.
- Keep visible_evidence and dialogue_visible_evidence concise. They should be
  compact clinical disclosure text, not paragraph-length case summaries. Usually
  1-4 short sentences per milestone is enough.
- Keep hidden_reference.source_evidence_items concise, usually 1-4 items per
  milestone. Put image-specific findings in hidden_reference.image_finding rather
  than repeating them in long lists.
- If raw images are attached to a milestone, dialogue_visible_evidence should give
  clinical context and non-image evidence needed to interpret the image, but it
  should not state the image findings that the test model is supposed to infer
  from the raw pixels. For example, do not put "bone marrow aspirate showed
  vacuoles" in dialogue_visible_evidence when marrow images are attached; put that
  in hidden_reference.image_finding or source_evidence_items instead.
- This rule applies even when the article narrative, not only the caption,
  states the image/pathology/microscopy/radiology/genetic-figure finding. In
  dialogue_visible_evidence, say that the test was performed and attach the
  figure, but omit the visual result. Examples:
  - Use: "Bone marrow aspirate was obtained (Figs. 3, 4)."
    Do not use: "Bone marrow aspirate showed vacuoles in myeloid and erythroid
    precursors."
  - Use: "Skin biopsy was performed (Fig. 1)."
    Do not use: "Skin biopsy confirmed Sweet's syndrome."
  - Use: "Genetic sequencing result is attached (Fig. 5)."
    Do not use: "UBA1 p.Met41Val mutation was detected" unless this is a
    non-image laboratory result being disclosed independently of the figure.
- If an image-related fact is itself a non-visual test result already disclosed
  in the clinical narrative, keep only the non-visual/actionable context in
  dialogue_visible_evidence and move the visual description to hidden_reference.
- Figure captions may be used for hidden image reference findings, image type,
  and diagnostic significance, but caption-derived diagnostic interpretation
  should not be exposed in dialogue_visible_evidence.
- Keep final diagnosis and article diagnostic reasoning only in hidden_reference.
- Mark diagnosis-revealing results as diagnosis_revealing=true and usually place
  them in later confirmatory milestones.
- Extraction should not generate new DDx or new clinical reasoning. DDx updates
  are generated later during dialogue generation.
- Treatment or outcome evidence may be retained, but it should usually be marked
  include_in_core_narrowing_metrics=false unless it introduces new diagnostic
  evidence. It should not be treated as a required core narrowing turn after a
  confirmatory diagnosis has already been disclosed.
- Case-level hidden_reference is the final evaluation reference. Extract it from
  article-reported final diagnosis and source-supported diagnostic reasoning.
  Keep final_diagnosis_verbatim and diagnostic_reasoning_verbatim_spans as close
  to the source wording as possible. diagnostic_reasoning_items may itemize those
  source spans into concise evidence items, but must not add new reasoning.
- Do not use case-level final diagnosis or final diagnostic reasoning to write
  intermediate milestone fields. Intermediate milestones may only use evidence
  available at that source-order point and image-specific hidden references for
  attached figures.
- Preserve source-stated uncertainty and missing evidence. Extract statements
  such as "not available", "unavailable", "not obtained", "not reported",
  "could not be performed", "suspected", "presumptive", "highly suspected",
  "consistent with", "no confirmatory test", or "unable to calculate ratio".
  Put unavailable or missing diagnostic tests in missing_or_unavailable_tests,
  not just free text. Do not convert an article-reported assumption into a
  biochemically/pathologically/molecularly confirmed diagnosis unless the
  confirming evidence is explicitly present in the case report.
- Preserve negative medication and exposure evidence when diagnostically
  relevant, especially when it rules out common causes. For hypoglycemia cases,
  keep source statements about glucose-lowering medications, herbal medicines,
  alcohol use, sulfonylurea/drug screens if reported, and whether such tests
  were not reported.
- Preserve supportive management evidence that affects diagnostic certainty,
  even when it is not a core narrowing step. For example, if hypoglycemia
  persists despite glucose infusion or frequent meals and resolves after tumor
  resection, extract both the supportive treatment and the outcome.

Diagnosis-revealing examples:
genetic mutation, IHC profile, microbiological identification, disease-specific
antibody/serology, culture identification, molecular markers, pathology
confirmation, "diagnostic of", "confirmed by", "consistent with final diagnosis".

Allowed stage labels:
- initial_presentation
- initial_workup
- major_visual_evidence
- targeted_investigation
- pathology_or_microscopy
- confirmatory_evidence
- treatment_or_outcome_if_relevant
- final_synthesis

Allowed order_basis:
- explicit_temporal_cue
- case_narrative_order
- clinical_stage_normalization
- figure_order
- uncertain

Allowed order_confidence:
- high
- medium
- low

Allowed diagnostic_role:
- initial_context
- broad_narrowing
- major_narrowing
- rule_out_alternative
- confirmatory
- treatment_response
- outcome
- clinical_complication
- care_transition
- management_context
- final_synthesis_support

Allowed expected_next_evidence_category:
- imaging
- pathology
- microbiology
- genetic
- serology
- molecular
- routine_lab
- additional_clinical_context
- final_synthesis
- none
- uncertain

Allowed expected_readiness:
- not_ready
- possibly_ready
- ready

Allowed milestone_function:
- broaden_context
- narrow_differential
- localize_disease
- characterize_abnormality
- rule_out_alternative
- confirm_diagnosis
- support_final_synthesis
- treatment_response
- outcome_context
- clinical_complication
- care_transition
- management_context

Field guidance:
- expected_next_evidence_category should be coarse-grained, not an exact test
  match.
- If the next source-derived milestone is pathology, IHC, or biopsy, use
  pathology.
- If the next source-derived milestone is sequencing, mutation testing, or
  molecular testing, use genetic or molecular as appropriate.
- If the current milestone contains diagnosis-revealing evidence and should
  support final synthesis, expected_readiness should usually be ready or
  possibly_ready.
- If the current milestone only provides early history, routine labs, or broad
  visual context, expected_readiness should usually be not_ready.
- milestone_function should describe how the milestone contributes to narrowing,
  not just what data type it contains.
- include_in_core_narrowing_metrics should be true for turns that materially
  narrow the diagnosis, and false for supportive treatment/outcome context after
  diagnostic confirmation.
- question should make the turn an explicit clinical VQA-style task. It should
  ask the model to interpret the newly disclosed evidence, update the
  differential diagnosis, and state uncertainty or next evidence needs. If raw
  images are attached, ask for medically relevant visible image findings and how
  they change the diagnostic state. Do not reveal hidden final diagnosis,
  caption-derived diagnostic labels, or future evidence in the question.
- hidden_reference.image_finding should contain article-derived image-specific
  findings for attached figures when available. If no source-derived
  image-specific finding is available, explicitly say that no source-derived
  image-specific finding was available rather than inventing one.
- missing_or_unavailable_tests should contain source-stated unavailable,
  missing, not-performed, not-reported, or clinically expected confirmatory
  tests only when directly supported by the article text. Do not invent missing
  tests from general medical knowledge during extraction.
- diagnostic_certainty should describe how strong the article-supported
  diagnosis is: e.g. "confirmed_by_pathology", "confirmed_by_genetics",
  "confirmed_by_microbiology", "clinically_inferred_not_biochemically_confirmed",
  "article_reported_presumptive", or "uncertain".
- evidence_limitations should explicitly state source-supported limitations,
  such as "IGF-2 assay unavailable" or "initial biopsy inconclusive".
- If a final diagnosis is article-reported but key confirmatory evidence is
  unavailable, set confirmatory_evidence_present=false and preserve the
  uncertainty in final_diagnosis_certainty and evidence_limitations.
- guardrail_flags are sparse review signals, not a second annotation task. Use
  only the allowed values and only when directly supported by the source:
  source_statement_conflict when two source passages conflict about the same
  diagnosis, staging, test result, or outcome; missing_confirmatory_evidence
  when the source explicitly states that an important confirming test is absent
  or unavailable; image_text_inconsistency when source narrative and figure
  text conflict. Otherwise return an empty list. Do not infer clinical errors
  from general medical knowledge during extraction.
- question_type should be one of:
  clinical_diagnostic_update, image_interpretation_and_diagnostic_update,
  pathology_or_microscopy_update, confirmatory_diagnostic_update,
  treatment_or_outcome_update.

Return this JSON shape:
{
  "schema_version": "2026-07-06-diagnostic-narrowing-v1",
  "case_id": "",
  "source_metadata": {
    "pmcid": "",
    "title": "",
    "nxml_path": "",
    "image_folder": "",
    "extraction_model": "",
    "notes": ""
  },
  "evidence_events": [
    {
      "event_id": "E0",
      "source_order": 0,
      "source_section": "",
      "source_span": "",
      "evidence_type": "history | physical_exam | routine_lab | imaging | pathology | microbiology | genetic | serology | molecular | procedure | treatment_response | outcome | other",
      "temporal_cue": "",
      "procedural_cue": "",
      "diagnostic_role": "initial_context",
      "diagnosis_revealing": false,
      "linked_figures": [],
      "hidden_reference_note": ""
    }
  ],
  "evidence_milestones": [
    {
      "turn_id": "T0",
      "included_event_ids": ["E0"],
      "grouping_reason": "Initial source-adjacent presentation evidence.",
      "stage": "initial_presentation",
      "display_name": "Initial presentation",
      "visible_evidence": "",
      "dialogue_visible_evidence": "",
      "test_visible_evidence": "",
      "evidence_types": ["history", "physical_exam"],
      "source_section": "",
      "source_span": "",
      "source_position": null,
      "temporal_cue": "",
      "order_basis": "case_narrative_order",
      "order_confidence": "medium",
      "diagnostic_role": "initial_context",
      "diagnosis_revealing": false,
      "expected_next_evidence_category": "imaging",
      "expected_readiness": "not_ready",
      "milestone_function": "broaden_context",
      "include_in_core_narrowing_metrics": true,
      "question": "Based on the newly disclosed source-derived clinical evidence, what are the leading diagnostic hypotheses at this stage, how should the differential diagnosis be updated, and what additional evidence would help narrow the diagnosis?",
      "question_type": "clinical_diagnostic_update",
      "linked_figures": [],
      "image_paths": [],
      "image_metadata": [],
      "hidden_reference": {
        "source_evidence_items": [],
        "image_finding": "",
        "diagnostic_significance": "",
        "evidence_explanation": ""
      },
      "missing_or_unavailable_tests": [
        {
          "test": "",
          "status": "unavailable | not_performed | not_reported | inconclusive",
          "source_span": "",
          "diagnostic_impact": ""
        }
      ],
      "diagnostic_certainty": "",
      "evidence_limitations": [],
      "guardrail_flags": [],
      "leakage_notes": "",
      "include_in_dialogue": true
    }
  ],
  "hidden_reference": {
    "final_diagnosis_verbatim": "",
    "diagnostic_reasoning_verbatim_spans": [],
    "diagnostic_reasoning_items": [],
    "final_diagnosis": "",
    "diagnostic_reasoning": "",
    "article_derived_evidence_summary": "",
    "diagnosis_synonyms_or_variants": [],
    "final_diagnosis_certainty": "",
    "confirmatory_evidence_present": true,
    "missing_confirmatory_evidence": [],
    "evidence_limitations": []
  },
  "guardrail_flags": [],
  "quality_flags": {
    "has_parseable_nxml": true,
    "has_retrievable_image": true,
    "has_final_diagnosis": true,
    "has_sufficient_evidence_sequence": true,
    "needs_manual_review": false,
    "review_notes": ""
  }
}

Case report NXML text:
{case_report}
"""


def read_text_limited(path: Path, limit_chars: int = 70000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit_chars:
        return text[:limit_chars] + "\n\n[TRUNCATED_FOR_EXTRACTION]"
    return text


def plain_text_from_nxml(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def source_stated_tnm_stages(nxml_text: str) -> list[str]:
    """Return distinct source-reported TNM-like stage strings in source order."""
    text = plain_text_from_nxml(nxml_text)
    pattern = re.compile(
        r"\b(p?T\d+[a-c]?)\s+(N(?:[0-3][a-c]?|x))\s+(M(?:[0-1][a-c]?|x))"
        r"(?:\s+(L(?:[0-1]|x)))?(?:\s+(V(?:[0-2]|x)))?(?:\s+(R(?:[0-2]|x)))?\b",
        re.IGNORECASE,
    )
    stages: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        stage = " ".join(value.upper() for value in match.groups() if value)
        if stage not in seen:
            seen.add(stage)
            stages.append(stage)
    return stages


def source_stated_limitation_spans(nxml_text: str, max_items: int = 12) -> list[dict[str, str]]:
    """Find source-stated missing/unavailable/inconclusive diagnostic evidence.

    This is a conservative safety net for cases where the model extractor misses
    clinically important limitations such as an unavailable confirmatory assay.
    """
    text = plain_text_from_nxml(nxml_text)
    sentence_candidates = re.split(r"(?<=[.!?])\s+", text)
    limitation_pattern = re.compile(
        r"\b(unavailable|not available|could not be obtained|unable to|not obtained|not performed|"
        r"not reported|unable to calculate|could not calculate|no confirmatory|not confirmed)\b",
        re.IGNORECASE,
    )
    diagnostic_pattern = re.compile(
        r"\b(assay|bioassay|test|testing|determination|ratio|culture|stain|biopsy|cytology|"
        r"histolog|patholog|immuno|genetic|molecular|sequenc|antibody|antibodies|IGF[- ]?2|"
        r"IGF[- ]?1|C[- ]?peptide|insulin|sulfonylurea|sulphonylurea)\b",
        re.IGNORECASE,
    )
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for sentence in sentence_candidates:
        sentence = normalize_whitespace(sentence)
        if not sentence or len(sentence) > 700:
            continue
        if not limitation_pattern.search(sentence):
            continue
        if not diagnostic_pattern.search(sentence):
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        test_match = diagnostic_pattern.search(sentence)
        status_match = limitation_pattern.search(sentence)
        status_text = normalize_whitespace(status_match.group(0)).lower() if status_match else "unknown"
        sentence_lower = sentence.lower()
        if (
            "unavailable" in sentence_lower
            or "unavailability" in sentence_lower
            or "not available" in sentence_lower
            or "could not be obtained" in sentence_lower
        ):
            status = "unavailable"
        elif status_text in {"not obtained", "not performed"}:
            status = "not_performed"
        elif status_text == "not reported":
            status = "not_reported"
        elif status_text == "inconclusive":
            status = "inconclusive"
        else:
            status = "unknown"
        items.append(
            {
                "test": normalize_whitespace(test_match.group(0)) if test_match else "",
                "status": status,
                "source_span": sentence,
                "diagnostic_impact": "Source-stated diagnostic limitation; preserve uncertainty and do not overstate confirmation.",
            }
        )
        if len(items) >= max_items:
            break
    return items


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def iter_case_nxmls(input_dir: Path) -> list[Path]:
    paths = []
    for case_dir in sorted(input_dir.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("PMC"):
            continue
        nxmls = sorted(case_dir.glob("*.nxml"))
        if nxmls:
            paths.append(nxmls[0])
    return paths


def stage_display_name(stage: str, fallback: str) -> str:
    value = normalize_whitespace(stage).replace("_", " ")
    return value.title() if value else fallback


def default_milestone_function(diagnostic_role: str) -> str:
    mapping = {
        "initial_context": "broaden_context",
        "broad_narrowing": "narrow_differential",
        "major_narrowing": "narrow_differential",
        "confirmatory": "confirm_diagnosis",
        "final_synthesis_support": "support_final_synthesis",
        "treatment_response": "treatment_response",
        "outcome": "outcome_context",
        "clinical_complication": "clinical_complication",
        "care_transition": "care_transition",
        "management_context": "management_context",
    }
    return mapping.get(normalize_whitespace(diagnostic_role), "narrow_differential")


def is_non_core_context_turn(milestone: dict[str, Any]) -> bool:
    text = " ".join(
        normalize_whitespace(milestone.get(key)).lower()
        for key in ("display_name", "visible_evidence", "dialogue_visible_evidence", "source_span", "grouping_reason")
    )
    role = normalize_whitespace(milestone.get("diagnostic_role"))
    function = normalize_whitespace(milestone.get("milestone_function"))
    if role in {"clinical_complication", "care_transition", "management_context", "treatment_response", "outcome"}:
        return True
    if function in {"clinical_complication", "care_transition", "management_context", "treatment_response", "outcome_context"}:
        return True
    if role in {"broad_narrowing", "major_narrowing", "rule_out_alternative", "confirmatory"}:
        return False
    if function in {
        "narrow_differential",
        "localize_disease",
        "characterize_abnormality",
        "rule_out_alternative",
        "confirm_diagnosis",
    }:
        return False
    complication_terms = ("cardiac arrest", "icu", "intubation", "cpr", "confusion")
    management_terms = ("surgical resection", "resection was performed", "operation", "surgery")
    if any(term in text for term in complication_terms):
        return True
    if any(term in text for term in management_terms) and not any(term in text for term in ("histolog", "patholog", "immunohistochem")):
        return True
    return False


def default_question_type(milestone: dict[str, Any]) -> str:
    stage = normalize_whitespace(milestone.get("stage"))
    evidence_types = {normalize_whitespace(value) for value in milestone.get("evidence_types") or []}
    has_image = bool(milestone.get("linked_figures") or milestone.get("image_paths") or milestone.get("image_metadata"))
    if stage == "confirmatory_evidence" or milestone.get("diagnosis_revealing"):
        return "confirmatory_diagnostic_update"
    if stage == "pathology_or_microscopy" or evidence_types.intersection({"pathology", "cytology"}):
        return "pathology_or_microscopy_update"
    if has_image or evidence_types.intersection({"imaging"}):
        return "image_interpretation_and_diagnostic_update"
    if stage == "treatment_or_outcome_if_relevant":
        return "treatment_or_outcome_update"
    return "clinical_diagnostic_update"


def effective_question_type(milestone: dict[str, Any]) -> str:
    provided = normalize_whitespace(milestone.get("question_type"))
    inferred = default_question_type(milestone)
    if not provided:
        return inferred
    if provided == "clinical_diagnostic_update" and inferred != "clinical_diagnostic_update":
        return inferred
    return provided


def default_question(milestone: dict[str, Any]) -> str:
    question_type = effective_question_type(milestone)
    if question_type == "image_interpretation_and_diagnostic_update":
        return (
            "Review the attached image(s) in the current clinical context. What findings are directly visible, "
            "how do they change the working differential diagnosis, and what should be checked next?"
        )
    if question_type == "pathology_or_microscopy_update":
        return (
            "Review the pathology or microscopy material in context. What source-visible or pixel-visible findings "
            "are diagnostically important, how should the working differential change, and what uncertainty remains?"
        )
    if question_type == "confirmatory_diagnostic_update":
        return (
            "Interpret the newly disclosed diagnostic test results in context. Which diagnoses are now better or "
            "less supported, what alternatives remain, and what uncertainty is still present?"
        )
    if question_type == "treatment_or_outcome_update":
        return (
            "Interpret the newly disclosed treatment response or outcome in context. Does it change or support the "
            "current diagnostic assessment, and what uncertainty remains?"
        )
    return (
        "Given the newly disclosed clinical evidence, what is your working differential at this point, what moved "
        "up or down, and what evidence would most help narrow the diagnosis?"
    )


def graphic_href_order(nxml_text: str) -> list[str]:
    hrefs = re.findall(r"(?:xlink:href|href)=[\"']([^\"']+)[\"']", nxml_text)
    seen = set()
    ordered = []
    for href in hrefs:
        base = Path(href).stem
        if base and base not in seen:
            seen.add(base)
            ordered.append(base)
    return ordered


def local_image_index(case_folder: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in case_folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"}:
            continue
        keys = {path.stem.lower(), path.name.lower()}
        for key in keys:
            index.setdefault(key, path)
    return index


def candidate_figure_keys(milestone: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("linked_figures", "image_paths"):
        raw = milestone.get(key)
        if isinstance(raw, list):
            values.extend(str(item) for item in raw if item)
        elif raw:
            values.append(str(raw))
    for meta in milestone.get("image_metadata") or []:
        if isinstance(meta, dict):
            for key in ("image_id", "image_name", "href", "file", "filename", "local_path"):
                if meta.get(key):
                    values.append(str(meta[key]))
    normalized = []
    for value in values:
        stem = Path(value).stem.lower()
        normalized.append(stem)
        normalized.append(value.lower())
        fig_match = re.search(r"fig(?:ure)?\.?\s*([0-9]+)", value, re.IGNORECASE)
        if fig_match:
            normalized.append(f"fig{fig_match.group(1)}")
    seen = set()
    out = []
    for value in normalized:
        value = normalize_whitespace(value).lower()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def image_path_matches_figure_keys(path: Path, figure_keys: list[str]) -> bool:
    if not figure_keys:
        return True
    path_tokens = {
        normalize_whitespace(path.name).lower().replace("figure", "fig"),
        normalize_whitespace(path.stem).lower().replace("figure", "fig"),
    }
    path_tokens = {"".join(ch for ch in token if ch.isalnum()) for token in path_tokens}
    normalized_keys = [
        "".join(ch for ch in key.lower().replace("figure", "fig") if ch.isalnum())
        for key in figure_keys
    ]
    return any(key and key in token for key in normalized_keys for token in path_tokens)


def resolve_milestone_image_paths(parsed: dict[str, Any], nxml_path: Path, nxml_text: str) -> list[str]:
    warnings: list[str] = []
    case_folder = nxml_path.parent
    images = local_image_index(case_folder)
    hrefs = graphic_href_order(nxml_text)
    href_by_figure_number = {f"fig{idx}": href for idx, href in enumerate(hrefs, start=1)}

    for milestone in parsed.get("evidence_milestones") or []:
        if not isinstance(milestone, dict):
            continue
        resolved: list[str] = []
        figure_keys = candidate_figure_keys(milestone)
        for key in figure_keys:
            image_path = images.get(key)
            if image_path is None and key in href_by_figure_number:
                image_path = images.get(href_by_figure_number[key].lower())
            if image_path is None:
                for image_key, candidate in images.items():
                    if key and (key in image_key or image_key in key):
                        image_path = candidate
                        break
            if image_path and image_path_matches_figure_keys(image_path, figure_keys) and str(image_path) not in resolved:
                resolved.append(str(image_path))

        original_paths = [str(p) for p in milestone.get("image_paths") or []]
        existing_originals = [
            str(Path(p))
            for p in original_paths
            if Path(p).exists() and image_path_matches_figure_keys(Path(p), figure_keys)
        ]
        for path in existing_originals:
            if path not in resolved:
                resolved.append(path)
        milestone["image_paths"] = resolved
        if (milestone.get("linked_figures") or milestone.get("image_metadata")) and not resolved:
            note = "Image resolver did not find existing local image paths for this milestone."
            warnings.append(f"{milestone.get('turn_id', '')}: {note}")
            leakage = normalize_whitespace(milestone.get("leakage_notes"))
            milestone["leakage_notes"] = f"{leakage} {note}".strip()
    return warnings


def postprocess_milestone_case(parsed: dict[str, Any], nxml_path: Path, nxml_text: str) -> list[str]:
    warnings: list[str] = []
    parsed["guardrail_flags"] = normalize_guardrail_flags(parsed.get("guardrail_flags"))
    source_tnm_stages = source_stated_tnm_stages(nxml_text)
    if len(source_tnm_stages) > 1:
        if "source_statement_conflict" not in parsed["guardrail_flags"]:
            parsed["guardrail_flags"].append("source_statement_conflict")
        warnings.append(
            "source_statement_conflict: distinct source-reported TNM stages: "
            + " | ".join(source_tnm_stages)
        )
    hidden_reference = parsed.get("hidden_reference")
    if isinstance(hidden_reference, dict):
        hidden_reference.setdefault("final_diagnosis_certainty", "")
        hidden_reference.setdefault("confirmatory_evidence_present", None)
        hidden_reference.setdefault("missing_confirmatory_evidence", [])
        hidden_reference.setdefault("evidence_limitations", [])
        auto_limitations = source_stated_limitation_spans(nxml_text)
        if auto_limitations:
            existing_spans = {
                normalize_whitespace(item.get("source_span")).lower()
                for item in hidden_reference.get("missing_confirmatory_evidence") or []
                if isinstance(item, dict)
            }
            for item in auto_limitations:
                if normalize_whitespace(item.get("source_span")).lower() not in existing_spans:
                    hidden_reference["missing_confirmatory_evidence"].append(item)
                    existing_spans.add(normalize_whitespace(item.get("source_span")).lower())
            limitation_spans = {
                normalize_whitespace(value).lower()
                for value in hidden_reference.get("evidence_limitations") or []
            }
            for item in auto_limitations:
                span = normalize_whitespace(item.get("source_span"))
                if span and span.lower() not in limitation_spans:
                    hidden_reference["evidence_limitations"].append(span)
                    limitation_spans.add(span.lower())
            if hidden_reference.get("confirmatory_evidence_present") is not False:
                hidden_reference["confirmatory_evidence_present"] = False
            if not normalize_whitespace(hidden_reference.get("final_diagnosis_certainty")):
                hidden_reference["final_diagnosis_certainty"] = "source_reported_with_explicit_evidence_limitation"
            if "missing_confirmatory_evidence" not in parsed["guardrail_flags"]:
                parsed["guardrail_flags"].append("missing_confirmatory_evidence")
            warnings.append("source-stated diagnostic limitations added to hidden_reference")
    events = parsed.get("evidence_events")
    if isinstance(events, list):
        seen_events = set()
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                continue
            event_id = normalize_whitespace(event.get("event_id"))
            if not event_id or event_id in seen_events:
                event_id = f"E{index}"
                event["event_id"] = event_id
                warnings.append(f"evidence event {index}: event_id set to {event_id}")
            seen_events.add(event_id)
            if event.get("source_order") is None:
                event["source_order"] = index
                warnings.append(f"{event_id}: source_order set to {index}")
    elif events is None:
        parsed["evidence_events"] = []
        warnings.append("missing evidence_events; initialized empty list")
    else:
        warnings.append("evidence_events is not a list; left unchanged for validation")

    milestones = parsed.get("evidence_milestones")
    if not isinstance(milestones, list):
        return ["evidence_milestones is not a list; post-processing skipped"]

    for index, milestone in enumerate(milestones):
        if not isinstance(milestone, dict):
            continue
        milestone.setdefault("turn_id", f"T{index}")
        if not normalize_whitespace(milestone.get("turn_id")):
            milestone["turn_id"] = f"T{index}"
            warnings.append(f"milestone {index}: missing turn_id set to T{index}")
        if not normalize_whitespace(milestone.get("display_name")):
            milestone["display_name"] = stage_display_name(milestone.get("stage", ""), milestone["turn_id"])
            warnings.append(f"{milestone['turn_id']}: missing display_name filled")
        if "included_event_ids" not in milestone:
            milestone["included_event_ids"] = []
            warnings.append(f"{milestone['turn_id']}: missing included_event_ids filled as empty list")
        milestone.setdefault("missing_or_unavailable_tests", [])
        milestone.setdefault("diagnostic_certainty", "")
        milestone.setdefault("evidence_limitations", [])
        milestone["guardrail_flags"] = normalize_guardrail_flags(milestone.get("guardrail_flags"))
        if not normalize_whitespace(milestone.get("grouping_reason")):
            milestone["grouping_reason"] = "Grouped as a source-supported diagnostic evidence bundle."
            warnings.append(f"{milestone['turn_id']}: missing grouping_reason filled")
        if not normalize_whitespace(milestone.get("dialogue_visible_evidence")):
            milestone["dialogue_visible_evidence"] = milestone.get("test_visible_evidence") or milestone.get("visible_evidence", "")
            warnings.append(
                f"{milestone['turn_id']}: missing dialogue_visible_evidence filled from available evidence; review for image-finding leakage"
            )
        if not normalize_whitespace(milestone.get("test_visible_evidence")):
            milestone["test_visible_evidence"] = milestone.get("dialogue_visible_evidence", "")
            warnings.append(f"{milestone['turn_id']}: test_visible_evidence alias filled from dialogue_visible_evidence")

        expected_next = normalize_whitespace(milestone.get("expected_next_evidence_category"))
        if expected_next not in EXPECTED_NEXT_EVIDENCE_CATEGORY_VALUES:
            milestone["expected_next_evidence_category"] = "uncertain"
            warnings.append(f"{milestone['turn_id']}: expected_next_evidence_category set to uncertain")

        expected_readiness = normalize_whitespace(milestone.get("expected_readiness"))
        if expected_readiness not in EXPECTED_READINESS_VALUES:
            milestone["expected_readiness"] = "ready" if milestone.get("stage") == "final_synthesis" else "not_ready"
            warnings.append(f"{milestone['turn_id']}: expected_readiness filled")

        milestone_function = normalize_whitespace(milestone.get("milestone_function"))
        if milestone_function not in MILESTONE_FUNCTION_VALUES:
            milestone["milestone_function"] = default_milestone_function(milestone.get("diagnostic_role", ""))
            warnings.append(f"{milestone['turn_id']}: milestone_function filled from diagnostic_role")
        if "include_in_core_narrowing_metrics" not in milestone:
            milestone["include_in_core_narrowing_metrics"] = milestone.get("stage") != "treatment_or_outcome_if_relevant"
            warnings.append(f"{milestone['turn_id']}: include_in_core_narrowing_metrics filled")
        if is_non_core_context_turn(milestone):
            if milestone.get("include_in_core_narrowing_metrics") is not False:
                warnings.append(f"{milestone['turn_id']}: non-core context/management turn removed from core narrowing metrics")
            milestone["include_in_core_narrowing_metrics"] = False
            if normalize_whitespace(milestone.get("expected_readiness")) == "ready":
                milestone["expected_readiness"] = "possibly_ready"
                warnings.append(f"{milestone['turn_id']}: expected_readiness lowered for non-core context/management turn")
        elif (
            normalize_whitespace(milestone.get("diagnostic_role"))
            in {"broad_narrowing", "major_narrowing", "rule_out_alternative", "confirmatory"}
            or normalize_whitespace(milestone.get("milestone_function"))
            in {
                "narrow_differential",
                "localize_disease",
                "characterize_abnormality",
                "rule_out_alternative",
                "confirm_diagnosis",
            }
        ) and milestone.get("include_in_core_narrowing_metrics") is not True:
            milestone["include_in_core_narrowing_metrics"] = True
            warnings.append(f"{milestone['turn_id']}: diagnostic narrowing turn restored to core metrics")
        corrected_question_type = effective_question_type(milestone)
        question_type_changed = normalize_whitespace(milestone.get("question_type")) != corrected_question_type
        if question_type_changed:
            milestone["question_type"] = corrected_question_type
            warnings.append(f"{milestone['turn_id']}: question_type filled or corrected")
        if question_type_changed or not normalize_whitespace(milestone.get("question")):
            milestone["question"] = default_question(milestone)
            warnings.append(f"{milestone['turn_id']}: question filled")

    warnings.extend(resolve_milestone_image_paths(parsed, nxml_path, nxml_text))
    return warnings


class MilestoneConverter:
    def __init__(self, model: str | None = None, max_tokens: int = 12000, timeout: int = 600) -> None:
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=timeout,
        )
        self.model = model or os.getenv("AZURE_FOUNDRY_EXTRACTOR_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT4") or "gpt-4.1"
        self.max_tokens = max_tokens

    def convert_text(self, case_report: str, nxml_path: Path) -> dict[str, Any]:
        prompt = MILESTONE_CONVERSION_PROMPT.replace("{case_report}", case_report)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.max_tokens,
        )
        raw = response.choices[0].message.content or ""
        try:
            parsed = extract_json_object(raw)
        except Exception as exc:
            raise ValueError(f"{exc}; raw_response_prefix={raw[:1000]!r}; raw_response_length={len(raw)}") from exc
        parsed.setdefault("schema_version", SCHEMA_VERSION)
        parsed.setdefault("source_metadata", {})
        parsed["source_metadata"]["nxml_path"] = str(nxml_path)
        parsed["source_metadata"]["image_folder"] = str(nxml_path.parent)
        parsed["source_metadata"]["extraction_model"] = self.model
        postprocess_warnings = postprocess_milestone_case(parsed, nxml_path, case_report)
        validation = validate_milestone_case(parsed)
        parsed["_conversion_diagnostics"] = {
            "created_at": datetime.now().isoformat(),
            "model": self.model,
            "validation_ok": validation.ok,
            "validation_warnings": postprocess_warnings + validation.warnings,
            "validation_errors": validation.errors,
            "raw_response_length": len(raw),
        }
        return parsed

    def convert_case(self, nxml_path: Path) -> dict[str, Any]:
        case_report = read_text_limited(nxml_path)
        try:
            result = self.convert_text(case_report, nxml_path)
            result.setdefault("case_id", nxml_path.parent.name)
            return result
        except Exception as exc:
            return {
                "schema_version": SCHEMA_VERSION,
                "case_id": nxml_path.parent.name,
                "status": "conversion_failed",
                "source_metadata": {
                    "pmcid": nxml_path.parent.name,
                    "nxml_path": str(nxml_path),
                    "image_folder": str(nxml_path.parent),
                    "extraction_model": self.model,
                    "notes": "",
                },
                "evidence_milestones": [],
                "guardrail_flags": [],
                "hidden_reference": {
                    "final_diagnosis_verbatim": "",
                    "diagnostic_reasoning_verbatim_spans": [],
                    "diagnostic_reasoning_items": [],
                    "final_diagnosis": "",
                    "diagnostic_reasoning": "",
                    "article_derived_evidence_summary": "",
                    "diagnosis_synonyms_or_variants": [],
                    "final_diagnosis_certainty": "",
                    "confirmatory_evidence_present": None,
                    "missing_confirmatory_evidence": [],
                    "evidence_limitations": [],
                },
                "quality_flags": {
                    "has_parseable_nxml": True,
                    "has_retrievable_image": False,
                    "has_final_diagnosis": False,
                    "has_sufficient_evidence_sequence": False,
                    "needs_manual_review": True,
                    "review_notes": str(exc),
                },
                "_conversion_diagnostics": {
                    "created_at": datetime.now().isoformat(),
                    "model": self.model,
                    "validation_ok": False,
                    "validation_warnings": [],
                    "validation_errors": [str(exc)],
                },
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PMC cases to milestone-based diagnostic narrowing schema.")
    parser.add_argument("--input_dir", default="../internalmed")
    parser.add_argument("--output_dir", default="data/pmc_milestone_conversion/internalmed")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--case_ids", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--refresh_postprocess",
        action="store_true",
        help="Refresh deterministic post-processing and validation for existing outputs without an API call.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MilestoneConverter(model=args.model, max_tokens=args.max_tokens, timeout=args.timeout)

    nxml_paths = iter_case_nxmls(input_dir)
    if args.case_ids:
        selected_case_ids = {value.strip() for item in args.case_ids for value in item.split(",") if value.strip()}
        nxml_paths = [path for path in nxml_paths if path.parent.name in selected_case_ids]
    if args.limit is not None:
        nxml_paths = nxml_paths[: args.limit]

    cases = []
    for nxml_path in nxml_paths:
        output_path = output_dir / f"{nxml_path.parent.name}.json"
        if args.refresh_postprocess:
            if not output_path.exists():
                print(f"[converter] {nxml_path.parent.name}: no existing output to refresh", flush=True)
                cases.append({
                    "case_id": nxml_path.parent.name,
                    "status": "missing_existing_output",
                    "output_path": str(output_path),
                })
                continue
            result = json.loads(output_path.read_text(encoding="utf-8"))
            refresh_warnings = postprocess_milestone_case(
                result,
                nxml_path,
                nxml_path.read_text(encoding="utf-8", errors="replace"),
            )
            validation = validate_milestone_case(result)
            diagnostics = result.setdefault("_conversion_diagnostics", {})
            diagnostics["postprocess_refreshed_at"] = datetime.now().isoformat()
            diagnostics["validation_ok"] = validation.ok
            diagnostics["validation_warnings"] = list(dict.fromkeys(refresh_warnings + validation.warnings))
            diagnostics["validation_errors"] = validation.errors
            save_json(output_path, result)
            status = "postprocess_refreshed" if validation.ok else "postprocess_refreshed_with_validation_issues"
            print(f"[converter] {nxml_path.parent.name}: {status}", flush=True)
            cases.append({"case_id": nxml_path.parent.name, "status": status, "output_path": str(output_path)})
            continue
        if output_path.exists() and not args.overwrite:
            print(f"[converter] {nxml_path.parent.name}: skipped existing", flush=True)
            cases.append({"case_id": nxml_path.parent.name, "status": "skipped_existing", "output_path": str(output_path)})
            continue
        print(f"[converter] {nxml_path.parent.name}: converting", flush=True)
        result = converter.convert_case(nxml_path)
        if result.get("status") == "conversion_failed" and output_path.exists():
            failed_path = output_path.with_suffix(".failed.json")
            save_json(failed_path, result)
            print(f"[converter] {nxml_path.parent.name}: failed; preserved existing output", flush=True)
            cases.append({
                "case_id": nxml_path.parent.name,
                "status": "conversion_failed_preserved_existing",
                "output_path": str(output_path),
                "failed_output_path": str(failed_path),
            })
            continue
        save_json(output_path, result)
        status = "completed" if result.get("_conversion_diagnostics", {}).get("validation_ok") else "completed_with_validation_issues"
        print(f"[converter] {nxml_path.parent.name}: {status}", flush=True)
        cases.append({"case_id": nxml_path.parent.name, "status": status, "output_path": str(output_path)})

    save_json(
        output_dir / "_manifest.json",
        {
            "created_at": datetime.now().isoformat(),
            "schema_version": SCHEMA_VERSION,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "limit": args.limit,
            "case_ids": args.case_ids,
            "refresh_postprocess": args.refresh_postprocess,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
