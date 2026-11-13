#!/usr/bin/env python3
"""Resource-level quality audit for DiagTrace reference resources.

This audit is intentionally separate from the older progressive/milestone
construction audit. The judge receives the source report text and the extracted
reference resource, builds a compact source-finding inventory, and labels
whether the event bank covers those findings. Python computes aggregate metrics
from the judge labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from interactive_diagnosis_workspace.interactive_diagnosis.extraction_converter import extraction_source
from interactive_diagnosis_workspace.interactive_diagnosis.nxml_utils import parse_nxml
from llm_judge_common import call_foundry_model, get_foundry_client, save_json, _system_json_message


PROMPT_VERSION = "2026-07-31-reference-resource-quality-v2-support"
DEFAULT_SOURCE_ROOT = Path("runs/pmc_control_gpt41mini_20perdept/narrowing_data_construction")
DEFAULT_OUTPUT_DIR = Path("eacl_benchmark/results/reference_resource_quality/gpt41mini_5case_resource_audit")

CSV_FIELDS = [
    "specialty",
    "case_id",
    "judge_status",
    "source_finding_count",
    "source_finding_full_count",
    "source_finding_partial_count",
    "source_finding_none_count",
    "source_finding_contradicted_count",
    "source_finding_strict_coverage",
    "source_finding_lenient_coverage",
    "source_finding_contradiction_rate",
    "source_event_count",
    "source_event_faithful_count",
    "source_event_partial_count",
    "source_event_unsupported_count",
    "source_event_contradicted_count",
    "source_event_support_rate",
    "source_event_unsupported_rate",
    "reference_reasoning_item_count",
    "reference_reasoning_full_count",
    "reference_reasoning_partial_count",
    "reference_reasoning_none_count",
    "reference_reasoning_contradicted_count",
    "reference_reasoning_strict_coverage",
    "reference_reasoning_lenient_coverage",
    "reference_reasoning_contradiction_rate",
    "leakage_present",
    "final_diagnosis_strict",
    "final_diagnosis_lenient",
    "elapsed_seconds",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "error",
]


@dataclass(frozen=True)
class CaseRecord:
    specialty: str
    case_id: str
    path: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def collect_cases(source_root: Path) -> list[CaseRecord]:
    records: list[CaseRecord] = []
    for specialty_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for path in sorted(specialty_dir.glob("PMC*.json")):
            records.append(CaseRecord(specialty=specialty_dir.name, case_id=path.stem, path=path))
    return records


def records_from_manifest(manifest_path: Path, source_root: Path) -> list[CaseRecord]:
    manifest = load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Manifest must contain a cases list: {manifest_path}")
    records: list[CaseRecord] = []
    for item in cases:
        if not isinstance(item, dict):
            continue
        specialty = str(item.get("specialty") or "")
        case_id = str(item.get("case_id") or "")
        source_file = item.get("source_file")
        if not specialty or not case_id:
            raise ValueError(f"Manifest case is missing specialty/case_id: {item}")
        path = Path(source_file) if source_file else source_root / specialty / f"{case_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Manifest source file not found for {specialty}__{case_id}: {path}")
        records.append(CaseRecord(specialty=specialty, case_id=case_id, path=path))
    return sorted(records, key=lambda item: (item.specialty, item.case_id))


def stratified_sample(records: list[CaseRecord], sample_size: int, seed: int) -> list[CaseRecord]:
    by_specialty: dict[str, list[CaseRecord]] = {}
    for record in records:
        by_specialty.setdefault(record.specialty, []).append(record)
    rng = random.Random(seed)
    for bucket in by_specialty.values():
        rng.shuffle(bucket)
    specialties = sorted(by_specialty)
    if sample_size >= len(records):
        return sorted(records, key=lambda item: (item.specialty, item.case_id))

    base = sample_size // max(len(specialties), 1)
    remainder = sample_size % max(len(specialties), 1)
    selected: list[CaseRecord] = []
    leftovers: list[CaseRecord] = []
    for index, specialty in enumerate(specialties):
        quota = base + (1 if index < remainder else 0)
        bucket = by_specialty[specialty]
        take = min(quota, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])
    if len(selected) < sample_size:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])
    return sorted(selected[:sample_size], key=lambda item: (item.specialty, item.case_id))


def default_deployment() -> str:
    load_dotenv()
    for key in (
        "AZURE_OPENAI_DEPLOYMENT_GPT41_MINI",
        "AZURE_OPENAI_DEPLOYMENT_GPT4_1_MINI",
        "AZURE_OPENAI_DEPLOYMENT_GPT4_1_MINI_TEST",
        "AZURE_FOUNDRY_GPT41_MINI_DEPLOYMENT",
    ):
        value = os.getenv(key)
        if value:
            return value
    _client, config = get_foundry_client()
    return config.extractor_deployment


def event_id(item: dict[str, Any]) -> str:
    return str(item.get("event_id") or item.get("evidence_id") or item.get("id") or "").strip()


def source_text(item: dict[str, Any]) -> str:
    return str(
        item.get("source_span")
        or item.get("source_text")
        or item.get("source_quote")
        or item.get("answer_text")
        or ""
    ).strip()


def compact_events(case_data: dict[str, Any], max_events: int = 40) -> list[dict[str, Any]]:
    events = case_data.get("evidence_events") or case_data.get("evidence_items") or []
    compact: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "event_id": event_id(item) or f"E{index}",
                "source_order": item.get("source_order") or item.get("phase_index"),
                "category": item.get("evidence_type") or item.get("category"),
                "diagnostic_role": item.get("diagnostic_role") or item.get("diagnostic_stage"),
                "diagnosis_revealing": bool(item.get("diagnosis_revealing"))
                or str(item.get("disclosure_policy") or "") == "withhold_final_diagnosis_leakage",
                "answer_or_source_text": source_text(item),
                "figure_ids": item.get("linked_figures") or item.get("figure_ids") or [],
            }
        )
    return compact[:max_events]


def diagnostic_reasoning_items(hidden: dict[str, Any]) -> list[dict[str, str]]:
    raw_items = hidden.get("diagnostic_reasoning_items")
    items: list[str] = []
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("reasoning") or item.get("claim") or "").strip()
            else:
                text = str(item or "").strip()
            if text:
                items.append(text)
    if not items:
        spans = hidden.get("diagnostic_reasoning_verbatim_spans")
        if isinstance(spans, list):
            items.extend(str(item).strip() for item in spans if str(item).strip())
    if not items:
        text = str(hidden.get("diagnostic_reasoning") or "").strip()
        if text:
            items.append(text)
    return [{"id": f"R{idx}", "text": text} for idx, text in enumerate(items, start=1)]


def case_source_text(case_data: dict[str, Any], max_chars: int) -> str:
    metadata = case_data.get("source_metadata") if isinstance(case_data.get("source_metadata"), dict) else {}
    nxml_path = Path(str(metadata.get("nxml_path") or ""))
    if nxml_path.exists():
        return extraction_source(parse_nxml(nxml_path), max_chars=max_chars)
    return json.dumps(
        {
            "source_metadata": metadata,
            "source_spans_from_event_bank": [event["answer_or_source_text"] for event in compact_events(case_data)],
        },
        ensure_ascii=False,
        indent=2,
    )


def compact_initial(case_data: dict[str, Any]) -> Any:
    return case_data.get("initial_observation") or case_data.get("initial_presentation") or {}


def compact_hidden_reference(case_data: dict[str, Any]) -> dict[str, Any]:
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    return {
        "final_diagnosis_verbatim": hidden.get("final_diagnosis_verbatim"),
        "final_diagnosis": hidden.get("final_diagnosis") or hidden.get("case_level_diagnosis"),
        "diagnosis_synonyms_or_variants": hidden.get("diagnosis_synonyms_or_variants") or hidden.get("accepted_aliases") or [],
        "diagnostic_reasoning_items": diagnostic_reasoning_items(hidden),
        "diagnostic_reasoning_verbatim_spans": hidden.get("diagnostic_reasoning_verbatim_spans") or [],
    }


def build_payload(case_data: dict[str, Any], record: CaseRecord, max_source_chars: int) -> dict[str, Any]:
    return {
        "case_id": case_data.get("case_id") or record.case_id,
        "specialty": record.specialty,
        "source_report_text": case_source_text(case_data, max_source_chars),
        "initial_presentation": compact_initial(case_data),
        "source_ordered_event_bank": compact_events(case_data),
        "article_derived_diagnostic_references": compact_hidden_reference(case_data),
    }


def build_prompt(payload: dict[str, Any]) -> str:
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
You are auditing a source-grounded reference resource for an interactive medical
diagnosis benchmark. Evaluate benchmark data quality, not downstream model
performance. Use only the supplied source report text and extracted reference
resource. Do not add external medical facts.

Part A: Source-finding inventory and event-bank coverage
First identify a compact inventory of 8-15 major diagnosis-relevant findings
from the source report text. The inventory should span the diagnostic trajectory
when present: initial presenting context, key history or symptom evolution,
physical examination or bedside findings, laboratory or microbiology results,
imaging, pathology/molecular/genetic confirmation, procedures or interventions,
treatment response, follow-up, and clinical course. Do not list every minor lab
value. Each source finding must include a short verbatim or near-verbatim source
phrase from the report.

For each source finding, decide whether it is represented by at least one item in
the source-ordered event bank. Labels:
- full: the event bank captures the clinically essential finding.
- partial: the event bank captures the main finding but misses an essential
  qualifier, site, subtype, organism, timing, or result detail.
- none: the finding is not represented by the event bank.
- contradicted: the event bank conflicts with the source finding.

Part B: Source-event support and verbatim fidelity
For each event in the source-ordered event bank, decide whether the event is
faithful to its quoted source text.
Labels:
- faithful: the event is directly supported by the source text and preserves
  clinically essential details.
- partial: the event is mostly supported but over-compresses findings, omits an
  important qualifier, or slightly overgeneralizes.
- unsupported: the event includes a claim not supported by the source text.
- contradicted: the event conflicts with the source text.

Part C: Diagnostic-reasoning coverage
For each article-derived diagnostic reasoning item, decide whether the
source-ordered event bank and diagnostic references cover it. Use the same
full, partial, none, and contradicted labels.

Part D: Initial-presentation leakage
Flag leakage only if the initial presentation exposes the article-reported final
diagnosis or retrospective diagnostic synthesis. Do not count source-backed
symptoms, signs, tests, imaging, pathology, microbiology, or molecular results
as leakage when they are ordinary documented findings.

Part E: Final-diagnosis agreement
Assess whether the extracted diagnostic reference preserves the article-reported
final diagnosis. Strict matching requires clinically essential qualifiers such
as etiology, organism, anatomical site, subtype, or named syndrome when present.
Lenient matching accepts the principal diagnostic entity if it is clinically
related but underspecified.

Input:
{payload_text}

Return JSON only with exactly this structure:
{{
  "case_id": "",
  "audit_status": "completed",
  "source_finding_coverage": {{
    "inventory": [
      {{
        "finding_id": "F1",
        "source_finding": "",
        "coverage_label": "full",
        "matched_event_ids": ["E1"],
        "rationale": ""
      }}
    ],
    "coverage_notes": ""
  }},
  "source_event_support": {{
    "event_labels": [
      {{
        "event_id": "E1",
        "support_label": "faithful",
        "rationale": ""
      }}
    ],
    "support_notes": ""
  }},
  "reference_reasoning_coverage": {{
    "item_matches": [
      {{
        "reference_item_id": "R1",
        "coverage_label": "full",
        "matched_event_ids": ["E1"],
        "rationale": ""
      }}
    ],
    "coverage_notes": ""
  }},
  "leakage_audit": {{
    "leakage_present": false,
    "leakage_locations": [],
    "rationale": ""
  }},
  "final_diagnosis_reference": {{
    "strict_match": false,
    "lenient_match": false,
    "quality_score": 1,
    "rationale": ""
  }}
}}
""".strip()


def normalize_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"full", "partial", "none", "contradicted"}:
        return label
    if label in {"contradiction", "conflict", "contradicts"}:
        return "contradicted"
    return "none"


def normalize_support_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    if label in {"faithful", "partial", "unsupported", "contradicted"}:
        return label
    if label in {"supported", "full"}:
        return "faithful"
    if label in {"contradiction", "conflict", "contradicts"}:
        return "contradicted"
    return "unsupported"


def safe_div(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def coverage_counts(labels: list[str]) -> dict[str, Any]:
    total = len(labels)
    full = sum(1 for label in labels if label == "full")
    partial = sum(1 for label in labels if label == "partial")
    contradicted = sum(1 for label in labels if label == "contradicted")
    none = total - full - partial - contradicted
    return {
        "count": total,
        "full": full,
        "partial": partial,
        "none": none,
        "contradicted": contradicted,
        "strict": safe_div(full, total),
        "lenient": safe_div(full + partial, total),
        "contradiction_rate": safe_div(contradicted, total),
    }


def source_finding_metrics(judge: dict[str, Any]) -> dict[str, Any]:
    coverage = judge.get("source_finding_coverage") if isinstance(judge.get("source_finding_coverage"), dict) else {}
    inventory = coverage.get("inventory") if isinstance(coverage.get("inventory"), list) else []
    labels = [normalize_label(item.get("coverage_label")) for item in inventory if isinstance(item, dict)]
    counts = coverage_counts(labels)
    return {
        "source_finding_count": counts["count"],
        "source_finding_full_count": counts["full"],
        "source_finding_partial_count": counts["partial"],
        "source_finding_none_count": counts["none"],
        "source_finding_contradicted_count": counts["contradicted"],
        "source_finding_strict_coverage": counts["strict"],
        "source_finding_lenient_coverage": counts["lenient"],
        "source_finding_contradiction_rate": counts["contradiction_rate"],
    }


def source_event_support_metrics(payload: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    events = payload.get("source_ordered_event_bank") or []
    event_ids = {str(item.get("event_id")) for item in events if isinstance(item, dict) and item.get("event_id")}
    support = judge.get("source_event_support") if isinstance(judge.get("source_event_support"), dict) else {}
    labels = support.get("event_labels") if isinstance(support.get("event_labels"), list) else []
    best_by_event = {eid: "unsupported" for eid in event_ids}
    rank = {"unsupported": 0, "partial": 1, "faithful": 2, "contradicted": 3}
    for item in labels:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("event_id") or "").strip()
        if eid not in event_ids:
            continue
        label = normalize_support_label(item.get("support_label"))
        if rank[label] > rank[best_by_event.get(eid, "unsupported")]:
            best_by_event[eid] = label

    values = list(best_by_event.values())
    total = len(values)
    faithful = sum(1 for label in values if label == "faithful")
    partial = sum(1 for label in values if label == "partial")
    unsupported = sum(1 for label in values if label == "unsupported")
    contradicted = sum(1 for label in values if label == "contradicted")
    return {
        "source_event_count": total,
        "source_event_faithful_count": faithful,
        "source_event_partial_count": partial,
        "source_event_unsupported_count": unsupported,
        "source_event_contradicted_count": contradicted,
        "source_event_support_rate": safe_div(faithful + partial, total),
        "source_event_unsupported_rate": safe_div(unsupported + contradicted, total),
    }


def reasoning_metrics(payload: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
    reference_items = (
        payload.get("article_derived_diagnostic_references", {}).get("diagnostic_reasoning_items") or []
    )
    reference_ids = {str(item.get("id")) for item in reference_items if isinstance(item, dict) and item.get("id")}
    coverage = judge.get("reference_reasoning_coverage") if isinstance(judge.get("reference_reasoning_coverage"), dict) else {}
    matches = coverage.get("item_matches") if isinstance(coverage.get("item_matches"), list) else []
    best_by_reference = {rid: "none" for rid in reference_ids}
    rank = {"none": 0, "partial": 1, "full": 2, "contradicted": 3}
    for item in matches:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("reference_item_id") or "").strip()
        if rid not in reference_ids:
            continue
        label = normalize_label(item.get("coverage_label"))
        if rank[label] > rank[best_by_reference.get(rid, "none")]:
            best_by_reference[rid] = label
    counts = coverage_counts(list(best_by_reference.values()))
    return {
        "reference_reasoning_item_count": counts["count"],
        "reference_reasoning_full_count": counts["full"],
        "reference_reasoning_partial_count": counts["partial"],
        "reference_reasoning_none_count": counts["none"],
        "reference_reasoning_contradicted_count": counts["contradicted"],
        "reference_reasoning_strict_coverage": counts["strict"],
        "reference_reasoning_lenient_coverage": counts["lenient"],
        "reference_reasoning_contradiction_rate": counts["contradiction_rate"],
    }


def usage_value(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    if value is None and key == "prompt_tokens":
        value = usage.get("input_tokens")
    if value is None and key == "completion_tokens":
        value = usage.get("output_tokens")
    if value is None and key == "total_tokens":
        value = usage.get("total_tokens")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def row_from_judge(record: CaseRecord, payload: dict[str, Any], judge: dict[str, Any], result: Any, elapsed: float) -> dict[str, Any]:
    leakage = judge.get("leakage_audit") if isinstance(judge.get("leakage_audit"), dict) else {}
    diagnosis = judge.get("final_diagnosis_reference") if isinstance(judge.get("final_diagnosis_reference"), dict) else {}
    usage = getattr(result, "usage", {}) or {}
    return {
        "specialty": record.specialty,
        "case_id": record.case_id,
        "judge_status": "completed",
        **source_finding_metrics(judge),
        **source_event_support_metrics(payload, judge),
        **reasoning_metrics(payload, judge),
        "leakage_present": bool(leakage.get("leakage_present")),
        "final_diagnosis_strict": bool(diagnosis.get("strict_match")),
        "final_diagnosis_lenient": bool(diagnosis.get("lenient_match")),
        "elapsed_seconds": round(elapsed, 3),
        "prompt_tokens": usage_value(usage, "prompt_tokens"),
        "completion_tokens": usage_value(usage, "completion_tokens"),
        "total_tokens": usage_value(usage, "total_tokens"),
        "error": "",
    }


def summarize(rows: list[dict[str, Any]], manifest_cases: list[dict[str, Any]], deployment: str) -> dict[str, Any]:
    completed = [row for row in rows if row.get("judge_status") == "completed"]

    def numeric_mean(key: str) -> float | None:
        values: list[float] = []
        for row in completed:
            value = row.get(key)
            if value in {"", None}:
                continue
            values.append(float(value))
        return mean(values) if values else None

    def bool_rate(key: str) -> float | None:
        values = [row.get(key) for row in completed if row.get(key) != ""]
        return mean(1.0 if value in {True, "True", "true", 1, "1"} else 0.0 for value in values) if values else None

    return {
        "created_at": now(),
        "prompt_version": PROMPT_VERSION,
        "judge_deployment": deployment,
        "num_cases_sampled": len(manifest_cases),
        "num_cases_evaluated": len(completed),
        "num_failed": sum(1 for row in rows if row.get("judge_status") == "failed"),
        "mean_source_finding_strict_coverage": numeric_mean("source_finding_strict_coverage"),
        "mean_source_finding_lenient_coverage": numeric_mean("source_finding_lenient_coverage"),
        "mean_source_finding_contradiction_rate": numeric_mean("source_finding_contradiction_rate"),
        "mean_source_event_support_rate": numeric_mean("source_event_support_rate"),
        "mean_source_event_unsupported_rate": numeric_mean("source_event_unsupported_rate"),
        "mean_reference_reasoning_strict_coverage": numeric_mean("reference_reasoning_strict_coverage"),
        "mean_reference_reasoning_lenient_coverage": numeric_mean("reference_reasoning_lenient_coverage"),
        "mean_reference_reasoning_contradiction_rate": numeric_mean("reference_reasoning_contradiction_rate"),
        "leakage_case_rate": bool_rate("leakage_present"),
        "final_diagnosis_strict_rate": bool_rate("final_diagnosis_strict"),
        "final_diagnosis_lenient_rate": bool_rate("final_diagnosis_lenient"),
        "mean_elapsed_seconds": numeric_mean("elapsed_seconds"),
        "total_prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in completed),
        "total_completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in completed),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in completed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.case_manifest:
        selected = records_from_manifest(args.case_manifest, args.source_root)
    else:
        records = collect_cases(args.source_root)
        selected = stratified_sample(records, args.sample_size, args.seed)
    output_dir = args.output_dir
    inputs_dir = output_dir / "judge_inputs"
    cases_dir = output_dir / "cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)

    deployment = args.deployment or default_deployment()
    manifest_cases = [{"specialty": record.specialty, "case_id": record.case_id, "source_file": str(record.path)} for record in selected]
    save_json(output_dir / "sample_manifest.json", {"created_at": now(), "seed": args.seed, "cases": manifest_cases})

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        case_key = f"{record.specialty}__{record.case_id}"
        output_path = cases_dir / f"{case_key}.reference_resource_quality.json"
        case_data = load_json(record.path)
        payload = build_payload(case_data, record, args.max_source_chars)
        save_json(inputs_dir / f"{case_key}.json", payload)
        if output_path.exists() and not args.overwrite:
            existing = load_json(output_path)
            existing_row = existing.get("row") or {}
            if existing_row.get("judge_status") != "failed":
                rows.append(existing_row)
                print(f"[{index}/{len(selected)}] {case_key}: loaded_existing", flush=True)
                continue
            print(f"[{index}/{len(selected)}] {case_key}: retrying previous failed result", flush=True)
        started = time.monotonic()
        try:
            result = call_foundry_model(
                deployment=deployment,
                messages=[_system_json_message("reference resource quality auditor"), {"role": "user", "content": build_prompt(payload)}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                role_name="reference_resource_quality_auditor",
                max_retries=args.max_retries,
                request_timeout=args.timeout,
            )
            elapsed = time.monotonic() - started
            row = row_from_judge(record, payload, result.parsed_json, result, elapsed)
            save_json(
                output_path,
                {
                    "created_at": now(),
                    "prompt_version": PROMPT_VERSION,
                    "judge_deployment": deployment,
                    "judge_output": result.parsed_json,
                    "raw_response": result.raw_response,
                    "usage": result.usage,
                    "row": row,
                },
            )
            rows.append(row)
            print(f"[{index}/{len(selected)}] {case_key}: completed in {elapsed:.1f}s", flush=True)
        except Exception as exc:
            elapsed = time.monotonic() - started
            row = {
                "specialty": record.specialty,
                "case_id": record.case_id,
                "judge_status": "failed",
                "elapsed_seconds": round(elapsed, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
            rows.append(row)
            save_json(output_path, {"created_at": now(), "prompt_version": PROMPT_VERSION, "row": row})
            print(f"[{index}/{len(selected)}] {case_key}: failed: {row['error']}", flush=True)

    write_csv(output_dir / "case_quality_summary.csv", rows)
    aggregate = summarize(rows, manifest_cases, deployment)
    save_json(output_dir / "_aggregate.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resource-level reference resource quality audit.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--case-manifest", type=Path, default=None, help="Optional manifest with fixed specialty/case_id/source_file records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=4023)
    parser.add_argument("--deployment", default="")
    parser.add_argument("--max-source-chars", type=int, default=45000)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    aggregate = run(parse_args())
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
