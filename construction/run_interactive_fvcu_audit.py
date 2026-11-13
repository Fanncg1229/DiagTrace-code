#!/usr/bin/env python3
"""LLM-as-judge FVCU audit for interactive reconstruction episodes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_judge_common import _system_json_message, call_foundry_model, get_foundry_client, save_json


PROMPT_VERSION = "2026-07-31-interactive-fvcu-v1"
DEFAULT_EPISODE_ROOT = Path(
    "eacl_benchmark/results/gpt41mini_target500/04_diagnosis_aware_reconstruction/"
    "setting2_diagnosis_aware_reconstruction"
)
DEFAULT_OUTPUT_DIR = Path("eacl_benchmark/results/interactive_fvcu_quality/gpt41mini_5case_interactive_fvcu")

CSV_FIELDS = [
    "specialty",
    "case_id",
    "judge_status",
    "fvcu_factuality",
    "fvcu_validity",
    "fvcu_coherence",
    "fvcu_utility",
    "fvcu_overall",
    "unsupported_or_unreleased_claims",
    "request_strategy_problem",
    "final_reconstruction_grounded",
    "turns_used",
    "answer_yield",
    "unavailable_turn_rate",
    "episode_file",
    "case_file",
    "evaluation_file",
    "error",
]


@dataclass(frozen=True)
class EpisodeRecord:
    specialty: str
    case_id: str
    episode_path: Path
    case_path: Path
    evaluation_path: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


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


def score_value(obj: dict[str, Any], *path: str) -> int | None:
    current: Any = obj
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, dict):
        current = current.get("score")
    try:
        score = int(current)
    except (TypeError, ValueError):
        return None
    return score if 1 <= score <= 5 else None


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def find_records(episode_root: Path) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for episode_path in sorted(episode_root.glob("*/episodes/*.json")):
        specialty = episode_path.parents[1].name
        case_id = episode_path.stem
        case_path = episode_path.parents[1] / "cases" / f"{case_id}.json"
        evaluation_path = episode_path.parents[1] / "evaluations" / f"{case_id}.json"
        if case_path.exists() and evaluation_path.exists():
            records.append(
                EpisodeRecord(
                    specialty=specialty,
                    case_id=case_id,
                    episode_path=episode_path,
                    case_path=case_path,
                    evaluation_path=evaluation_path,
                )
            )
    return records


def specialty_from_path(path: Path) -> str:
    parts = path.parts
    for marker in ("episodes", "cases", "evaluations"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return parts[index - 1]
    return path.parent.parent.name if len(path.parents) > 1 else ""


def records_from_metrics_csv(metrics_csv: Path, case_ids: set[str] | None = None) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    with metrics_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") and row.get("status") != "completed":
                continue
            case_id = str(row.get("case_id") or "").strip()
            if not case_id or (case_ids is not None and case_id not in case_ids):
                continue
            episode_path = Path(str(row.get("episode_path") or ""))
            case_path = Path(str(row.get("case_path") or ""))
            if not episode_path.exists() or not case_path.exists():
                continue
            evaluation_path = episode_path.parent.parent / "evaluations" / f"{case_id}.json"
            if not evaluation_path.exists():
                continue
            records.append(
                EpisodeRecord(
                    specialty=specialty_from_path(episode_path),
                    case_id=case_id,
                    episode_path=episode_path,
                    case_path=case_path,
                    evaluation_path=evaluation_path,
                )
            )
    return sorted(records, key=lambda item: (item.specialty, item.case_id))


def case_ids_from_manifest(manifest_path: Path) -> set[str]:
    manifest = load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Manifest must contain a cases list: {manifest_path}")
    return {str(item.get("case_id") or "") for item in cases if isinstance(item, dict) and item.get("case_id")}


def records_from_manifest(manifest_path: Path, episode_root: Path) -> list[EpisodeRecord]:
    manifest = load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"Manifest must contain a cases list: {manifest_path}")
    available = {(record.specialty, record.case_id): record for record in find_records(episode_root)}
    records: list[EpisodeRecord] = []
    missing: list[str] = []
    for item in cases:
        if not isinstance(item, dict):
            continue
        specialty = str(item.get("specialty") or "")
        case_id = str(item.get("case_id") or "")
        record = available.get((specialty, case_id))
        if record is None:
            missing.append(f"{specialty}__{case_id}")
            continue
        records.append(record)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest cases do not have complete interactive traces under {episode_root}: "
            + ", ".join(missing[:10])
        )
    return sorted(records, key=lambda item: (item.specialty, item.case_id))


def stratified_sample(records: list[EpisodeRecord], sample_size: int, seed: int) -> list[EpisodeRecord]:
    by_specialty: dict[str, list[EpisodeRecord]] = {}
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
    selected: list[EpisodeRecord] = []
    leftovers: list[EpisodeRecord] = []
    for idx, specialty in enumerate(specialties):
        quota = base + (1 if idx < remainder else 0)
        bucket = by_specialty[specialty]
        selected.extend(bucket[:quota])
        leftovers.extend(bucket[quota:])
    if len(selected) < sample_size:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: sample_size - len(selected)])
    return sorted(selected[:sample_size], key=lambda item: (item.specialty, item.case_id))


def compact_events(case_data: dict[str, Any], limit: int = 40) -> list[dict[str, Any]]:
    events = case_data.get("evidence_items") if isinstance(case_data.get("evidence_items"), list) else []
    compact: list[dict[str, Any]] = []
    for item in events[:limit]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "id": item.get("id"),
                "category": item.get("category") or item.get("evidence_type"),
                "source_text": item.get("source_text"),
                "timeline_order": item.get("timeline_order"),
                "phase_label": item.get("phase_label"),
                "diagnosis_revealing": item.get("diagnosis_revealing"),
            }
        )
    return compact


def compact_turns(episode: dict[str, Any]) -> list[dict[str, Any]]:
    turns = episode.get("turns") if isinstance(episode.get("turns"), list) else []
    compact: list[dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        clinician = turn.get("clinician") if isinstance(turn.get("clinician"), dict) else {}
        report = turn.get("report") if isinstance(turn.get("report"), dict) else None
        compact_report = None
        if report is not None:
            compact_report = {
                "status": report.get("status"),
                "evidence_id": report.get("evidence_id"),
                "source_text": report.get("source_text"),
                "category": report.get("category"),
                "explanation": report.get("explanation"),
            }
        compact.append(
            {
                "turn_number": turn.get("turn_number"),
                "action": clinician.get("action"),
                "request": clinician.get("request"),
                "working_reconstruction": clinician.get("current_trajectory_hypothesis"),
                "confirmed_evidence": clinician.get("confirmed_evidence") or [],
                "missing_key_evidence": clinician.get("missing_key_evidence") or [],
                "evidence_sufficiency_confidence": clinician.get("confidence_that_diagnosis_is_supported"),
                "final_diagnosis": clinician.get("final_diagnosis"),
                "trajectory_summary": clinician.get("trajectory_summary"),
                "key_supporting_evidence": clinician.get("key_supporting_evidence") or [],
                "remaining_uncertainties": clinician.get("remaining_uncertainties") or [],
                "report": compact_report,
            }
        )
    return compact


def build_payload(record: EpisodeRecord, case_data: dict[str, Any], episode: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    hidden = case_data.get("hidden_reference") if isinstance(case_data.get("hidden_reference"), dict) else {}
    final = episode.get("final") if isinstance(episode.get("final"), dict) else {}
    return {
        "case_id": record.case_id,
        "specialty": record.specialty,
        "reference_resource": {
            "initial_presentation": case_data.get("initial_observation"),
            "known_final_diagnosis": evaluation.get("known_final_diagnosis")
            or hidden.get("final_diagnosis_verbatim")
            or hidden.get("final_diagnosis"),
            "article_derived_diagnostic_references": {
                "final_diagnosis": hidden.get("final_diagnosis"),
                "final_diagnosis_verbatim": hidden.get("final_diagnosis_verbatim"),
                "diagnostic_reasoning": hidden.get("diagnostic_reasoning") or hidden.get("reasoning_items") or [],
            },
            "source_ordered_event_bank": compact_events(case_data),
        },
        "interactive_trace": {
            "turns_used": episode.get("turns_used"),
            "max_turns": episode.get("max_turns"),
            "termination": episode.get("termination"),
            "disclosed_evidence_ids": episode.get("disclosed_evidence_ids") or [],
            "turns": compact_turns(episode),
            "final_reconstruction": {
                "final_diagnosis": final.get("final_diagnosis"),
                "trajectory_summary": final.get("trajectory_summary"),
                "key_supporting_evidence": final.get("key_supporting_evidence") or [],
                "remaining_uncertainties": final.get("remaining_uncertainties") or [],
                "evidence_sufficiency_confidence": final.get("confidence_that_diagnosis_is_supported"),
            },
        },
        "deterministic_episode_summary": {
            "turns_used": evaluation.get("turns_used"),
            "answer_yield": evaluation.get("answer_yield"),
            "unavailable_turn_rate": evaluation.get("unavailable_turn_rate"),
            "action_counts": evaluation.get("action_counts") or {},
            "report_status_counts": evaluation.get("report_status_counts") or {},
            "invalid_cited_evidence_ids": evaluation.get("invalid_cited_evidence_ids") or [],
            "disclosed_evidence_count": evaluation.get("disclosed_evidence_count"),
        },
    }


def judge_prompt(payload: dict[str, Any]) -> str:
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
You are evaluating an interactive diagnostic-trajectory reconstruction episode.

Evaluate the diagnosis-aware clinician agent, not the report agent. Use only the
provided reference resource, source-ordered event bank, article-derived
diagnostic references, and interaction trace. The clinician agent knows the
final diagnosis, but must actively recover the intermediate supporting evidence
through report-agent responses.

Score four FVCU dimensions on a 1-5 scale:

1. Factuality
Whether the clinician agent's case-specific claims, confirmed evidence, and final
reconstruction are grounded in report-agent responses available by that turn.
Penalize fabricated findings, unsupported claims, invalid evidence citations, or
use of unreleased source events as if observed.

2. Validity
Whether the clinician agent's requests and interpretations are clinically
reasonable given the known diagnosis and acquired evidence. Penalize vague,
irrelevant, redundant, or overly diagnostic requests, and unsupported
interpretations of returned evidence.

3. Coherence
Whether the working reconstruction, confirmed evidence, missing evidence, and
confidence evolve consistently across turns. Penalize contradictions, unexplained
shifts, repeated requests that do not update the reconstruction, or final
reconstructions disconnected from the interaction trace.

4. Utility
Whether the clinician agent uses the fixed turn budget to acquire high-yield
documented evidence needed to reconstruct the source-derived diagnostic
trajectory. A useful episode prioritizes specific evidence categories, avoids
unnecessary requests, and produces a final reconstruction supported by released
event identifiers.

Overall quality:
Provide a holistic 1-5 overall_quality score. Do not mechanically average the
four scores. Major unsupported claims, clinically invalid strategy, or very low
utility should lower the overall score because they reduce the usefulness of the
interactive episode.

Also return flags:
- unsupported_or_unreleased_claims: true if the clinician presents unsupported or
  unreleased evidence as observed.
- request_strategy_problem: true if the request pattern is dominated by vague,
  redundant, unavailable, or overly diagnostic requests.
- final_reconstruction_grounded: true if the final reconstruction is grounded in
  released event identifiers and report-agent responses.

Input:
{payload_text}

Return JSON only:
{{
  "case_id": "",
  "judge_status": "completed",
  "interactive_fvcu": {{
    "factuality": {{"score": 1, "major_issues": [], "rationale": ""}},
    "validity": {{"score": 1, "major_issues": [], "rationale": ""}},
    "coherence": {{"score": 1, "major_issues": [], "rationale": ""}},
    "utility": {{"score": 1, "major_issues": [], "rationale": ""}},
    "overall_quality": {{"score": 1, "rationale": ""}}
  }},
  "trajectory_flags": {{
    "unsupported_or_unreleased_claims": false,
    "request_strategy_problem": false,
    "final_reconstruction_grounded": true
  }},
  "episode_notes": ""
}}
"""


def row_from_judge(record: EpisodeRecord, evaluation: dict[str, Any], judge: dict[str, Any], status: str, error: str = "") -> dict[str, Any]:
    flags = judge.get("trajectory_flags") if isinstance(judge.get("trajectory_flags"), dict) else {}
    return {
        "specialty": record.specialty,
        "case_id": record.case_id,
        "judge_status": status,
        "fvcu_factuality": score_value(judge, "interactive_fvcu", "factuality"),
        "fvcu_validity": score_value(judge, "interactive_fvcu", "validity"),
        "fvcu_coherence": score_value(judge, "interactive_fvcu", "coherence"),
        "fvcu_utility": score_value(judge, "interactive_fvcu", "utility"),
        "fvcu_overall": score_value(judge, "interactive_fvcu", "overall_quality"),
        "unsupported_or_unreleased_claims": bool_value(flags.get("unsupported_or_unreleased_claims")) if judge else "",
        "request_strategy_problem": bool_value(flags.get("request_strategy_problem")) if judge else "",
        "final_reconstruction_grounded": bool_value(flags.get("final_reconstruction_grounded")) if judge else "",
        "turns_used": evaluation.get("turns_used"),
        "answer_yield": evaluation.get("answer_yield"),
        "unavailable_turn_rate": evaluation.get("unavailable_turn_rate"),
        "episode_file": str(record.episode_path),
        "case_file": str(record.case_path),
        "evaluation_file": str(record.evaluation_path),
        "error": error,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def aggregate_rows(rows: list[dict[str, Any]], usage: dict[str, int]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("judge_status") in {"completed", "loaded_existing"}]

    def numeric_mean(key: str) -> float | None:
        values: list[float] = []
        for row in completed:
            value = row.get(key)
            if value in ("", None):
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return mean(values) if values else None

    def bool_rate(key: str) -> float | None:
        values = [bool_value(row.get(key)) for row in completed if row.get(key) not in ("", None)]
        return mean(values) if values else None

    return {
        "created_at": now(),
        "prompt_version": PROMPT_VERSION,
        "num_cases_requested": len(rows),
        "num_cases_judged": len(completed),
        "num_failed": len(rows) - len(completed),
        "mean_fvcu_factuality": numeric_mean("fvcu_factuality"),
        "mean_fvcu_validity": numeric_mean("fvcu_validity"),
        "mean_fvcu_coherence": numeric_mean("fvcu_coherence"),
        "mean_fvcu_utility": numeric_mean("fvcu_utility"),
        "mean_fvcu_overall": numeric_mean("fvcu_overall"),
        "unsupported_or_unreleased_claim_rate": bool_rate("unsupported_or_unreleased_claims"),
        "request_strategy_problem_rate": bool_rate("request_strategy_problem"),
        "final_reconstruction_grounded_rate": bool_rate("final_reconstruction_grounded"),
        "mean_turns_used": numeric_mean("turns_used"),
        "mean_answer_yield": numeric_mean("answer_yield"),
        "mean_unavailable_turn_rate": numeric_mean("unavailable_turn_rate"),
        "usage": usage,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    cases_dir = output_dir / "cases"
    inputs_dir = output_dir / "judge_inputs"
    cases_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if args.case_manifest:
        if args.metrics_csv:
            records = records_from_metrics_csv(args.metrics_csv, case_ids_from_manifest(args.case_manifest))
        else:
            records = records_from_manifest(args.case_manifest, args.episode_root)
    else:
        records = records_from_metrics_csv(args.metrics_csv) if args.metrics_csv else find_records(args.episode_root)
        records = stratified_sample(records, args.sample_size, args.seed)
    save_json(
        output_dir / "sample_manifest.json",
        {
            "created_at": now(),
            "seed": args.seed,
            "episode_root": str(args.episode_root),
            "case_manifest": str(args.case_manifest) if args.case_manifest else "",
            "cases": [
                {
                    "specialty": record.specialty,
                    "case_id": record.case_id,
                    "episode_file": str(record.episode_path),
                    "case_file": str(record.case_path),
                    "evaluation_file": str(record.evaluation_path),
                }
                for record in records
            ],
        },
    )

    get_foundry_client()
    deployment = args.deployment or default_deployment()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records, start=1):
        result_path = cases_dir / f"{record.specialty}__{record.case_id}.interactive_fvcu.json"
        try:
            case_data = load_json(record.case_path)
            episode = load_json(record.episode_path)
            evaluation = load_json(record.evaluation_path)
            payload = build_payload(record, case_data, episode, evaluation)
            input_path = inputs_dir / f"{record.specialty}__{record.case_id}.json"
            save_json(input_path, payload)

            if result_path.exists() and not args.overwrite:
                existing = load_json(result_path)
                judge = existing.get("judge_output") if isinstance(existing.get("judge_output"), dict) else {}
                rows.append(row_from_judge(record, evaluation, judge, "loaded_existing"))
                print(f"[{idx}/{len(records)}] {record.specialty}__{record.case_id}: loaded existing")
                continue

            result = call_foundry_model(
                deployment=deployment,
                messages=[_system_json_message("interactive reconstruction FVCU judge"), {"role": "user", "content": judge_prompt(payload)}],
                max_tokens=args.max_tokens,
                temperature=0,
                role_name="interactive_fvcu_judge",
            )
            usage_data = result.usage if isinstance(result.usage, dict) else {}
            prompt_tokens = int(usage_data.get("prompt_tokens") or 0)
            completion_tokens = int(usage_data.get("completion_tokens") or 0)
            total_tokens = int(usage_data.get("total_tokens") or prompt_tokens + completion_tokens)
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += total_tokens
            save_json(
                result_path,
                {
                    "prompt_version": PROMPT_VERSION,
                    "judge_deployment": deployment,
                    "judge_input_file": str(input_path),
                    "judge_output": result.parsed_json,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "created_at": now(),
                },
            )
            rows.append(row_from_judge(record, evaluation, result.parsed_json, "completed"))
            print(f"[{idx}/{len(records)}] {record.specialty}__{record.case_id}: completed")
        except Exception as exc:  # noqa: BLE001
            rows.append(row_from_judge(record, {}, {}, "failed", f"{type(exc).__name__}: {exc}"))
            print(f"[{idx}/{len(records)}] {record.specialty}__{record.case_id}: failed: {exc}")

    write_csv(output_dir / "case_quality_summary.csv", rows)
    aggregate = aggregate_rows(rows, usage)
    aggregate["judge_deployment"] = deployment
    aggregate["episode_root"] = str(args.episode_root)
    save_json(output_dir / "_aggregate.json", aggregate)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an interactive reconstruction FVCU audit.")
    parser.add_argument("--episode-root", type=Path, default=DEFAULT_EPISODE_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=None, help="Optional final-scoring case_metrics.csv with episode_path and case_path.")
    parser.add_argument("--case-manifest", type=Path, default=None, help="Optional manifest with fixed specialty/case_id records.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--deployment", default="")
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
