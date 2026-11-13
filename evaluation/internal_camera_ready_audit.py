"""Audit Setting 2 snapshots and recompute camera-ready metrics without API calls."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "interactive_diagnosis_workspace"))

from setting_2_evaluation.metrics import evaluate_case  # noqa: E402
from trajectory_workflow.evaluator import evaluate_episode  # noqa: E402


BASE = ROOT / "eacl_benchmark/results/setting2_model_comparison/final_scoring"
CHECKLIST = ROOT / "eacl_benchmark/rebuttal/real_checklist_baseline"
OUT = ROOT / "eacl_benchmark/camera_ready/setting2_audit"
CLAUDE_RUN = ROOT / "runs/pmc_control_setting2_clinicianA-claude-haiku45_report-gpt41mini_target510"
MODELS = [
    ("gpt41mini_target510", "GPT-4.1 mini"),
    ("claude_haiku45_target510", "Claude Haiku 4.5"),
    ("gemini25flash_target510", "Gemini 2.5 Flash"),
    ("kimi_k25_combined", "Kimi K2.5"),
]
METRICS = [
    "weighted_tversky_alignment",
    "weighted_lcs_recall",
    "weighted_pairwise_order_accuracy",
]
FVCU_DIRS = {
    "GPT-4.1 mini": "matched225_100case_gpt41mini",
    "Claude Haiku 4.5": "matched225_100case_claude_haiku45",
    "Gemini 2.5 Flash": "matched225_100case_gemini25flash",
    "Kimi K2.5": "matched225_100case_kimi_k25",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rows_for(model_key: str) -> dict[str, dict[str, str]]:
    path = BASE / model_key / "case_metrics.csv"
    return {
        row["case_id"]: row
        for row in csv.DictReader(path.open())
        if row["status"] == "completed"
    }


def event_bank_signature(case: dict[str, Any]) -> str:
    bank = [
        {
            "id": item.get("id") or item.get("event_id"),
            "phase_index": item.get("phase_index"),
            "category": item.get("category") or item.get("evidence_type"),
            "source_text": item.get("source_text") or item.get("source_span") or item.get("text"),
        }
        for item in case.get("evidence_items") or case.get("evidence_events") or []
    ]
    payload = json.dumps(bank, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def snapshot_audit(
    common: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> tuple[list[dict[str, Any]], set[str]]:
    manifest: list[dict[str, Any]] = []
    mismatched: set[str] = set()
    for case_id in sorted(common):
        signatures: dict[str, str] = {}
        for key, name in MODELS:
            row = model_rows[key][case_id]
            case_path = ROOT / row["case_path"]
            case = load_json(case_path)
            signature = event_bank_signature(case)
            signatures[name] = signature
            manifest.append(
                {
                    "case_id": case_id,
                    "model": name,
                    "case_path": row["case_path"],
                    "specialty": case_path.parts[-3],
                    "event_count": len(case.get("evidence_items") or case.get("evidence_events") or []),
                    "event_bank_sha256": signature,
                }
            )
        if len(set(signatures.values())) > 1:
            mismatched.add(case_id)
    for row in manifest:
        row["snapshot_consistent"] = row["case_id"] not in mismatched
    return manifest, mismatched


def canonicalize_claude_rows(
    mismatched: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> tuple[dict[str, dict[str, dict[str, str]]], list[dict[str, Any]]]:
    canonical = {
        key: {case_id: dict(row) for case_id, row in rows.items()}
        for key, rows in model_rows.items()
    }
    replacements = []
    for case_id in sorted(mismatched):
        gpt_row = canonical["gpt41mini_target510"][case_id]
        canonical_specialty = (ROOT / gpt_row["case_path"]).parts[-3]
        case_path = CLAUDE_RUN / canonical_specialty / "cases" / f"{case_id}.json"
        episode_path = CLAUDE_RUN / canonical_specialty / "episodes" / f"{case_id}.json"
        if not case_path.is_file() or not episode_path.is_file():
            raise FileNotFoundError(f"Missing canonical Claude artifacts for {case_id}: {canonical_specialty}")
        gpt_signature = event_bank_signature(load_json(ROOT / gpt_row["case_path"]))
        claude_signature = event_bank_signature(load_json(case_path))
        if gpt_signature != claude_signature:
            raise ValueError(f"Canonical event banks still differ for {case_id}")
        old = canonical["claude_haiku45_target510"][case_id]
        replacements.append(
            {
                "case_id": case_id,
                "old_specialty": (ROOT / old["case_path"]).parts[-3],
                "canonical_specialty": canonical_specialty,
                "old_case_path": old["case_path"],
                "canonical_case_path": str(case_path.relative_to(ROOT)),
                "old_episode_path": old["episode_path"],
                "canonical_episode_path": str(episode_path.relative_to(ROOT)),
            }
        )
        canonical["claude_haiku45_target510"][case_id]["case_path"] = str(case_path.relative_to(ROOT))
        canonical["claude_haiku45_target510"][case_id]["episode_path"] = str(episode_path.relative_to(ROOT))
    return canonical, replacements


def mean(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return st.mean(valid) if valid else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def first_unique(sequence: list[str]) -> list[str]:
    out = []
    seen = set()
    previous = None
    for item in sequence:
        if item == previous:
            continue
        previous = item
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def category_metrics(
    case_ids: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    out = []
    for key, name in MODELS:
        stored: dict[str, list[float]] = defaultdict(list)
        corrected: dict[str, list[float]] = defaultdict(list)
        for case_id in sorted(case_ids):
            row = model_rows[key][case_id]
            case = load_json(ROOT / row["case_path"])
            episode = load_json(ROOT / row["episode_path"])
            result = evaluate_case(
                case_id=case_id,
                case_data=case,
                episode=episode,
                include_initial_stage=False,
            )
            for metric in METRICS:
                if row.get(metric):
                    stored[metric].append(float(row[metric]))
                value = result["metrics"].get(metric)
                if value is not None:
                    corrected[metric].append(float(value))
        record: dict[str, Any] = {"model": name, "n": len(case_ids)}
        for metric in METRICS:
            old = mean(stored[metric])
            new = mean(corrected[metric])
            record[f"{metric}_stored"] = old
            record[f"{metric}_corrected"] = new
            record[f"{metric}_delta"] = new - old if old is not None and new is not None else None
        out.append(record)
    return out


def eligible_gold(case: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or item.get("event_id"))
        for item in case.get("evidence_items") or case.get("evidence_events") or []
        if int(item.get("phase_index", 0) or 0) > 0
    }


def checklist_released(episode: dict[str, Any], turns: int) -> set[str]:
    released = set()
    non_final = [
        turn
        for turn in episode.get("turns") or []
        if str((turn.get("clinician") or {}).get("action") or "").upper()
        not in {"FINAL", "FINAL_TRAJECTORY"}
    ]
    for turn in non_final[:turns]:
        report = turn.get("report") or {}
        if str(report.get("status") or "").lower() == "found" and report.get("evidence_id"):
            released.add(str(report["evidence_id"]))
    return released


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def checklist_sensitivity(
    consistent_ids: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    checklist_episodes = {
        path.stem: load_json(path) for path in (CHECKLIST / "episodes").glob("PMC*.json")
    }
    sample_ids = sorted(set(checklist_episodes) & consistent_ids)
    results = []
    rng = np.random.default_rng(20260918)
    for key, name in MODELS:
        pairs = []
        for case_id in sample_ids:
            row = model_rows[key][case_id]
            case = load_json(ROOT / row["case_path"])
            episode = load_json(ROOT / row["episode_path"])
            gold = eligible_gold(case)
            if not gold:
                continue
            acquisition_turns = max(int(episode.get("turns_used") or 1) - 1, 1)
            model_released = set(episode.get("disclosed_evidence_ids") or [])
            checklist_hits = checklist_released(checklist_episodes[case_id], acquisition_turns)
            model_recall = len(model_released & gold) / len(gold)
            checklist_recall = len(checklist_hits & gold) / len(gold)
            pairs.append((model_recall, checklist_recall))

        differences = [model - checklist for model, checklist in pairs]
        values = np.array(differences)
        indices = rng.integers(0, len(values), size=(10_000, len(values)))
        boot = values[indices].mean(axis=1).tolist()
        results.append(
            {
                "model": name,
                "n": len(pairs),
                "model_event_recall": st.mean(model for model, _ in pairs),
                "checklist_event_recall": st.mean(checklist for _, checklist in pairs),
                "mean_difference": st.mean(differences),
                "ci_low": percentile(boot, 0.025),
                "ci_high": percentile(boot, 0.975),
                "cases_model_better": sum(value > 0 for value in differences),
                "cases_checklist_better": sum(value < 0 for value in differences),
                "cases_tied": sum(value == 0 for value in differences),
            }
        )
    return results


def event_recall(case_ids: set[str], model_rows: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    results = []
    for key, name in MODELS:
        recalls = []
        hits = 0
        gold_total = 0
        for case_id in sorted(case_ids):
            row = model_rows[key][case_id]
            case = load_json(ROOT / row["case_path"])
            episode = load_json(ROOT / row["episode_path"])
            gold = eligible_gold(case)
            if not gold:
                continue
            released = set(episode.get("disclosed_evidence_ids") or [])
            case_hits = len(released & gold)
            recalls.append(case_hits / len(gold))
            hits += case_hits
            gold_total += len(gold)
        results.append(
            {
                "model": name,
                "n_defined": len(recalls),
                "macro_event_recall": st.mean(recalls),
                "pooled_event_recall": hits / gold_total,
                "released_gold_events": hits,
                "gold_events": gold_total,
            }
        )
    return results


def checklist_category_metrics(
    model_rows: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    checklist_episodes = {
        path.stem: load_json(path) for path in (CHECKLIST / "episodes").glob("PMC*.json")
    }
    results = []
    values: dict[str, list[float]] = defaultdict(list)
    for case_id, episode in checklist_episodes.items():
        case = load_json(CHECKLIST / "cases_staged" / f"{case_id}.json")
        detail = evaluate_case(
            case_id=case_id,
            case_data=case,
            episode=episode,
            include_initial_stage=False,
        )
        for metric in METRICS:
            value = detail["metrics"].get(metric)
            if value is not None:
                values[metric].append(float(value))
    results.append(
        {"policy_or_model": "Fixed Checklist", "n": len(checklist_episodes),
         **{metric: mean(values[metric]) for metric in METRICS}}
    )

    sample_ids = set(checklist_episodes)
    for key, name in MODELS:
        values = defaultdict(list)
        for case_id in sorted(sample_ids):
            row = model_rows[key][case_id]
            detail = evaluate_case(
                case_id=case_id,
                case_data=load_json(ROOT / row["case_path"]),
                episode=load_json(ROOT / row["episode_path"]),
                include_initial_stage=False,
            )
            for metric in METRICS:
                value = detail["metrics"].get(metric)
                if value is not None:
                    values[metric].append(float(value))
        results.append(
            {"policy_or_model": name, "n": len(sample_ids),
             **{metric: mean(values[metric]) for metric in METRICS}}
        )
    return results


def extra_category_metrics(
    case_ids: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = []
    category_rows = []
    off_protocol_statuses = {"too_broad", "invalid_request", "request_too_diagnostic"}
    for key, name in MODELS:
        precision_values = []
        recall_values = []
        f1_values = []
        redundant_values = []
        category_gold: dict[str, int] = defaultdict(int)
        category_hit: dict[str, int] = defaultdict(int)
        off_protocol = 0
        non_final_turns = 0
        for case_id in sorted(case_ids):
            row = model_rows[key][case_id]
            case = load_json(ROOT / row["case_path"])
            episode = load_json(ROOT / row["episode_path"])
            detail = evaluate_case(
                case_id=case_id,
                case_data=case,
                episode=episode,
                include_initial_stage=False,
            )
            metrics = detail["metrics"]
            matched = float(metrics.get("matched_weight") or 0)
            missed = float(metrics.get("missed_weight") or 0)
            extra = float(metrics.get("extra_weight") or 0)
            precision = safe_div(matched, matched + extra)
            recall = safe_div(matched, matched + missed)
            f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
            if precision is not None:
                precision_values.append(precision)
            if recall is not None:
                recall_values.append(recall)
            if f1 is not None:
                f1_values.append(f1)

            raw_prediction = [
                step["category"] for step in detail.get("prediction_steps_raw") or [] if step.get("category")
            ]
            seen = set()
            previous = None
            redundant = 0
            for category in raw_prediction:
                if category == previous:
                    continue
                previous = category
                if category in seen:
                    redundant += 1
                else:
                    seen.add(category)
            if raw_prediction:
                redundant_values.append(redundant / len(raw_prediction))

            gold_sequence = first_unique([str(value) for value in detail.get("gold_sequence") or []])
            prediction_set = set(first_unique([str(value) for value in detail.get("prediction_sequence") or []]))
            for category in gold_sequence:
                category_gold[category] += 1
                if category in prediction_set:
                    category_hit[category] += 1

            for turn in episode.get("turns") or []:
                action = str((turn.get("clinician") or {}).get("action") or "").upper()
                if action in {"FINAL", "FINAL_TRAJECTORY"}:
                    continue
                status = str((turn.get("report") or {}).get("status") or "").lower()
                if not status:
                    continue
                non_final_turns += 1
                if status in off_protocol_statuses:
                    off_protocol += 1

        summaries.append(
            {
                "model": name,
                "n": len(case_ids),
                "weighted_precision": mean(precision_values),
                "weighted_recall": mean(recall_values),
                "weighted_f1": mean(f1_values),
                "off_protocol_request_rate": safe_div(off_protocol, non_final_turns),
                "redundant_category_rate": mean(redundant_values),
            }
        )
        for category in sorted(category_gold):
            category_rows.append(
                {
                    "model": name,
                    "category": category,
                    "gold_cases": category_gold[category],
                    "hit_cases": category_hit[category],
                    "category_recall": safe_div(category_hit[category], category_gold[category]),
                }
            )
    return summaries, category_rows


def turn_metrics(
    case_ids: set[str], model_rows: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    results = []
    for key, name in MODELS:
        evaluations = []
        for case_id in sorted(case_ids):
            row = model_rows[key][case_id]
            evaluations.append(
                evaluate_episode(
                    load_json(ROOT / row["case_path"]),
                    load_json(ROOT / row["episode_path"]),
                )
            )
        results.append(
            {
                "model": name,
                "n": len(evaluations),
                "mean_turns_used": mean([float(item["turns_used"]) for item in evaluations]),
                "mean_answer_yield": mean([float(item["answer_yield"]) for item in evaluations]),
                "mean_unavailable_turn_rate": mean(
                    [float(item["unavailable_turn_rate"]) for item in evaluations]
                ),
            }
        )
    return results


def fvcu_existing_judgments(
    canonical_rows: dict[str, dict[str, dict[str, str]]]
) -> list[dict[str, Any]]:
    base = ROOT / "eacl_benchmark/results/interactive_fvcu_quality"
    results = []
    for key, name in MODELS:
        directory = base / FVCU_DIRS[name]
        rows = list(csv.DictReader((directory / "case_quality_summary.csv").open()))
        completed = [row for row in rows if row["judge_status"] == "completed"]
        stale = []
        for row in completed:
            case_id = row["case_id"]
            canonical_episode = canonical_rows[key][case_id]["episode_path"]
            if row["episode_file"] != canonical_episode:
                stale.append(case_id)
        results.append(
            {
                "model": name,
                "n_judged": len(completed),
                "canonical_episode_judgments": len(completed) - len(stale),
                "stale_case_ids": ";".join(sorted(stale)),
                "factuality": mean([float(row["fvcu_factuality"]) for row in completed]),
                "validity": mean([float(row["fvcu_validity"]) for row in completed]),
                "coherence": mean([float(row["fvcu_coherence"]) for row in completed]),
                "utility": mean([float(row["fvcu_utility"]) for row in completed]),
            }
        )
    return results


def table3_rows(
    category: list[dict[str, Any]],
    extra: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fvcu: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_model = {
        "category": {row["model"]: row for row in category},
        "extra": {row["model"]: row for row in extra},
        "events": {row["model"]: row for row in events},
        "fvcu": {row["model"]: row for row in fvcu},
        "turns": {row["model"]: row for row in turns},
    }
    model_names = [name for _, name in MODELS]
    specifications = [
        ("Request-category alignment", "Tversky alignment", "category", "weighted_tversky_alignment_corrected", "225-case deterministic"),
        ("Request-category alignment", "LCS recall", "category", "weighted_lcs_recall_corrected", "225-case deterministic"),
        ("Request-category alignment", "Pairwise order accuracy", "category", "weighted_pairwise_order_accuracy_corrected", "225-case deterministic"),
        ("Coverage decomposition and request behavior", "Precision", "extra", "weighted_precision", "225-case deterministic"),
        ("Coverage decomposition and request behavior", "Recall", "extra", "weighted_recall", "225-case deterministic"),
        ("Coverage decomposition and request behavior", "F1", "extra", "weighted_f1", "225-case deterministic"),
        ("Coverage decomposition and request behavior", "Event-level recall", "events", "macro_event_recall", "224 defined cases; macro average"),
        ("Coverage decomposition and request behavior", "Off-protocol rate", "extra", "off_protocol_request_rate", "225-case deterministic"),
        ("Coverage decomposition and request behavior", "Redundant rate", "extra", "redundant_category_rate", "225-case deterministic"),
        ("Interactive FVCU quality", "Factuality", "fvcu", "factuality", "100-case existing LLM judgments"),
        ("Interactive FVCU quality", "Validity", "fvcu", "validity", "100-case existing LLM judgments"),
        ("Interactive FVCU quality", "Coherence", "fvcu", "coherence", "100-case existing LLM judgments"),
        ("Interactive FVCU quality", "Utility", "fvcu", "utility", "100-case existing LLM judgments"),
        ("Turn efficiency and availability", "Mean turns used", "turns", "mean_turns_used", "225-case deterministic"),
        ("Turn efficiency and availability", "Answer yield", "turns", "mean_answer_yield", "225-case deterministic"),
        ("Turn efficiency and availability", "Unavailable-turn rate", "turns", "mean_unavailable_turn_rate", "225-case deterministic"),
    ]
    output = []
    for section, metric, source, key, note in specifications:
        row = {"section": section, "metric": metric, "source": note}
        for model in model_names:
            row[model] = by_model[source][model][key]
        output.append(row)
    return output


def rounded_table3_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        digits = 2 if row["section"] == "Interactive FVCU quality" or row["metric"] == "Mean turns used" else 3
        rounded = dict(row)
        for _, model in MODELS:
            rounded[model] = f"{float(row[model]):.{digits}f}"
        output.append(rounded)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = set(load_json(BASE / "matched_225_common_subset/model_comparison_common225.json")["case_ids"])
    model_rows = {key: rows_for(key) for key, _ in MODELS}
    manifest, mismatched = snapshot_audit(common, model_rows)
    consistent = common - mismatched
    canonical_rows, replacements = canonicalize_claude_rows(mismatched, model_rows)
    _, remaining_mismatches = snapshot_audit(common, canonical_rows)
    if remaining_mismatches:
        raise ValueError(f"Canonicalization left mismatches: {sorted(remaining_mismatches)}")

    write_csv(OUT / "snapshot_manifest.csv", manifest)
    write_csv(OUT / "claude_canonical_replacements.csv", replacements)
    (OUT / "snapshot_mismatches.json").write_text(
        json.dumps({"count": len(mismatched), "case_ids": sorted(mismatched)}, indent=2)
    )
    write_csv(OUT / "category_metrics_corrected_matched225.csv", category_metrics(common, model_rows))
    write_csv(
        OUT / "category_metrics_corrected_snapshot_consistent.csv",
        category_metrics(consistent, model_rows),
    )
    canonical_category = category_metrics(common, canonical_rows)
    canonical_events = event_recall(common, canonical_rows)
    write_csv(OUT / "category_metrics_corrected_canonical225.csv", canonical_category)
    write_csv(OUT / "event_recall_canonical225.csv", canonical_events)
    write_csv(OUT / "checklist_snapshot_consistent15.csv", checklist_sensitivity(consistent, model_rows))
    write_csv(OUT / "checklist_canonical16.csv", checklist_sensitivity(common, canonical_rows))
    write_csv(
        OUT / "checklist_category_metrics_corrected_canonical16.csv",
        checklist_category_metrics(canonical_rows),
    )
    extra_summary, category_recall = extra_category_metrics(common, canonical_rows)
    write_csv(OUT / "extra_category_metrics_corrected_canonical225.csv", extra_summary)
    write_csv(OUT / "category_recall_corrected_canonical225.csv", category_recall)
    canonical_turns = turn_metrics(common, canonical_rows)
    fvcu = fvcu_existing_judgments(canonical_rows)
    write_csv(OUT / "turn_metrics_recomputed_canonical225.csv", canonical_turns)
    write_csv(OUT / "fvcu_existing_judgments_reaggregated.csv", fvcu)
    table3 = table3_rows(canonical_category, extra_summary, canonical_events, fvcu, canonical_turns)
    write_csv(OUT / "table3_recomputed_canonical225.csv", table3)
    write_csv(OUT / "table3_recomputed_canonical225_rounded.csv", rounded_table3_rows(table3))

    print(f"snapshot mismatches: {len(mismatched)} -> {', '.join(sorted(mismatched))}")
    print(f"snapshot-consistent matched subset: {len(consistent)}")
    print(f"canonicalized Claude snapshots: {len(replacements)}; remaining mismatches: 0")
    print(f"wrote camera-ready audit outputs to {OUT}")


if __name__ == "__main__":
    main()
