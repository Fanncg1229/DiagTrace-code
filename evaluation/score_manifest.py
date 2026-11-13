"""Offline corrected DiagTrace scoring from a case/episode JSONL manifest.

The script never calls an LLM. Paths in the manifest may be absolute or relative
to --artifact-root. Case identity is the pair (specialty, case_id).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from interactive_diagnosis_workspace.setting_2_evaluation.metrics import evaluate_case
from interactive_diagnosis_workspace.trajectory_workflow.evaluator import evaluate_episode


METRICS = [
    "weighted_tversky_alignment",
    "weighted_lcs_recall",
    "weighted_pairwise_order_accuracy",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "event_level_recall",
    "redundant_category_rate",
    "off_protocol_rate",
    "mean_turns_used",
    "answer_yield",
    "unavailable_turn_rate",
]
OFF_PROTOCOL = {"too_broad", "invalid_request", "request_too_diagnostic"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float]) -> float | None:
    return st.mean(values) if values else None


def signature(case: dict[str, Any]) -> str:
    items = [
        {
            "id": item.get("id") or item.get("event_id"),
            "phase_index": item.get("phase_index"),
            "category": item.get("category") or item.get("evidence_type"),
            "source_text": item.get("source_text") or item.get("source_span") or item.get("text"),
        }
        for item in case.get("evidence_items") or case.get("evidence_events") or []
    ]
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_records(manifest: Path, artifact_root: Path) -> list[dict[str, Any]]:
    records = []
    seen = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        identity = (str(row["model"]), str(row["specialty"]), str(row["case_id"]))
        if identity in seen:
            raise ValueError(f"Duplicate model/specialty/case identity on line {line_number}: {identity}")
        seen.add(identity)
        case_path = resolve(artifact_root, str(row["case_path"]))
        episode_path = resolve(artifact_root, str(row["episode_path"]))
        case = load_json(case_path)
        episode = load_json(episode_path)
        if str(case.get("case_id")) != identity[2] or str(episode.get("case_id")) != identity[2]:
            raise ValueError(f"Case ID mismatch for {identity}")
        if not isinstance(episode.get("final"), dict) or not episode["final"]:
            raise ValueError(f"Incomplete episode for {identity}")
        records.append(
            {
                **row,
                "model": identity[0],
                "specialty": identity[1],
                "case_id": identity[2],
                "matched": bool(row.get("matched", False)),
                "case": case,
                "episode": episode,
                "signature": signature(case),
            }
        )
    return records


def validate_shared_snapshots(records: list[dict[str, Any]]) -> None:
    by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        by_identity[(row["specialty"], row["case_id"])].add(row["signature"])
    mismatches = [identity for identity, values in by_identity.items() if len(values) > 1]
    if mismatches:
        raise ValueError(f"Cross-model event-bank signature mismatches: {mismatches[:10]}")


def score(rows: list[dict[str, Any]]) -> tuple[dict[str, float | None], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)
    off_protocol = 0
    report_turns = 0
    event_defined = 0
    event_hits = 0
    event_gold = 0
    for row in rows:
        case, episode = row["case"], row["episode"]
        result = evaluate_case(
            case_id=row["case_id"],
            case_data=case,
            episode=episode,
            include_initial_stage=False,
        )
        metrics = result["metrics"]
        for key in (
            "weighted_tversky_alignment",
            "weighted_lcs_recall",
            "weighted_pairwise_order_accuracy",
        ):
            if metrics.get(key) is not None:
                values[key].append(float(metrics[key]))

        matched = float(metrics.get("matched_weight") or 0)
        missed = float(metrics.get("missed_weight") or 0)
        extra = float(metrics.get("extra_weight") or 0)
        precision = safe_div(matched, matched + extra)
        recall = safe_div(matched, matched + missed)
        f1 = safe_div(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None
        for key, value in (("weighted_precision", precision), ("weighted_recall", recall), ("weighted_f1", f1)):
            if value is not None:
                values[key].append(float(value))

        prediction = [step["category"] for step in result.get("prediction_steps_raw") or [] if step.get("category")]
        seen, previous, redundant = set(), None, 0
        for category in prediction:
            if category == previous:
                continue
            previous = category
            if category in seen:
                redundant += 1
            else:
                seen.add(category)
        if prediction:
            values["redundant_category_rate"].append(redundant / len(prediction))

        gold = {
            str(item.get("id") or item.get("event_id"))
            for item in case.get("evidence_items") or case.get("evidence_events") or []
            if int(item.get("phase_index", 0) or 0) > 0
        }
        if gold:
            released = set(str(value) for value in episode.get("disclosed_evidence_ids") or [])
            hits = len(gold & released)
            values["event_level_recall"].append(hits / len(gold))
            event_defined += 1
            event_hits += hits
            event_gold += len(gold)

        interaction = evaluate_episode(case, episode)
        values["mean_turns_used"].append(float(interaction["turns_used"]))
        values["answer_yield"].append(float(interaction["answer_yield"]))
        values["unavailable_turn_rate"].append(float(interaction["unavailable_turn_rate"]))
        for turn in episode.get("turns") or []:
            action = str((turn.get("clinician") or {}).get("action") or "").upper()
            if action in {"FINAL", "FINAL_TRAJECTORY"}:
                continue
            status = str((turn.get("report") or {}).get("status") or "").lower()
            if status:
                report_turns += 1
                off_protocol += status in OFF_PROTOCOL

    summary = {key: mean(values[key]) for key in METRICS if key != "off_protocol_rate"}
    summary["off_protocol_rate"] = safe_div(off_protocol, report_turns) or 0.0
    diagnostics = {
        "cases": len(rows),
        "event_recall_defined_cases": event_defined,
        "released_gold_events": event_hits,
        "gold_events": event_gold,
        "report_turns": report_turns,
        "off_protocol_turns": off_protocol,
    }
    return summary, diagnostics


def category_recall(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gold_counts: dict[str, int] = defaultdict(int)
    hit_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        result = evaluate_case(
            case_id=row["case_id"],
            case_data=row["case"],
            episode=row["episode"],
            include_initial_stage=False,
        )
        gold = [str(value) for value in result.get("gold_sequence") or []]
        predicted = {str(value) for value in result.get("prediction_sequence") or []}
        for category in gold:
            gold_counts[category] += 1
            if category in predicted:
                hit_counts[category] += 1
    return [
        {
            "category": category,
            "gold_cases": gold_counts[category],
            "hit_cases": hit_counts[category],
            "category_recall": hit_counts[category] / gold_counts[category],
        }
        for category in sorted(gold_counts)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.manifest, args.artifact_root)
    validate_shared_snapshots(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    category_rows = []
    diagnostics = {}
    for model in sorted({row["model"] for row in records}):
        full = [row for row in records if row["model"] == model]
        matched = [row for row in full if row["matched"]]
        if not matched:
            matched = full
        matched_scores, matched_diag = score(matched)
        full_scores, full_diag = score(full)
        diagnostics[model] = {"matched": matched_diag, "full": full_diag}
        for row in category_recall(matched):
            category_rows.append({"model": model, **row})
        for metric in METRICS:
            comparison.append(
                {
                    "model": model,
                    "n_matched": len(matched),
                    "n_full": len(full),
                    "metric": metric,
                    "matched_score": matched_scores[metric],
                    "full_score": full_scores[metric],
                    "difference_full_minus_matched": (
                        full_scores[metric] - matched_scores[metric]
                        if full_scores[metric] is not None and matched_scores[metric] is not None
                        else None
                    ),
                }
            )
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    with (args.output_dir / "category_recall_matched.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(category_rows[0]))
        writer.writeheader()
        writer.writerows(category_rows)
    (args.output_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
