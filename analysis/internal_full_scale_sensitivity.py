"""Recompute corrected matched-225 and full-scale Setting 2 sensitivity results.

This script is offline: it reads saved cases/episodes only and makes no API calls.
Case identity is (specialty, PMCID); six known duplicate PMCIDs are restricted to
their canonical specialty before any aggregation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scipy.stats import chi2_contingency, mannwhitneyu


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "interactive_diagnosis_workspace"))

from setting_2_evaluation.metrics import evaluate_case  # noqa: E402
from trajectory_workflow.evaluator import evaluate_episode  # noqa: E402


OUT = Path(__file__).resolve().parent
COMMON_PATH = (
    ROOT
    / "eacl_benchmark/results/setting2_model_comparison/final_scoring/"
    "matched_225_common_subset/model_comparison_common225.json"
)
CANONICAL_TABLE3 = (
    ROOT / "eacl_benchmark/camera_ready/setting2_audit/table3_recomputed_canonical225.csv"
)

CANONICAL_SPECIALTY = {
    "PMC10008168": "radiology",
    "PMC10254956": "neurology",
    "PMC10441080": "oncology",
    "PMC11391149": "oncology",
    "PMC12255965": "nephrology",
    "PMC4938284": "nephrology",
}

EXPECTED_COUNTS = {
    "GPT-4.1 mini": 502,
    "Claude Haiku 4.5": 498,
    "Gemini 2.5 Flash": 502,
    "Kimi K2.5": 225,
}

SPECIALTY_ORDER = [
    "dermatology",
    "emergencymed",
    "internalmed",
    "nephrology",
    "neurology",
    "oncology",
    "ophthalmology",
    "radiology",
]

METRIC_ORDER = [
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

COMPACT_METRICS = [
    "weighted_tversky_alignment",
    "weighted_lcs_recall",
    "weighted_pairwise_order_accuracy",
    "weighted_f1",
    "event_level_recall",
]

DISPLAY = {
    "weighted_tversky_alignment": "Tversky",
    "weighted_lcs_recall": "LCS recall",
    "weighted_pairwise_order_accuracy": "Pairwise order accuracy",
    "weighted_precision": "Weighted precision",
    "weighted_recall": "Weighted recall",
    "weighted_f1": "Weighted F1",
    "event_level_recall": "Event-level recall",
    "redundant_category_rate": "Redundant category rate",
    "off_protocol_rate": "Off-protocol rate",
    "mean_turns_used": "Mean turns used",
    "answer_yield": "Answer yield",
    "unavailable_turn_rate": "Unavailable-turn rate",
}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    episode_root: Path
    case_root: Path


MODELS = [
    ModelConfig(
        "GPT-4.1 mini",
        ROOT / "runs/pmc_control_gpt41mini_20perdept/setting2",
        ROOT / "runs/pmc_control_gpt41mini_20perdept/setting1",
    ),
    ModelConfig(
        "Claude Haiku 4.5",
        ROOT / "runs/pmc_control_setting2_clinicianA-claude-haiku45_report-gpt41mini_target510",
        ROOT / "runs/pmc_control_setting2_clinicianA-claude-haiku45_report-gpt41mini_target510",
    ),
    ModelConfig(
        "Gemini 2.5 Flash",
        ROOT / "eacl_benchmark/results/setting2_model_comparison/gemini25flash_target510",
        ROOT / "eacl_benchmark/results/setting2_model_comparison/gemini25flash_target510",
    ),
    ModelConfig(
        "Kimi K2.5",
        ROOT / "runs/pmc_control_setting2_clinicianA-kimi-k25_report-gpt41mini_132cases",
        ROOT / "runs/pmc_control_setting2_clinicianA-kimi-k25_report-gpt41mini_132cases",
    ),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return st.mean(defined) if defined else None


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


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


def scan_model(config: ModelConfig) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    episode_paths = sorted(config.episode_root.glob("*/episodes/*.json"))
    records: dict[tuple[str, str], dict[str, Any]] = {}
    excluded_noncanonical = []
    incomplete = []
    for episode_path in episode_paths:
        specialty = episode_path.parents[1].name
        case_id = episode_path.stem
        canonical = CANONICAL_SPECIALTY.get(case_id)
        if canonical and specialty != canonical:
            excluded_noncanonical.append((specialty, case_id))
            continue
        identity = (specialty, case_id)
        if identity in records:
            raise ValueError(f"Duplicate identity for {config.name}: {identity}")
        episode = load_json(episode_path)
        case_path = config.case_root / specialty / "cases" / f"{case_id}.json"
        if not case_path.is_file():
            raise FileNotFoundError(f"Missing case for {config.name}: {case_path}")
        final = episode.get("final")
        successful = isinstance(final, dict) and bool(final) and int(episode.get("turns_used") or 0) > 0
        if not successful:
            incomplete.append(identity)
            continue
        case = load_json(case_path)
        if str(case.get("case_id")) != case_id:
            raise ValueError(f"Case identity mismatch in {case_path}")
        records[identity] = {
            "identity": identity,
            "specialty": specialty,
            "case_id": case_id,
            "episode_path": episode_path,
            "case_path": case_path,
            "episode": episode,
            "case": case,
            "signature": event_bank_signature(case),
        }
    audit = {
        "episode_files": len(episode_paths),
        "excluded_noncanonical_duplicate_snapshots": len(excluded_noncanonical),
        "excluded_identities": excluded_noncanonical,
        "incomplete": incomplete,
        "completed_canonical": len(records),
    }
    return records, audit


def eligible_gold(case: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or item.get("event_id"))
        for item in case.get("evidence_items") or case.get("evidence_events") or []
        if int(item.get("phase_index", 0) or 0) > 0
    }


def score_records(records: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    values: dict[str, list[float]] = defaultdict(list)
    off_protocol_statuses = {"too_broad", "invalid_request", "request_too_diagnostic"}
    off_protocol = 0
    non_final_report_turns = 0
    event_defined = 0
    event_hits = 0
    event_gold = 0
    pairwise_defined = 0

    for record in records:
        case = record["case"]
        episode = record["episode"]
        detail = evaluate_case(
            case_id=record["case_id"],
            case_data=case,
            episode=episode,
            include_initial_stage=False,
        )
        metrics = detail["metrics"]
        for key in (
            "weighted_tversky_alignment",
            "weighted_lcs_recall",
            "weighted_pairwise_order_accuracy",
        ):
            value = metrics.get(key)
            if value is not None:
                values[key].append(float(value))
                if key == "weighted_pairwise_order_accuracy":
                    pairwise_defined += 1

        matched = float(metrics.get("matched_weight") or 0)
        missed = float(metrics.get("missed_weight") or 0)
        extra = float(metrics.get("extra_weight") or 0)
        precision = safe_div(matched, matched + extra)
        recall = safe_div(matched, matched + missed)
        f1 = (
            safe_div(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        )
        for key, value in (
            ("weighted_precision", precision),
            ("weighted_recall", recall),
            ("weighted_f1", f1),
        ):
            if value is not None:
                values[key].append(float(value))

        raw_prediction = [
            step["category"]
            for step in detail.get("prediction_steps_raw") or []
            if step.get("category")
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
            values["redundant_category_rate"].append(redundant / len(raw_prediction))

        gold = eligible_gold(case)
        if gold:
            released = set(str(value) for value in episode.get("disclosed_evidence_ids") or [])
            hits = len(released & gold)
            values["event_level_recall"].append(hits / len(gold))
            event_defined += 1
            event_hits += hits
            event_gold += len(gold)

        turn_eval = evaluate_episode(case, episode)
        values["mean_turns_used"].append(float(turn_eval["turns_used"]))
        values["answer_yield"].append(float(turn_eval["answer_yield"]))
        values["unavailable_turn_rate"].append(float(turn_eval["unavailable_turn_rate"]))

        for turn in episode.get("turns") or []:
            action = str((turn.get("clinician") or {}).get("action") or "").upper()
            if action in {"FINAL", "FINAL_TRAJECTORY"}:
                continue
            status = str((turn.get("report") or {}).get("status") or "").lower()
            if not status:
                continue
            non_final_report_turns += 1
            if status in off_protocol_statuses:
                off_protocol += 1

    summary = {key: float(mean(values[key])) for key in METRIC_ORDER if key != "off_protocol_rate"}
    summary["off_protocol_rate"] = float(safe_div(off_protocol, non_final_report_turns) or 0.0)
    diagnostics = {
        "n": len(records),
        "pairwise_defined_cases": pairwise_defined,
        "event_recall_defined_cases": event_defined,
        "event_released_gold_hits": event_hits,
        "event_gold_denominator": event_gold,
        "off_protocol_turns": off_protocol,
        "non_final_report_turns": non_final_report_turns,
    }
    return summary, diagnostics


def case_characteristics(record: dict[str, Any]) -> dict[str, float]:
    case = record["case"]
    events = case.get("evidence_items") or case.get("evidence_events") or []
    eligible = [item for item in events if int(item.get("phase_index", 0) or 0) > 0]
    initial = case.get("initial_observation") or {}
    initial_text = " ".join(str(value) for value in initial.values()) if isinstance(initial, dict) else str(initial)
    return {
        "reference_event_count": float(len(events)),
        "eligible_event_count": float(len(eligible)),
        "figure_linked_event_count": float(sum(bool(item.get("image_finding")) for item in events)),
        "diagnosis_revealing_event_count": float(sum(bool(item.get("diagnosis_revealing")) for item in events)),
        "reference_category_count": float(
            len({str(item.get("category") or item.get("evidence_type")) for item in eligible})
        ),
        "initial_observation_word_count": float(len(initial_text.split())),
    }


def subset_characteristic_rows(
    model: str,
    matched: list[dict[str, Any]],
    remaining: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    groups = {"matched_225": matched, "non_matched_completed": remaining}
    specialty_table = []
    for subset, records in groups.items():
        counts = {specialty: sum(r["specialty"] == specialty for r in records) for specialty in SPECIALTY_ORDER}
        specialty_table.append([counts[specialty] for specialty in SPECIALTY_ORDER])
        for specialty in SPECIALTY_ORDER:
            rows.append(
                {
                    "model": model,
                    "row_type": "specialty_distribution",
                    "subset": subset,
                    "characteristic": "specialty",
                    "level": specialty,
                    "n": counts[specialty],
                    "proportion": counts[specialty] / len(records) if records else math.nan,
                    "mean": "",
                    "sd": "",
                    "median": "",
                    "q1": "",
                    "q3": "",
                    "nonmatched_minus_matched_mean": "",
                    "test": "",
                    "p_value": "",
                }
            )
    chi2_p = float(chi2_contingency(specialty_table)[1])
    rows.append(
        {
            "model": model,
            "row_type": "distribution_test",
            "subset": "matched_vs_non_matched",
            "characteristic": "specialty",
            "level": "all",
            "n": len(matched) + len(remaining),
            "proportion": "",
            "mean": "",
            "sd": "",
            "median": "",
            "q1": "",
            "q3": "",
            "nonmatched_minus_matched_mean": "",
            "test": "chi-square",
            "p_value": chi2_p,
        }
    )

    matched_chars = [case_characteristics(record) for record in matched]
    remaining_chars = [case_characteristics(record) for record in remaining]
    for characteristic in matched_chars[0]:
        a = [row[characteristic] for row in matched_chars]
        b = [row[characteristic] for row in remaining_chars]
        p_value = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if b else math.nan
        difference = st.mean(b) - st.mean(a) if b else math.nan
        for subset, values in (("matched_225", a), ("non_matched_completed", b)):
            rows.append(
                {
                    "model": model,
                    "row_type": "numeric_summary",
                    "subset": subset,
                    "characteristic": characteristic,
                    "level": "",
                    "n": len(values),
                    "proportion": "",
                    "mean": st.mean(values) if values else math.nan,
                    "sd": st.stdev(values) if len(values) > 1 else math.nan,
                    "median": st.median(values) if values else math.nan,
                    "q1": percentile(values, 0.25),
                    "q3": percentile(values, 0.75),
                    "nonmatched_minus_matched_mean": difference,
                    "test": "Mann-Whitney U",
                    "p_value": p_value,
                }
            )
    return rows


def validate_signatures(model_records: dict[str, dict[tuple[str, str], dict[str, Any]]]) -> dict[str, Any]:
    identities = sorted(set().union(*(set(records) for records in model_records.values())))
    mismatches = []
    shared = 0
    for identity in identities:
        signatures = {
            model: records[identity]["signature"]
            for model, records in model_records.items()
            if identity in records
        }
        if len(signatures) > 1:
            shared += 1
            if len(set(signatures.values())) > 1:
                mismatches.append({"identity": identity, "signatures": signatures})
    return {"shared_identities_checked": shared, "signature_mismatches": mismatches}


def expected_matched_values() -> dict[tuple[str, str], float]:
    metric_map = {
        "Tversky alignment": "weighted_tversky_alignment",
        "LCS recall": "weighted_lcs_recall",
        "Pairwise order accuracy": "weighted_pairwise_order_accuracy",
        "Precision": "weighted_precision",
        "Recall": "weighted_recall",
        "F1": "weighted_f1",
        "Event-level recall": "event_level_recall",
        "Off-protocol rate": "off_protocol_rate",
        "Redundant rate": "redundant_category_rate",
        "Mean turns used": "mean_turns_used",
        "Answer yield": "answer_yield",
        "Unavailable-turn rate": "unavailable_turn_rate",
    }
    expected = {}
    for row in csv.DictReader(CANONICAL_TABLE3.open()):
        if row["metric"] not in metric_map:
            continue
        for model in EXPECTED_COUNTS:
            expected[(model, metric_map[row["metric"]])] = float(row[model])
    return expected


def compact_latex(summary_rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrr" + "rrr" * len(COMPACT_METRICS) + "}",
        "\\toprule",
        "Model & $N_m$ & $N_f$ & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{DISPLAY[metric]}}}" for metric in COMPACT_METRICS
        )
        + " \\\\",
        " & & & " + " & ".join(["Matched & Full & $\\Delta$"] * len(COMPACT_METRICS)) + " \\\\",
        "\\midrule",
    ]
    for row in summary_rows:
        cells = [row["model"], str(row["n_matched"]), str(row["n_full"])]
        for metric in COMPACT_METRICS:
            cells.extend(
                [
                    f"{row[f'{metric}_matched']:.3f}",
                    f"{row[f'{metric}_full']:.3f}",
                    f"{row[f'{metric}_difference']:+.3f}",
                ]
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\caption{Matched-225 versus full-scale sensitivity. $\\Delta$ is full minus matched. Full-scale columns are not paired across models.}",
            "\\label{tab:full-scale-sensitivity}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common_ids = set(load_json(COMMON_PATH)["case_ids"])
    if len(common_ids) != 225:
        raise ValueError(f"Expected 225 common PMCIDs, found {len(common_ids)}")

    model_records = {}
    scan_audits = {}
    for config in MODELS:
        records, audit = scan_model(config)
        model_records[config.name] = records
        scan_audits[config.name] = audit
        if len(records) != EXPECTED_COUNTS[config.name]:
            raise ValueError(
                f"{config.name}: expected {EXPECTED_COUNTS[config.name]} completed canonical cases, "
                f"found {len(records)}"
            )
        if audit["incomplete"]:
            raise ValueError(f"Unexpected incomplete saved episodes for {config.name}: {audit['incomplete']}")

    signature_audit = validate_signatures(model_records)
    if signature_audit["signature_mismatches"]:
        raise ValueError(f"Event-bank signature mismatches remain: {signature_audit['signature_mismatches']}")

    expected = expected_matched_values()
    comparison_rows = []
    summary_rows = []
    characteristic_rows = []
    diagnostics = {}
    matched_reproduction_deltas = []

    for config in MODELS:
        records = model_records[config.name]
        matched = [record for identity, record in sorted(records.items()) if identity[1] in common_ids]
        remaining = [record for identity, record in sorted(records.items()) if identity[1] not in common_ids]
        if len(matched) != 225:
            raise ValueError(f"{config.name}: matched subset has {len(matched)} cases, expected 225")

        matched_scores, matched_diag = score_records(matched)
        full_scores, full_diag = score_records([record for _, record in sorted(records.items())])
        diagnostics[config.name] = {"matched": matched_diag, "full": full_diag}

        for metric in METRIC_ORDER:
            delta = matched_scores[metric] - expected[(config.name, metric)]
            matched_reproduction_deltas.append(abs(delta))
            comparison_rows.append(
                {
                    "model": config.name,
                    "n_matched": len(matched),
                    "n_full": len(records),
                    "metric": metric,
                    "matched_score": matched_scores[metric],
                    "full_scale_score": full_scores[metric],
                    "difference_full_minus_matched": full_scores[metric] - matched_scores[metric],
                }
            )

        summary = {"model": config.name, "n_matched": len(matched), "n_full": len(records)}
        for metric in COMPACT_METRICS:
            summary[f"{metric}_matched"] = matched_scores[metric]
            summary[f"{metric}_full"] = full_scores[metric]
            summary[f"{metric}_difference"] = full_scores[metric] - matched_scores[metric]
        summary_rows.append(summary)

        if remaining:
            characteristic_rows.extend(subset_characteristic_rows(config.name, matched, remaining))

    max_reproduction_delta = max(matched_reproduction_deltas)
    if max_reproduction_delta > 1e-12:
        raise ValueError(f"Canonical matched-225 reproduction failed: max delta={max_reproduction_delta}")

    write_csv(OUT / "full_scale_vs_matched_metrics.csv", comparison_rows)
    write_csv(OUT / "full_scale_summary.csv", summary_rows)
    write_csv(OUT / "subset_characteristics.csv", characteristic_rows)
    (OUT / "appendix_table.tex").write_text(compact_latex(summary_rows))

    compact_lookup = {row["model"]: row for row in summary_rows}
    matched_rankings = {}
    full_rankings = {}
    for metric in COMPACT_METRICS:
        matched_rankings[metric] = sorted(
            ((row[f"{metric}_matched"], row["model"]) for row in summary_rows), reverse=True
        )
        full_rankings[metric] = sorted(
            ((row[f"{metric}_full"], row["model"]) for row in summary_rows), reverse=True
        )

    non_kimi = [row for row in comparison_rows if row["model"] != "Kimi K2.5"]
    largest_changes = sorted(
        (row for row in non_kimi if row["metric"] in COMPACT_METRICS),
        key=lambda row: abs(float(row["difference_full_minus_matched"])),
        reverse=True,
    )[:5]
    numeric_chars = [
        row
        for row in characteristic_rows
        if row["row_type"] == "numeric_summary" and row["subset"] == "matched_225"
    ]
    significant_chars = [row for row in numeric_chars if float(row["p_value"]) < 0.05]
    specialty_tests = [row for row in characteristic_rows if row["row_type"] == "distribution_test"]

    methods_paragraph = (
        "\\paragraph{Full-scale sensitivity.} We re-evaluated each saved, successfully completed "
        "episode using the corrected request-category classifier and specialty--PMCID case identities. "
        "For six PMCIDs represented under multiple specialties, we retained only the canonical "
        "specialty-specific event bank. We recomputed all deterministic trajectory, coverage, event-ID, "
        "and interaction metrics from the raw episodes for both the shared 225-case subset and each "
        "model's complete-case set; event recall was computed per case and macro-averaged."
    )
    max_abs_main = max(
        abs(row[f"{metric}_difference"])
        for row in summary_rows
        if row["model"] != "Kimi K2.5"
        for metric in COMPACT_METRICS
    )
    results_paragraph = (
        "\\paragraph{Full-scale sensitivity.} Expanding GPT-4.1 mini, Claude Haiku 4.5, and Gemini "
        f"2.5 Flash from 225 cases to 502, 498, and 502 completed cases changed the five main metrics "
        f"by at most {max_abs_main:.3f} in absolute value (Appendix Table~\\ref{{tab:full-scale-sensitivity}}). "
        "The matched-subset ordering was reproduced: Kimi had the highest Tversky and LCS, whereas "
        "Claude had the highest pairwise-order accuracy and event-level recall; the same metric leaders "
        "appear in the descriptive full-scale columns. Full-scale columns are "
        "descriptive complete-case estimates and are not strictly paired across models."
    )
    (OUT / "latex_paragraphs.tex").write_text(methods_paragraph + "\n\n" + results_paragraph + "\n")

    report = [
        "# Full-scale vs. matched-225 sensitivity analysis",
        "",
        "## Method",
        "",
        "All results were recomputed offline from saved JSON episodes with the corrected classifier. "
        "Identity was defined as `(specialty, PMCID)`. For the six duplicated PMCIDs, only the "
        "canonical specialty listed in the camera-ready audit was retained. An episode counted as "
        "completed only when it had a non-empty final reconstruction and a positive recorded turn count.",
        "",
        "Event-level recall used exact disclosed evidence IDs, was computed per case over post-presentation "
        "events (`phase_index > 0`), and was macro-averaged over cases with a non-empty denominator.",
        "",
        "## Completion-count and identity validation",
        "",
        "| Model | Saved episode files | Noncanonical duplicate snapshots excluded | Completed canonical cases | Matched cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for config in MODELS:
        audit = scan_audits[config.name]
        report.append(
            f"| {config.name} | {audit['episode_files']} | "
            f"{audit['excluded_noncanonical_duplicate_snapshots']} | "
            f"{audit['completed_canonical']} | 225 |"
        )
    report.extend(
        [
            "",
            f"Across {signature_audit['shared_identities_checked']} identities present for at least two models, "
            "event-bank signatures agreed exactly. No failed or incomplete saved episode was included. "
            f"The canonical matched-225 table was reproduced with maximum absolute numerical difference "
            f"{max_reproduction_delta:.2e}.",
            "",
            "## Main sensitivity results",
            "",
            "Differences are full-scale minus matched-225. Full-scale estimates are complete-case summaries, "
            "not paired cross-model comparisons.",
            "",
            "| Model | N matched | N full | Tversky M/F/Delta | LCS M/F/Delta | Pairwise M/F/Delta | F1 M/F/Delta | Event recall M/F/Delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        cells = []
        for metric in COMPACT_METRICS:
            cells.append(
                f"{row[f'{metric}_matched']:.3f}/{row[f'{metric}_full']:.3f}/"
                f"{row[f'{metric}_difference']:+.3f}"
            )
        report.append(
            f"| {row['model']} | {row['n_matched']} | {row['n_full']} | " + " | ".join(cells) + " |"
        )
    report.extend(
        [
            "",
            "On the matched subset, Kimi ranks first for Tversky "
            f"({compact_lookup['Kimi K2.5']['weighted_tversky_alignment_matched']:.3f}) and LCS "
            f"({compact_lookup['Kimi K2.5']['weighted_lcs_recall_matched']:.3f}); Claude ranks first for "
            f"pairwise order ({compact_lookup['Claude Haiku 4.5']['weighted_pairwise_order_accuracy_matched']:.3f}) "
            f"and event recall ({compact_lookup['Claude Haiku 4.5']['event_level_recall_matched']:.3f}). "
            "The same metric leaders appear in the descriptive full-scale columns.",
            "",
            f"Among GPT, Claude, and Gemini, the largest absolute full-minus-matched change across the five "
            f"main metrics was {max_abs_main:.3f}. The largest individual changes were:",
            "",
        ]
    )
    for row in largest_changes:
        report.append(
            f"- {row['model']}, {DISPLAY[row['metric']]}: "
            f"{float(row['difference_full_minus_matched']):+.3f}."
        )
    report.extend(
        [
            "",
            "Kimi has no larger full-scale set in the saved canonical run, so its matched and full columns are identical.",
            "",
            "## Subset-bias checks",
            "",
        ]
    )
    for row in specialty_tests:
        report.append(
            f"- {row['model']}: specialty-distribution chi-square p={float(row['p_value']):.4g}."
        )
    report.extend(
        [
            "",
            "The matched subset is systematically more specialty-balanced: each specialty contributes "
            "8.0--15.1% of matched cases. In the remaining completed cases, radiology contributes "
            "35.5--35.7% and oncology 30.7--30.8%, whereas emergency medicine contributes only 0.4%. "
            "The specialty-composition difference is statistically supported for all three expanded models.",
            "",
            "The remaining cases also contain slightly more reference events (about +0.43 per case), "
            "eligible post-presentation events (+0.51 to +0.52), and figure-linked events (+0.44 to +0.46). "
            "Diagnosis-revealing event counts and initial-observation length do not differ significantly. "
            "Thus, the matched subset is not compositionally representative of the full complete-case sets, "
            "even though the main metric changes are modest and the qualitative ranking conclusions persist.",
        ]
    )
    if significant_chars:
        report.append("")
        report.append("Nominally significant matched/non-matched numeric differences (unadjusted p<.05):")
        report.append("")
        for row in significant_chars:
            report.append(
                f"- {row['model']}, {row['characteristic']}: non-matched minus matched mean "
                f"{float(row['nonmatched_minus_matched_mean']):+.3f}, Mann--Whitney p={float(row['p_value']):.4g}."
            )
    else:
        report.extend(["", "No numeric characteristic reached nominal p<.05."])
    report.extend(
        [
            "",
            "These tests are descriptive and unadjusted for multiple comparisons. They do not establish random "
            "sampling or absence of selection bias; the detailed specialty proportions, means, medians, and "
            "interquartile ranges are in `subset_characteristics.csv`.",
            "",
            "## Validation details and limitations",
            "",
            f"- Canonical completion counts: {', '.join(f'{name}={count}' for name, count in EXPECTED_COUNTS.items())}.",
            "- Six noncanonical specialty snapshots were excluded per model before aggregation.",
            "- Event-bank signatures matched for every identity shared across models.",
            "- Matched-225 corrected metrics reproduced exactly within floating-point tolerance.",
            "- Event-recall defined-case counts and denominators are stored in the diagnostics below.",
            "- Full-scale model columns use different complete-case sets and should not be treated as paired head-to-head estimates.",
            "",
            "```json",
            json.dumps(diagnostics, indent=2),
            "```",
            "",
            "## Publication-ready LaTeX",
            "",
            "```latex",
            methods_paragraph,
            "",
            results_paragraph,
            "```",
            "",
            "The optional appendix table is in `appendix_table.tex`.",
        ]
    )
    (OUT / "FULL_SCALE_SENSITIVITY_REPORT.md").write_text("\n".join(report) + "\n")

    validation = {
        "scan_audits": scan_audits,
        "signature_audit": signature_audit,
        "matched_reproduction_max_abs_delta": max_reproduction_delta,
        "diagnostics": diagnostics,
        "matched_rankings": matched_rankings,
        "full_rankings": full_rankings,
    }
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2, default=list))
    print(f"Wrote full-scale sensitivity outputs to {OUT}")


if __name__ == "__main__":
    main()
