from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .metrics import evaluate_case, load_json


DEFAULT_OUTPUT_DIR = Path("interactive_diagnosis_workspace/setting_2_evaluation/results/source_trajectory_alignment")


CSV_FIELDS = [
    "case_id",
    "status",
    "episode_path",
    "case_path",
    "num_gold_steps",
    "num_pred_steps",
    "weighted_tversky_alignment",
    "weighted_tversky_distance",
    "matched_weight",
    "missed_weight",
    "extra_weight",
    "weighted_lcs_recall",
    "weighted_lcs_weight",
    "weighted_pairwise_order_accuracy",
    "pairwise_comparable_pairs",
    "pairwise_correct_pairs",
    "pairwise_inversions",
    "gold_sequence",
    "prediction_sequence",
    "error",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def case_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem.split(".")[0]


def collect_episode_paths(paths: list[Path]) -> dict[str, Path]:
    episodes: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and path.name.startswith("PMC") and path.suffix == ".json":
            episodes[case_id_from_path(path)] = path
            continue
        episode_dir = path / "episodes" if (path / "episodes").is_dir() else path
        if not episode_dir.is_dir():
            continue
        for episode_path in sorted(episode_dir.glob("PMC*.json")):
            episodes[case_id_from_path(episode_path)] = episode_path
    return episodes


def collect_case_paths(paths: list[Path]) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for path in paths:
        case_dir = path / "cases" if (path / "cases").is_dir() else path
        if not case_dir.is_dir():
            continue
        for case_path in sorted(case_dir.glob("PMC*.json")):
            if any(suffix in case_path.stem for suffix in (".ddx_alignment_judge", ".final_diagnosis_judge", ".reasoning_judge", ".reasoning_quality")):
                continue
            cases[case_id_from_path(case_path)] = case_path
    return cases


def case_ids_from_manifest(path: Path) -> list[str]:
    manifest = load_json(path)
    ids = []
    for item in manifest.get("cases") or []:
        if isinstance(item, dict) and item.get("case_id"):
            ids.append(str(item["case_id"]))
    return ids


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    metric_keys = [
        "weighted_tversky_alignment",
        "weighted_tversky_distance",
        "weighted_lcs_recall",
        "weighted_pairwise_order_accuracy",
    ]
    summary: dict[str, Any] = {
        "num_cases": len(rows),
        "num_completed": len(completed),
        "num_missing_episode": sum(row.get("status") == "missing_episode" for row in rows),
        "num_missing_case": sum(row.get("status") == "missing_case" for row in rows),
    }
    for key in metric_keys:
        values = finite_values(completed, key)
        summary[f"mean_{key}"] = mean(values) if values else None
    return summary


def evaluate_batch(
    *,
    case_ids: list[str],
    episode_paths: dict[str, Path],
    case_paths: dict[str, Path],
    output_dir: Path,
    alpha: float,
    beta: float,
    include_initial_stage: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for case_id in case_ids:
        episode_path = episode_paths.get(case_id)
        case_path = case_paths.get(case_id)
        base_row: dict[str, Any] = {
            "case_id": case_id,
            "episode_path": str(episode_path or ""),
            "case_path": str(case_path or ""),
            "error": "",
        }
        if episode_path is None:
            rows.append({**base_row, "status": "missing_episode"})
            continue
        if case_path is None:
            rows.append({**base_row, "status": "missing_case"})
            continue
        try:
            detail = evaluate_case(
                case_id=case_id,
                case_data=load_json(case_path),
                episode=load_json(episode_path),
                alpha=alpha,
                beta=beta,
                include_initial_stage=include_initial_stage,
            )
            details[case_id] = detail
            metric_values = detail["metrics"]
            rows.append(
                {
                    **base_row,
                    "status": "completed",
                    "num_gold_steps": detail["num_gold_steps"],
                    "num_pred_steps": detail["num_pred_steps"],
                    "gold_sequence": " > ".join(detail["gold_sequence"]),
                    "prediction_sequence": " > ".join(detail["prediction_sequence"]),
                    **metric_values,
                }
            )
        except Exception as exc:
            rows.append({**base_row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    summary = summarize(rows)
    payload = {
        "created_at": now(),
        "alpha": alpha,
        "beta": beta,
        "include_initial_stage": include_initial_stage,
        "summary": summary,
        "rows": rows,
    }
    write_csv(output_dir / "case_metrics.csv", rows)
    write_json(output_dir / "case_details.json", details)
    write_json(output_dir / "_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Setting 2 source-derived trajectory acquisition order.")
    parser.add_argument("--episode-path", action="append", type=Path, default=[], help="Episode JSON, episodes directory, or run directory.")
    parser.add_argument("--case-dir", action="append", type=Path, default=[], help="Case directory or run directory containing cases/.")
    parser.add_argument("--case-id", action="append", default=[], help="Restrict/evaluate explicit case IDs.")
    parser.add_argument("--case-list-manifest", type=Path, default=None, help="Manifest whose cases define the denominator.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=2.0, help="Tversky overuse penalty.")
    parser.add_argument("--beta", type=float, default=1.0, help="Tversky underuse penalty.")
    parser.add_argument(
        "--exclude-initial-stage",
        action="store_true",
        help="Do not prepend the initial presentation as the first gold/prediction stage.",
    )
    args = parser.parse_args()

    episode_paths = collect_episode_paths(args.episode_path)
    case_paths = collect_case_paths(args.case_dir)
    case_ids = list(args.case_id)
    if args.case_list_manifest:
        case_ids.extend(case_ids_from_manifest(args.case_list_manifest))
    if not case_ids:
        case_ids = sorted(set(episode_paths) | set(case_paths))
    else:
        case_ids = sorted(dict.fromkeys(case_ids))

    payload = evaluate_batch(
        case_ids=case_ids,
        episode_paths=episode_paths,
        case_paths=case_paths,
        output_dir=args.output_dir,
        alpha=args.alpha,
        beta=args.beta,
        include_initial_stage=not args.exclude_initial_stage,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
