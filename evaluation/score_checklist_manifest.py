"""Offline fixed-checklist scoring from canonical data-release manifests."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from interactive_diagnosis_workspace.setting_2_evaluation.metrics import evaluate_case


MODEL_ORDER = ["GPT-4.1 mini", "Claude Haiku 4.5", "Gemini 2.5 Flash", "Kimi K2.5"]
METRICS = ["weighted_tversky_alignment", "weighted_lcs_recall", "weighted_pairwise_order_accuracy"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def eligible_ids(case: dict[str, Any]) -> set[str]:
    return {
        str(item.get("id") or item.get("event_id"))
        for item in case.get("evidence_items") or case.get("evidence_events") or []
        if int(item.get("phase_index", 0) or 0) > 0
    }


def released_ids(episode: dict[str, Any], acquisition_turns: int | None = None) -> set[str]:
    turns = [
        turn
        for turn in episode.get("turns") or []
        if str((turn.get("clinician") or {}).get("action") or "").upper()
        not in {"FINAL", "FINAL_TRAJECTORY"}
    ]
    if acquisition_turns is not None:
        turns = turns[:acquisition_turns]
    return {
        str((turn.get("report") or {}).get("evidence_id"))
        for turn in turns
        if str((turn.get("report") or {}).get("status") or "").lower() == "found"
        and (turn.get("report") or {}).get("evidence_id")
    }


def percentile(values: np.ndarray, probability: float) -> float:
    return float(np.percentile(values, probability * 100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checklist-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checklist_rows = load_jsonl(args.checklist_manifest)
    model_rows = load_jsonl(args.model_manifest)
    checklist: dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in checklist_rows:
        identity = (str(row["specialty"]), str(row["case_id"]))
        checklist[identity] = (
            load_json(resolve(args.artifact_root, str(row["case_path"]))),
            load_json(resolve(args.artifact_root, str(row["episode_path"]))),
        )

    models: dict[str, dict[tuple[str, str], tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for row in model_rows:
        identity = (str(row["specialty"]), str(row["case_id"]))
        if identity not in checklist:
            continue
        models[str(row["model"])][identity] = (
            load_json(resolve(args.artifact_root, str(row["case_path"]))),
            load_json(resolve(args.artifact_root, str(row["episode_path"]))),
        )

    expected = set(checklist)
    for model in MODEL_ORDER:
        if set(models[model]) != expected:
            raise ValueError(f"Checklist/model identity mismatch for {model}")

    rng = np.random.default_rng(20260918)
    event_rows = []
    for model in MODEL_ORDER:
        pairs = []
        for identity in sorted(expected, key=lambda value: value[1]):
            case, model_episode = models[model][identity]
            checklist_case, checklist_episode = checklist[identity]
            if eligible_ids(case) != eligible_ids(checklist_case):
                raise ValueError(f"Checklist event-bank mismatch: {identity}")
            gold = eligible_ids(case)
            if not gold:
                continue
            turns = max(int(model_episode.get("turns_used") or 1) - 1, 1)
            model_recall = len(released_ids(model_episode) & gold) / len(gold)
            checklist_recall = len(released_ids(checklist_episode, turns) & gold) / len(gold)
            pairs.append((model_recall, checklist_recall))
        differences = np.array([model_value - checklist_value for model_value, checklist_value in pairs])
        indices = rng.integers(0, len(differences), size=(10_000, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        event_rows.append(
            {
                "model": model,
                "n": len(pairs),
                "model_event_recall": st.mean(value for value, _ in pairs),
                "checklist_event_recall": st.mean(value for _, value in pairs),
                "mean_difference": float(differences.mean()),
                "ci_low": percentile(bootstrap, 0.025),
                "ci_high": percentile(bootstrap, 0.975),
                "cases_model_better": int((differences > 0).sum()),
                "cases_checklist_better": int((differences < 0).sum()),
                "cases_tied": int((differences == 0).sum()),
            }
        )

    category_rows = []
    values: dict[str, list[float]] = defaultdict(list)
    for identity, (case, episode) in checklist.items():
        result = evaluate_case(case_id=identity[1], case_data=case, episode=episode, include_initial_stage=False)
        for metric in METRICS:
            if result["metrics"].get(metric) is not None:
                values[metric].append(float(result["metrics"][metric]))
    category_rows.append(
        {"policy_or_model": "Fixed Checklist", "n": len(checklist), **{key: st.mean(values[key]) for key in METRICS}}
    )
    for model in MODEL_ORDER:
        values = defaultdict(list)
        for identity, (case, episode) in models[model].items():
            result = evaluate_case(case_id=identity[1], case_data=case, episode=episode, include_initial_stage=False)
            for metric in METRICS:
                if result["metrics"].get(metric) is not None:
                    values[metric].append(float(result["metrics"][metric]))
        category_rows.append(
            {"policy_or_model": model, "n": len(models[model]), **{key: st.mean(values[key]) for key in METRICS}}
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("checklist_event_comparison.csv", event_rows), ("checklist_category_metrics.csv", category_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
