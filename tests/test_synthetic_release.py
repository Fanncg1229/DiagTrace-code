from __future__ import annotations

import json
from pathlib import Path

from interactive_diagnosis_workspace.setting_2_evaluation.metrics import evaluate_case


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_example_scores_without_api() -> None:
    case = json.loads((ROOT / "examples/synthetic/case.json").read_text())
    episode = json.loads((ROOT / "examples/synthetic/episode.json").read_text())
    result = evaluate_case(
        case_id="SYNTHETIC001",
        case_data=case,
        episode=episode,
        include_initial_stage=False,
    )
    assert result["metrics"]["weighted_tversky_alignment"] is not None
    assert result["metrics"]["weighted_lcs_recall"] is not None
    assert set(episode["disclosed_evidence_ids"]) <= {item["id"] for item in case["evidence_items"]}


def test_examples_are_explicitly_synthetic() -> None:
    case = json.loads((ROOT / "examples/synthetic/case.json").read_text())
    assert case["case_id"].startswith("SYNTHETIC")
    assert "Synthetic" in json.dumps(case)
