from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPECTED = {
    "Tversky alignment": ["0.360", "0.334", "0.359", "0.399"],
    "LCS recall": ["0.465", "0.510", "0.500", "0.561"],
    "Pairwise order accuracy": ["0.646", "0.690", "0.654", "0.601"],
    "Precision": ["0.626", "0.560", "0.607", "0.623"],
    "Recall": ["0.525", "0.582", "0.573", "0.672"],
    "F1": ["0.592", "0.589", "0.603", "0.638"],
    "Event-level recall": ["0.577", "0.665", "0.655", "0.661"],
}
MODELS = ["GPT-4.1 mini", "Claude Haiku 4.5", "Gemini 2.5 Flash", "Kimi K2.5"]


def test_rounded_table_matches_full_precision_and_camera_ready_reference() -> None:
    result_dir = ROOT / "results/canonical_matched225"
    with (result_dir / "table3_recomputed_canonical225.csv").open(newline="") as handle:
        full = {row["metric"]: row for row in csv.DictReader(handle)}
    with (result_dir / "table3_recomputed_canonical225_rounded.csv").open(newline="") as handle:
        rounded = {row["metric"]: row for row in csv.DictReader(handle)}

    for metric, expected in EXPECTED.items():
        observed = [rounded[metric][model] for model in MODELS]
        assert observed == expected
        assert observed == [f"{float(full[metric][model]):.3f}" for model in MODELS]


def test_event_recall_is_labeled_macro_and_defined_case_count_is_visible() -> None:
    path = ROOT / "results/canonical_matched225/table3_recomputed_canonical225.csv"
    with path.open(newline="") as handle:
        event_row = next(row for row in csv.DictReader(handle) if row["metric"] == "Event-level recall")
    assert event_row["source"] == "224 defined cases; macro average"


def test_no_case_identifiers_in_public_aggregate_csvs() -> None:
    for path in (ROOT / "results").rglob("*.csv"):
        text = path.read_text()
        assert "PMC10254956" not in text
