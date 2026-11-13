"""Aggregate saved FVCU judge outputs without making model calls."""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path


MODELS = {
    "gpt_4_1_mini": "GPT-4.1 mini",
    "claude_haiku_4_5": "Claude Haiku 4.5",
    "gemini_2_5_flash": "Gemini 2.5 Flash",
    "kimi_k2_5": "Kimi K2.5",
}
DIMENSIONS = ("factuality", "validity", "coherence", "utility")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for slug, model in MODELS.items():
        scores = {dimension: [] for dimension in DIMENSIONS}
        for path in sorted((args.root / slug).glob("*.json")):
            payload = json.loads(path.read_text())
            result = payload.get("judge_output") or {}
            if result.get("judge_status") != "completed":
                continue
            fvcu = result.get("interactive_fvcu") or {}
            for dimension in DIMENSIONS:
                scores[dimension].append(float((fvcu.get(dimension) or {})["score"]))
        counts = {len(values) for values in scores.values()}
        if counts != {100}:
            raise ValueError(f"Expected 100 complete judgments for {model}, got {counts}")
        rows.append({"model": model, "n_judged": 100, **{key: st.mean(scores[key]) for key in DIMENSIONS}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
