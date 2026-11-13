"""Paired bootstrap CIs for model vs turn-matched fixed-checklist event recall (n=16).

Descriptive only: with 16 cases these intervals indicate stability, not significance.
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

OUT = Path("eacl_benchmark/rebuttal")
B = 10000
rng = np.random.default_rng(20260918)

pairs = defaultdict(list)
for r in csv.DictReader((OUT / "checklist_turn_matched.csv").open()):
    if r["paired_difference"]:
        pairs[r["comparison_model"]].append(
            (float(r["model_event_recall"]), float(r["checklist_event_recall_at_matched_budget"])))

rows = []
print(f"paired bootstrap, {B} resamples, n=16 cases\n")
print(f"{'model':18s} {'model':>7s} {'checklist':>10s} {'diff':>8s} {'95% CI':>20s} {'P(diff>0)':>10s}")
for name, ps in pairs.items():
    a = np.array([p[0] for p in ps]); b = np.array([p[1] for p in ps])
    d = a - b
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p_pos = float((boot > 0).mean())
    print(f"{name:18s} {a.mean():7.3f} {b.mean():10.3f} {d.mean():+8.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]".rjust(0) + f" {p_pos:10.3f}")
    rows.append({"model": name, "n": len(d), "model_event_recall": a.mean(),
                 "checklist_matched": b.mean(), "mean_diff": d.mean(),
                 "ci_low": lo, "ci_high": hi, "boot_p_diff_gt_0": p_pos,
                 "cases_model_better": int((d > 0).sum()),
                 "cases_checklist_better": int((d < 0).sum()), "cases_tied": int((d == 0).sum())})

print(f"\n{'model':18s} {'模型更好':>9s} {'清单更好':>9s} {'持平':>6s}")
for r in rows:
    print(f"{r['model']:18s} {r['cases_model_better']:9d} {r['cases_checklist_better']:9d} {r['cases_tied']:6d}")

with (OUT / "checklist_bootstrap_ci.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT/'checklist_bootstrap_ci.csv'}")
