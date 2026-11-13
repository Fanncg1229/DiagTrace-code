"""Score the fixed-checklist baseline against the four existing LLM traces on the same cases.

No clinician LLM is rerun: GPT-4.1 mini, Claude Haiku 4.5, Gemini 2.5 Flash and Kimi K2.5 results
are loaded from the existing benchmark outputs. Only the scripted checklist was newly executed,
and it used the unmodified report agent.

Category-level metrics (Tversky / LCS / pairwise) and event-level recall are kept separate:
the former scores requested evidence *types*, the latter scores whether the actual documented
source events were released.
"""
from __future__ import annotations
import csv, json, statistics as st, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "interactive_diagnosis_workspace"))
from setting_2_evaluation.metrics import evaluate_case

BASE = ROOT / "eacl_benchmark/results/setting2_model_comparison/final_scoring"
CL = ROOT / "eacl_benchmark/rebuttal/real_checklist_baseline"
OUT = ROOT / "eacl_benchmark/rebuttal"
MODELS = [("gpt41mini_target510", "GPT-4.1 mini"), ("claude_haiku45_target510", "Claude Haiku 4.5"),
          ("gemini25flash_target510", "Gemini 2.5 Flash"), ("kimi_k25_combined", "Kimi K2.5")]
ANSWERED = {"found", "already_disclosed"}


def eligible_gold(case: dict) -> set[str]:
    """Same eligibility rule as p0_event_and_final.py: post-initial-presentation events."""
    out = set()
    for e in case.get("evidence_items") or []:
        try:
            phase = int(e.get("phase_index", 0) or 0)
        except (TypeError, ValueError):
            phase = 0
        if phase > 0:
            out.add(e["id"])
    return out


def interaction_stats(ep: dict, k: int | None = None) -> dict:
    turns = [t for t in (ep.get("turns") or [])
             if str((t.get("clinician") or {}).get("action") or "").upper() not in {"FINAL", "FINAL_TRAJECTORY"}]
    if k is not None:
        turns = turns[:k]
    n = len(turns)
    answered = unavailable = 0
    for t in turns:
        s = str((t.get("report") or {}).get("status") or "").lower()
        if s in ANSWERED:
            answered += 1
        elif s:
            unavailable += 1
    return {"non_final_turns": n,
            "answer_yield": answered / n if n else None,
            "unavailable_rate": unavailable / n if n else None}


def released(ep: dict, k: int | None = None) -> set[str]:
    turns = ep.get("turns") or []
    if k is not None:
        turns = [t for t in turns
                 if str((t.get("clinician") or {}).get("action") or "").upper()
                 not in {"FINAL", "FINAL_TRAJECTORY"}][:k]
    out = set()
    for t in turns:
        rep = t.get("report") or {}
        if str(rep.get("status", "")).lower() == "found" and rep.get("evidence_id"):
            out.add(rep["evidence_id"])
    return out


def redundant_rate(detail: dict) -> float | None:
    raw = [s["category"] for s in detail.get("prediction_steps_raw") or []]
    seen, prev, red = set(), None, 0
    for c in raw:
        if c == prev:
            prev = c
            continue
        prev = c
        if c in seen:
            red += 1
        else:
            seen.add(c)
    return red / len(raw) if raw else None


def row_for(name, case_id, specialty, case, ep, detail_metrics):
    gold = eligible_gold(case)
    rel = released(ep)
    stats = interaction_stats(ep)
    turns = int(ep.get("turns_used") or 0)
    return {
        "case_id": case_id, "specialty": specialty, "policy_or_model": name,
        "turns_used": turns,
        "tversky": detail_metrics.get("weighted_tversky_alignment"),
        "lcs": detail_metrics.get("weighted_lcs_recall"),
        "pairwise_order": detail_metrics.get("weighted_pairwise_order_accuracy"),
        "event_recall": len(rel & gold) / len(gold) if gold else None,
        "answer_yield": stats["answer_yield"],
        "unavailable_rate": stats["unavailable_rate"],
        "unique_events_acquired": len(rel & gold),
        "new_events_per_turn": (len(rel & gold) / stats["non_final_turns"]) if stats["non_final_turns"] else None,
    }


def main() -> None:
    cl_eps = {p.stem: json.loads(p.read_text()) for p in sorted((CL / "episodes").glob("*.json"))}
    if not cl_eps:
        print("no checklist episodes yet"); return
    case_ids = set(cl_eps)
    print(f"cases with a checklist episode: {len(case_ids)}")

    rows, matched = [], []
    # --- checklist ---
    for cid, ep in cl_eps.items():
        case = json.loads((CL / "cases_staged" / f"{cid}.json").read_text())
        res = evaluate_case(case_id=cid, case_data=case, episode=ep, include_initial_stage=False)
        r = row_for("Fixed Checklist", cid, "", case, ep, res["metrics"])
        r["redundant_rate"] = redundant_rate(res)
        rows.append(r)

    # --- existing model traces (loaded, never rerun) ---
    for key, name in MODELS:
        detail = json.loads((BASE / key / "case_details.json").read_text())
        for m in csv.DictReader((BASE / key / "case_metrics.csv").open()):
            cid = m["case_id"]
            if m["status"] != "completed" or cid not in case_ids:
                continue
            ep = json.loads((ROOT / m["episode_path"]).read_text())
            case = json.loads((ROOT / m["case_path"]).read_text())
            spec = Path(m["case_path"]).parts[-3]
            d = detail.get(cid, {})
            r = row_for(name, cid, spec, case, ep, d.get("metrics", {}))
            r["redundant_rate"] = redundant_rate(d)
            rows.append(r)
            # turn-matched checklist truncation (no new API calls)
            gold = eligible_gold(case)
            k = max(int(ep.get("turns_used") or 1) - 1, 1)
            cl_r = len(released(cl_eps[cid], k) & gold) / len(gold) if gold else None
            matched.append({"case_id": cid, "specialty": spec, "comparison_model": name,
                            "matched_turn_budget": k,
                            "checklist_event_recall_at_matched_budget": cl_r,
                            "model_event_recall": r["event_recall"],
                            "paired_difference": (r["event_recall"] - cl_r)
                            if (cl_r is not None and r["event_recall"] is not None) else None})

    for cid in case_ids:
        spec = next((r["specialty"] for r in rows if r["case_id"] == cid and r["specialty"]), "")
        for r in rows:
            if r["case_id"] == cid and not r["specialty"]:
                r["specialty"] = spec

    fields = ["case_id", "specialty", "policy_or_model", "turns_used", "tversky", "lcs",
              "pairwise_order", "event_recall", "answer_yield", "unavailable_rate",
              "redundant_rate", "unique_events_acquired", "new_events_per_turn"]
    with (OUT / "checklist_case_level.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows([{k: r.get(k) for k in fields} for r in rows])
    with (OUT / "checklist_turn_matched.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(matched[0].keys())); w.writeheader(); w.writerows(matched)

    def mean(vals):
        v = [x for x in vals if x is not None]
        return st.mean(v) if v else None

    order = ["Fixed Checklist"] + [n for _, n in MODELS]
    print(f"\n=== aggregate over the same {len(case_ids)} cases ===")
    hdr = f"{'policy/model':18s} {'n':>3s} {'turns':>6s} {'Tversky':>8s} {'LCS':>7s} {'pair':>7s} " \
          f"{'evRecall':>9s} {'yield':>7s} {'unavail':>8s} {'redund':>7s} {'uniqEv':>7s} {'ev/turn':>8s}"
    print(hdr)
    agg = []
    for name in order:
        rs = [r for r in rows if r["policy_or_model"] == name]
        if not rs: continue
        a = {"policy_or_model": name, "n": len(rs),
             "turns_used": mean(r["turns_used"] for r in rs),
             "tversky": mean(r["tversky"] for r in rs), "lcs": mean(r["lcs"] for r in rs),
             "pairwise_order": mean(r["pairwise_order"] for r in rs),
             "event_recall": mean(r["event_recall"] for r in rs),
             "answer_yield": mean(r["answer_yield"] for r in rs),
             "unavailable_rate": mean(r["unavailable_rate"] for r in rs),
             "redundant_rate": mean(r["redundant_rate"] for r in rs),
             "unique_events_acquired": mean(r["unique_events_acquired"] for r in rs),
             "new_events_per_turn": mean(r["new_events_per_turn"] for r in rs)}
        agg.append(a)
        print(f"{name:18s} {a['n']:3d} {a['turns_used']:6.2f} {a['tversky']:8.3f} {a['lcs']:7.3f} "
              f"{a['pairwise_order']:7.3f} {a['event_recall']:9.3f} {a['answer_yield']:7.3f} "
              f"{a['unavailable_rate']:8.3f} {a['redundant_rate']:7.3f} "
              f"{a['unique_events_acquired']:7.2f} {a['new_events_per_turn']:8.4f}")
    with (OUT / "checklist_aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys())); w.writeheader(); w.writerows(agg)

    print("\n=== turn-matched: checklist truncated to each model's own per-case turn count ===")
    print(f"{'model':18s} {'n':>3s} {'model':>8s} {'checklist':>10s} {'diff':>8s}")
    for _, name in MODELS:
        ms = [m for m in matched if m["comparison_model"] == name and m["paired_difference"] is not None]
        if not ms: continue
        print(f"{name:18s} {len(ms):3d} {mean(m['model_event_recall'] for m in ms):8.3f} "
              f"{mean(m['checklist_event_recall_at_matched_budget'] for m in ms):10.3f} "
              f"{mean(m['paired_difference'] for m in ms):+8.3f}")

    print("\nClinician LLMs were not rerun; all GPT/Claude/Gemini/Kimi clinician results were "
          "loaded from existing traces.")
    print(f"wrote {OUT/'checklist_case_level.csv'}, {OUT/'checklist_turn_matched.csv'}, "
          f"{OUT/'checklist_aggregate.csv'}")


if __name__ == "__main__":
    main()
