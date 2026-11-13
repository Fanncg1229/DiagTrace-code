"""Non-adaptive fixed-checklist baseline run through the REAL report agent.

The clinician is replaced by a scripted policy that issues the same request sequence for every
case, ignoring the initial presentation, the known diagnosis and all prior report responses.
The report agent is the unmodified LLM report agent used in the paper, so the baseline and the
models face the identical environment.

Because the policy is non-adaptive, the first k requests are identical at any budget, so one run
at MAX_TURNS can be truncated to any smaller budget for budget-matched comparison.

  python3 eacl_benchmark/rebuttal/run_real_checklist_baseline.py --n 30
  python3 eacl_benchmark/rebuttal/run_real_checklist_baseline.py --score-only
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "interactive_diagnosis_workspace"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

BASE = ROOT / "eacl_benchmark/results/setting2_model_comparison/final_scoring"
OUT = ROOT / "eacl_benchmark/rebuttal/real_checklist_baseline"
MAX_TURNS = 15          # 14 request turns + 1 FINAL_TRAJECTORY
REQUEST_TURNS = 14

# Fixed, case-independent request sequence: a conventional clinical acquisition order.
# Diagnosis-agnostic and non-adaptive - identical text for every case, never conditioned on the
# known diagnosis, prior report responses, corpus statistics or case content.
# Deviation from the spec (documented): the harness forces FINAL_TRAJECTORY on the final turn, so
# a 15-turn cap yields 14 request turns for the models and for this policy alike. The spec's
# turn-15 (second pathology) step is therefore dropped to keep the budget identical.
# ORDER_TEST / ORDER_PROCEDURE are not used: by protocol they record an order without releasing
# a documented result. REQUEST_RESULT is the retrieval action; WAIT_REASSESS advances the
# documented clinical course.
POLICY: list[tuple[str, str]] = [
    ("ASK_HISTORY",    "Please provide the documented relevant clinical history."),
    ("REQUEST_EXAM",   "Please provide the documented physical examination findings."),
    ("REQUEST_RESULT", "Please provide the documented laboratory findings/results."),
    ("REQUEST_RESULT", "Please provide the documented microbiology findings/results."),
    ("REQUEST_RESULT", "Please provide the documented imaging or radiology findings."),
    ("REQUEST_RESULT", "Please provide the documented pathology or histopathology findings."),
    ("REQUEST_RESULT", "Please provide the documented procedure or intervention findings."),
    ("WAIT_REASSESS",  "Please provide the documented treatment or management response."),
    ("WAIT_REASSESS",  "Please provide the documented subsequent clinical course."),
    ("ASK_HISTORY",    "Please provide the documented relevant clinical history."),
    ("REQUEST_EXAM",   "Please provide the documented physical examination findings."),
    ("REQUEST_RESULT", "Please provide the documented laboratory findings/results."),
    ("REQUEST_RESULT", "Please provide the documented microbiology findings/results."),
    ("REQUEST_RESULT", "Please provide the documented imaging or radiology findings."),
]
assert len(POLICY) == REQUEST_TURNS


class ScriptedClinicianClient:
    """Duck-typed chat client that replays POLICY regardless of the prompt it receives."""

    def __init__(self) -> None:
        self.turn = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model: str, messages: list[dict[str, str]], **_: object):
        if self.turn < len(POLICY):
            action, request = POLICY[self.turn]
            payload = {
                "action": action,
                "request": request,
                "current_trajectory_hypothesis": "fixed non-adaptive checklist policy",
                "confirmed_evidence": [],
                "missing_key_evidence": [],
                "confidence_that_diagnosis_is_supported": 0.0,
            }
        else:
            payload = {
                "action": "FINAL_TRAJECTORY",
                "final_diagnosis": "",
                "trajectory_summary": "fixed non-adaptive checklist policy",
                "key_supporting_evidence": [],
                "ruled_out_or_less_likely_alternatives": [],
                "remaining_uncertainties": [],
                "confidence_that_diagnosis_is_supported": 0.0,
            }
        self.turn += 1
        msg = SimpleNamespace(content=json.dumps(payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0))


def sample_cases(n: int) -> list[tuple[str, Path]]:
    sys.path.insert(0, str(ROOT / "eacl_benchmark/rebuttal"))
    from run_report_agent_swap import sample_cases as shared
    return shared(n)


def run_one(case_id: str, case_path: Path, report_model: str) -> dict:
    from interactive_diagnosis_workspace.trajectory_workflow.workflow import TrajectoryWorkflow
    out_path = OUT / "episodes" / f"{case_id}.json"
    if out_path.exists():
        return {"case_id": case_id, "status": "existing"}
    case = json.loads(case_path.read_text())
    wf = TrajectoryWorkflow(
        case_data=case,
        clinician_client=ScriptedClinicianClient(),
        clinician_model="fixed-checklist-policy",
        report_model=report_model,
        max_turns=MAX_TURNS,
    )
    try:
        episode = wf.run_episode()
    except Exception as exc:  # noqa: BLE001
        return {"case_id": case_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2))
    return {"case_id": case_id, "status": "completed"}


def run(n: int, workers: int, report_model: str) -> None:
    cases = sample_cases(n)
    (OUT / "episodes").mkdir(parents=True, exist_ok=True)
    (OUT / "cases_staged").mkdir(parents=True, exist_ok=True)
    for cid, src in cases:
        (OUT / "cases_staged" / f"{cid}.json").write_text(Path(src).read_text())
    print(f"fixed-checklist baseline: {len(cases)} cases, {REQUEST_TURNS} request turns, "
          f"report agent = {report_model}")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_one, cid, src, report_model): cid for cid, src in cases}
        for fut in as_completed(futs):
            r = fut.result(); done += 1
            print(f"  [{done}/{len(cases)}] {r['case_id']}: {r['status']}"
                  + (f" {r.get('error','')}" if r["status"] == "failed" else ""))


def score() -> None:
    """Budget-matched comparison: truncate the checklist trace to each model's own turn count."""
    eps = {}
    for p in sorted((OUT / "episodes").glob("*.json")):
        ep = json.loads(p.read_text())
        eps[ep["case_id"]] = ep
    if not eps:
        print("no baseline episodes yet"); return

    def gold(case_id: str) -> set[str]:
        case = json.loads((OUT / "cases_staged" / f"{case_id}.json").read_text())
        return {e["id"] for e in case.get("evidence_items") or []
                if int(e.get("phase_index", 0) or 0) > 0}

    def released(ep: dict, k: int) -> set[str]:
        out = set()
        for t in (ep.get("turns") or [])[:k]:
            rep = t.get("report") or {}
            if str(rep.get("status", "")).lower() == "found" and rep.get("evidence_id"):
                out.add(rep["evidence_id"])
        return out

    models = [("gpt41mini_target510", "GPT-4.1 mini"), ("claude_haiku45_target510", "Claude Haiku 4.5"),
              ("gemini25flash_target510", "Gemini 2.5 Flash"), ("kimi_k25_combined", "Kimi K2.5")]
    import numpy as np
    from scipy.stats import wilcoxon

    print(f"\nbudget-matched event recall on {len(eps)} shared cases\n")
    print(f"{'model':18s} {'n':>3s} {'model':>8s} {'checklist':>10s} {'diff':>8s} {'p':>10s}")
    rows = []
    for key, name in models:
        a, b = [], []
        for r in csv.DictReader((BASE / key / "case_metrics.csv").open()):
            if r["status"] != "completed" or r["case_id"] not in eps:
                continue
            g = gold(r["case_id"])
            if not g:
                continue
            ep_model = json.loads((ROOT / r["episode_path"]).read_text())
            turns = int(ep_model.get("turns_used") or 0)
            if not turns:
                continue
            k = max(turns - 1, 1)  # final turn carries no request
            a.append(len(set(ep_model.get("disclosed_evidence_ids") or []) & g) / len(g))
            b.append(len(released(eps[r["case_id"]], k) & g) / len(g))
        if len(a) < 5:
            print(f"{name:18s} too few cases"); continue
        A, B = np.array(a), np.array(b)
        try:
            _, p = wilcoxon(A, B)
        except ValueError:
            p = float("nan")
        print(f"{name:18s} {len(A):3d} {A.mean():8.3f} {B.mean():10.3f} {A.mean()-B.mean():+8.3f} {p:10.4g}")
        rows.append({"model": name, "n": len(A), "model_event_recall": A.mean(),
                     "checklist_event_recall": B.mean(), "diff": A.mean() - B.mean(), "p_wilcoxon": p})
    if rows:
        with (OUT / "budget_matched_comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {OUT/'budget_matched_comparison.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-model", default=os.getenv("AZURE_OPENAI_DEPLOYMENT") or "gpt-4.1-mini-test")
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if not a.score_only:
        run(a.n, a.workers, a.report_model)
    score()
