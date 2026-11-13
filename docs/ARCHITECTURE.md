# Architecture and code map

| Function | Canonical release location | Notes |
|---|---|---|
| Reference conversion | `src/diagnostic_narrowing/milestone_converter.py` | Converts structured milestones/evidence into the diagnostic-narrowing schema. |
| Reference audit | `construction/run_reference_resource_audit.py` | LLM-assisted audit; API dependent. |
| FVCU audit | `construction/run_interactive_fvcu_audit.py` | Auxiliary LLM judging; separate from deterministic metrics. |
| Clinician/report interaction | `src/interactive_diagnosis_workspace/trajectory_workflow/workflow.py` | Source-bounded interaction orchestration. |
| Episode execution | `trajectory_workflow/runner.py` | Provider-backed execution. |
| Request classifier | `setting_2_evaluation/metrics.py::categorize_request_text` | Corrected camera-ready word-form patterns. |
| Tversky/LCS/pairwise | `setting_2_evaluation/metrics.py` | Deterministic category metrics. |
| Weighted P/R/F1 | `evaluation/score_manifest.py` | Derived per case from matched/extra/missed weighted mass. |
| Exact-ID event recall | `evaluation/score_manifest.py` | Eligible IDs have `phase_index > 0`; macro average. |
| Turn/availability metrics | `trajectory_workflow/evaluator.py` | Deterministic from saved turns/statuses. |
| Figure 2 | `evaluation/plot_category_recall_heatmap.py` | Reads corrected aggregate CSV. |
| Fixed checklist | `baselines/` | Generation, scoring, and bootstrap CI scripts. |
| Full-scale sensitivity | `analysis/internal_full_scale_sensitivity.py` | Exact corrected camera-ready analysis; restricted artifacts required. |

The legacy `interactive_diagnosis/` package is retained because it contains shared schemas, action validation, evidence matching, and earlier workflow utilities imported by the current pipeline. Experiment outputs, caches, and local environments from the working repository were not copied.
