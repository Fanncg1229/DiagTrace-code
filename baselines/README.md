# Fixed-checklist baseline

The baseline issues the same diagnosis-independent request sequence for every case and does not adapt to the initial presentation, known diagnosis, or previous responses. It uses the unchanged report agent and the same 15-turn cap (14 acquisition requests plus the forced final trajectory turn).

- `run_real_checklist_baseline.py`: exact experiment policy and original execution harness; API and authorized case resources required.
- `score_checklist_vs_models.py`: exact case-level and paired scoring workflow used with saved model traces.
- `bootstrap_checklist_ci.py`: paired bootstrap confidence intervals.

These provenance scripts retain the original artifact-layout assumptions so the camera-ready run can be audited. They are not part of the offline quick start. For a clean manifest-based deterministic rerun, use `evaluation/score_manifest.py` after generating authorized checklist episodes.
