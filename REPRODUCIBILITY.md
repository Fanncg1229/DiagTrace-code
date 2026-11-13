# Reproducibility guide

## Reproducibility levels

This release distinguishes three levels rather than overstating what can be reproduced publicly:

1. **Fully offline:** package installation, corrected classifier regression tests, deterministic scoring on the synthetic fixture, Figure 2 rendering from aggregate data, and aggregate-result consistency checks.
2. **Offline with the separate data release:** Table 3 deterministic metrics, fixed-checklist scoring, and full-scale sensitivity can be recomputed when the reviewed camera-ready data addendum is installed in the public data repository.
3. **API dependent:** reference-resource generation/audit, new clinician/report-agent episodes, and new FVCU judgments require provider credentials and may vary with model/version changes.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

No credential is needed for offline scoring. `.env.example` lists placeholders only. Never commit a populated `.env`.

## Manifest contract

Each JSONL record supplied to `evaluation/score_manifest.py` has:

```json
{
  "model": "model display name",
  "specialty": "canonical specialty",
  "case_id": "canonical PMCID or synthetic ID",
  "case_path": "relative/path/to/case.json",
  "episode_path": "relative/path/to/episode.json",
  "matched": true
}
```

Identity is `(model, specialty, case_id)`. The scorer checks duplicate identities, completed final outputs, matching case IDs, and identical event-bank signatures for the same specialty-case across models. This prevents a trajectory from being scored against a different reference snapshot.

## Deterministic metrics

Run:

```bash
python evaluation/score_manifest.py \
  --manifest /path/to/manifest.jsonl \
  --artifact-root /path/to/artifacts \
  --output-dir /path/to/output
```

`metrics.csv` contains matched and full-scale macro averages. `diagnostics.json` records case counts, event-recall denominators, report turns, and off-protocol turns. Event-level recall is calculated per case from exact eligible evidence IDs and then macro-averaged; it is not a pooled micro-recall.

## Figure 2

```bash
python evaluation/plot_category_recall_heatmap.py \
  --input results/canonical_matched225/category_recall_corrected_canonical225.csv \
  --output-dir /tmp/diagtrace-figure2
```

## FVCU

FVCU values in the canonical CSV are saved aggregate LLM-judge outputs. `construction/run_interactive_fvcu_audit.py` documents the judging workflow, but running it creates new model calls and is outside offline reproduction. Core Tversky, LCS, pairwise, category P/R/F1, event recall, and interaction statistics do not depend on the FVCU judge.

## Known limitations

- Real case resources and traces are maintained in the separate data repository; the audited public commit still lacks the camera-ready addendum described in `docs/DATA_REPOSITORY_AUDIT.md`.
- Model-backed generation cannot be made bitwise deterministic across changing hosted endpoints.
- The construction-stage 5,911-event corpus and the Setting 2 report-agent event bank use different schemas and denominators; see `docs/EVENT_DEFINITIONS.md`.
- The included full-scale script records the exact internal artifact workflow. It requires the original saved artifact hierarchy or a local adaptation of its input paths; aggregate outputs are included for inspection.
