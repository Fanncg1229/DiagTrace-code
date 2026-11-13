# Full-scale sensitivity analysis

`internal_full_scale_sensitivity.py` is the exact offline camera-ready analysis script. It distinguishes case identity by `(specialty, PMCID)`, applies the canonical specialty mapping for duplicated PMCIDs, validates cross-model reference snapshots, and reports matched-225 versus all-completed metrics and subset characteristics.

The script retains the original private artifact-layout assumptions for provenance. It makes no API calls, but it cannot run from this redacted package because case resources and episodes are excluded. The portable manifest scorer in `evaluation/score_manifest.py` implements the corrected deterministic metric path for authorized artifacts.
