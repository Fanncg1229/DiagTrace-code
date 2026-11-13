# Evaluation

- `score_manifest.py` is the portable, offline scorer for authorized case/episode manifests. It computes corrected category metrics, weighted P/R/F1, exact-ID macro event recall, and turn metrics; it also checks reference snapshot identity.
- `plot_category_recall_heatmap.py` regenerates Figure 2 from the corrected aggregate CSV.
- `internal_camera_ready_audit.py` is the exact internal audit/provenance script used to produce the canonical outputs. It retains the original artifact-layout assumptions and requires excluded case-level artifacts.

FVCU is intentionally not recomputed here. It is an auxiliary LLM-judged metric; deterministic metrics do not depend on that judge.
