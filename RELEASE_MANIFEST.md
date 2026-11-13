# Release manifest

## Included

- `src/`: Python implementation required by the construction, interaction, and corrected Setting 2 evaluation workflows.
- `construction/`: reference-resource and FVCU audit entry points plus shared judge utilities.
- `evaluation/`: offline manifest scorer, corrected heatmap generator, manifest-based checklist scorer, saved-FVCU aggregator, and exact internal camera-ready audit provenance script.
- `baselines/`: fixed-checklist generation, scoring, and bootstrap scripts.
- `analysis/`: exact corrected full-scale sensitivity script.
- `examples/synthetic/`: invented case, episode, and manifest for offline validation.
- `tests/`: classifier, synthetic workflow, and release-result regression tests.
- `results/canonical_matched225/`: aggregate corrected Table 3 and Figure 2 data/figures.
- `results/fixed_checklist/`: aggregate corrected 16-case checklist results.
- `results/full_scale_sensitivity/`: aggregate full-vs-matched outputs and manuscript-ready fragments.
- Documentation, license, and environment metadata: `README.md`, `REPRODUCIBILITY.md`, `docs/`, `LICENSE`, `pyproject.toml`, `requirements.txt`, `requirements-api.txt`, `.env.example`, and `.gitignore`.

## Deliberately excluded

- Real benchmark JSON/JSONL, PMC text/XML/PDFs, source excerpts, and figures: benchmark data are hosted separately and are not duplicated in the code repository.
- Real episodes, transcripts, and report-agent outputs: maintained in the separate data release; provider/local metadata must remain sanitized.
- API keys and populated environment files: secrets.
- Original experiment caches, virtual environments, logs, temporary files, and model downloads: unnecessary or potentially sensitive.
- The existing working repository's `.git` history: may expose paths, deleted data, or credentials and is not needed for review.
- DOI or ACL Anthology identifier: not yet assigned/verified.

## Provenance and result status

Aggregate CSVs are copied from the corrected camera-ready audit outputs without alteration. They do not contain PMCID-level rows or source text. Deterministic values can be recomputed when an authorized frozen artifact manifest is supplied. FVCU values are retained as saved aggregate LLM judgments and are explicitly identified as such.
