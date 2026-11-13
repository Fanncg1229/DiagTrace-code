# DiagTrace

**DiagTrace: An Interactive Benchmark for Clinical Decision Making**  
Yufan Wang, Rui Yang, Yi Lin, Chunhua Weng, and Yifan Peng  
Accepted to *Findings of the Association for Computational Linguistics: EACL 2027*.

This is a pre-publication reproducibility package. It contains the corrected camera-ready evaluation code, aggregate outputs, safe synthetic examples, and documentation. Benchmark data are distributed separately through the public data repository and are intentionally not duplicated in this code repository.

## 1. Overview

DiagTrace evaluates diagnosis-aware interactive reconstruction of a documented clinical trajectory. The clinician agent receives the initial presentation and known final diagnosis, then requests evidence from a source-bounded report agent. The report agent releases at most one eligible, previously unreleased event per request, selecting the earliest matching event in source order; otherwise it returns an unavailable status. The clinician has a 15-turn budget and ends with a reconstructed trajectory.

The main components are:

1. Reference-resource construction and audit (`construction/` and `src/diagnostic_narrowing/`).
2. Clinician/report-agent interaction (`src/interactive_diagnosis_workspace/trajectory_workflow/`).
3. Corrected Setting 2 classification and deterministic metrics (`src/interactive_diagnosis_workspace/setting_2_evaluation/`).
4. Event-ID and interaction metrics (`evaluation/score_manifest.py` and `trajectory_workflow/evaluator.py`).
5. Fixed-checklist and sensitivity analyses (`baselines/`, `analysis/`).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the code map.

## 2. Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Dependencies are declared in `pyproject.toml` and mirrored in `requirements.txt`. API-backed generation scripts are included for methodological transparency but are not used by the offline quick start or tests.

For provider-backed Gemini generation, install the optional API dependency with `python -m pip install -e '.[api,test]'` (or use `requirements-api.txt`).

## 3. Quick start (synthetic, offline)

```bash
python evaluation/score_manifest.py \
  --manifest examples/synthetic/manifest.jsonl \
  --artifact-root examples/synthetic \
  --output-dir /tmp/diagtrace-synthetic-output

python -m pytest -q
```

The example is explicitly synthetic and is not part of the benchmark or reported results.

## Data repository

The benchmark data repository is [Fanncg1229/Diagtrace](https://github.com/Fanncg1229/Diagtrace). Clone it next to this code repository:

```bash
git clone https://github.com/Fanncg1229/DiagTrace-code.git
git clone https://github.com/Fanncg1229/Diagtrace.git
cd DiagTrace-code
python -m pip install -e '.[test]'
```

The public data repository currently contains the 510 construction resources and four matched sets of 225 interaction traces. At the audited commit, it does **not** yet contain the canonical Setting 2 event banks, six corrected Claude specialty episodes, full-scale episodes, fixed-checklist episodes, or saved FVCU outputs required for end-to-end camera-ready reproduction. See [docs/DATA_REPOSITORY_AUDIT.md](docs/DATA_REPOSITORY_AUDIT.md).

After the reviewed data addendum is installed as `Diagtrace/camera_ready_release`, deterministic matched and full-scale metrics can be recomputed with:

```bash
python evaluation/score_manifest.py \
  --manifest ../Diagtrace/camera_ready_release/manifests/matched225_manifest.jsonl \
  --artifact-root ../Diagtrace/camera_ready_release \
  --output-dir /tmp/diagtrace-matched225

python evaluation/score_manifest.py \
  --manifest ../Diagtrace/camera_ready_release/manifests/full_scale_manifest.jsonl \
  --artifact-root ../Diagtrace/camera_ready_release \
  --output-dir /tmp/diagtrace-full-scale

python evaluation/score_checklist_manifest.py \
  --checklist-manifest ../Diagtrace/camera_ready_release/manifests/fixed_checklist_manifest.jsonl \
  --model-manifest ../Diagtrace/camera_ready_release/manifests/matched225_manifest.jsonl \
  --artifact-root ../Diagtrace/camera_ready_release \
  --output-dir /tmp/diagtrace-checklist

python evaluation/aggregate_fvcu.py \
  --root ../Diagtrace/camera_ready_release/fvcu \
  --output /tmp/diagtrace-fvcu.csv
```

The checklist scorer preserves the seeded PMCID-ordered bootstrap used by the camera-ready audit,
so the saved Monte Carlo confidence intervals reproduce exactly.

Do not use `data/diagtrace_510.jsonl` as the Setting 2 event bank: its 5,911 `evidence_events` belong to the construction resource, whereas Setting 2 is scored against a separate frozen `evidence_items` bank.

## 4. Benchmark construction

The original workflow converts source-grounded case material into ordered evidence items and hidden reference fields, then audits finding coverage, event faithfulness, and reasoning support. Relevant code is in `src/diagnostic_narrowing/` and `construction/`. The audit scripts can consume saved resources and, when configured, call an LLM judge. No API call is needed for deterministic trajectory scoring.

Two event counts must not be conflated:

- Table 1 reports 5,911 construction-stage `evidence_events` over 510 cases (11.59 per case).
- Setting 2 and `subset_characteristics.csv` use the separate report-agent `evidence_items` bank. In the matched 225 cases this bank contains 1,790 items (7.96 per case before the phase filter); event-level recall uses only items with `phase_index > 0`.

These are different schemas/stages, not two summaries of the same denominator. See [docs/EVENT_DEFINITIONS.md](docs/EVENT_DEFINITIONS.md).

## 5. Interactive reconstruction protocol

The interaction implementation is in `trajectory_workflow/workflow.py`, with schemas and runners in the same package. Requests are classified into coarse clinical categories by `setting_2_evaluation/metrics.py`. The camera-ready classifier fixes partial-word matching failures by matching complete morphological forms (for example, *pathology*, *histopathology*, *sonographic*, and *immunosuppressive*).

The report agent is source bounded: returned evidence is selected from the frozen event bank, and final evidence citations are checked against disclosed evidence IDs. To rerun model interactions, users must supply their own legally obtained resources and API credentials. Those runs are not part of the offline validation.

## 6. Evaluation metrics

The deterministic Setting 2 metrics are computed per case and macro-averaged:

- **Tversky alignment:** weighted category overlap, with extra requests weighted by alpha=2 and missed categories by beta=1.
- **LCS recall:** weighted longest-common-subsequence coverage of the gold category order.
- **Pairwise order accuracy:** weighted agreement for comparable category pairs.
- **Precision/Recall/F1:** derived from matched, extra, and missed weighted category mass.
- **Event-level recall:** exact released evidence-ID overlap with eligible (`phase_index > 0`) reference items, computed per defined case and macro-averaged. It does not use the request-category classifier.
- **Turn metrics:** turns used, answer yield, unavailable-turn rate, redundant-category rate, and off-protocol rate.

FVCU (factuality, validity, coherence, utility) is an auxiliary LLM-judged evaluation. Included aggregate FVCU values come from saved judgments; they are not recomputed by the offline deterministic pipeline.

## 7. Reproducing Table 3

The exact reported aggregate values are preserved in:

- `results/canonical_matched225/table3_recomputed_canonical225.csv`
- `results/canonical_matched225/table3_recomputed_canonical225_rounded.csv`

To recompute deterministic metrics, create a JSONL manifest following `examples/synthetic/manifest.jsonl`; each row identifies `model`, `specialty`, `case_id`, `case_path`, `episode_path`, and whether the case belongs to the matched subset. Then run `evaluation/score_manifest.py`. The scorer rejects duplicate specialty-case-model identities and inconsistent cross-model event-bank snapshots.

The real manifest, event banks, and episodes are not duplicated in this code repository. At the audited public-data commit, the camera-ready addendum needed for end-to-end raw reproduction was not yet present; see the data-repository audit for the exact missing artifacts.

## 8. Reproducing Figure 2

The corrected aggregate category-recall data are releasable because they contain no article text or case identifiers:

```bash
python evaluation/plot_category_recall_heatmap.py \
  --input results/canonical_matched225/category_recall_corrected_canonical225.csv \
  --output-dir /tmp/diagtrace-figure2
```

The command creates PNG and PDF versions of the heatmap.

To derive the corrected category-recall source from raw canonical artifacts first, use
`category_recall_matched.csv`, which is emitted by `evaluation/score_manifest.py` for the matched subset.

## 9. Fixed-checklist baseline

`baselines/run_real_checklist_baseline.py` runs the untuned, diagnosis-independent checklist through the unchanged report agent under the 15-turn cap. `score_checklist_vs_models.py` computes paired event-recall comparisons and `bootstrap_checklist_ci.py` computes uncertainty intervals. Aggregate corrected results for the canonical 16 specialty-stratified cases are in `results/fixed_checklist/`.

Regenerating the baseline episodes requires the frozen real event banks and report-agent access; those inputs are excluded. The included CSVs are aggregate outputs, not a substitute synthetic rerun.

## 10. Full-scale sensitivity analysis

`analysis/internal_full_scale_sensitivity.py` is the exact corrected camera-ready analysis used with saved artifacts. It uses canonical `(specialty, PMCID)` identities, checks event-bank signatures across models, and compares matched-225 with all completed episodes. Aggregate outputs are in `results/full_scale_sensitivity/`. The raw episodes required for rerunning it are excluded pending redistribution review.

## 11. Data access and redistribution

The DiagTrace benchmark data have been approved for public redistribution and are maintained in the separate [data repository](https://github.com/Fanncg1229/Diagtrace). This code repository contains no raw article files and does not duplicate the benchmark payload. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md) and [docs/DATA_REPOSITORY_AUDIT.md](docs/DATA_REPOSITORY_AUDIT.md).

## 12. Citation

The final ACL Anthology identifier and DOI were not available when this package was prepared.

```bibtex
@inproceedings{wang2027diagtrace,
  title     = {DiagTrace: An Interactive Benchmark for Clinical Decision Making},
  author    = {Wang, Yufan and Yang, Rui and Lin, Yi and Weng, Chunhua and Peng, Yifan},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year      = {2027},
  note      = {Accepted; final anthology identifier pending}
}
```

## Release status

The code is distributed under the MIT License; see `LICENSE`. Complete the remaining data-addendum and citation checks in `PUBLIC_RELEASE_CHECKLIST.md` before the coordinated public release.
