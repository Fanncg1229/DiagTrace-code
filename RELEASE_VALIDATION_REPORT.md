# Release validation report

Validation date: 2026-11-13  
Scope: local pre-publication package only; no repository was published or pushed.

## Passed checks

- **Isolated installation:** created a new virtual environment and installed the package editable with `python -m pip install --no-deps -e .`.
- **Automated tests:** `python -m pytest -q` completed with **18 passed, 0 failed**.
- **Python syntax/import compilation:** `python -m compileall` completed for `src/`, `construction/`, `evaluation/`, `baselines/`, `analysis/`, and `tests/`.
- **Offline synthetic scoring:** `evaluation/score_manifest.py` produced `metrics.csv` and `diagnostics.json` without network or API access.
- **Figure reproduction:** the documented Figure 2 command produced PNG and PDF files from the included corrected aggregate CSV.
- **Classifier regression:** tests cover the corrected complete word forms for pathology, histology, cytology, morphology, sonography, elastography, echocardiography, angiography, endoscopy, laboratory/serology, and immunosuppressive treatment.
- **Result regression:** included full-precision Table 3 values round to the seven required camera-ready metric rows; event recall is labeled as a macro average over 224 defined cases.
- **Snapshot protection:** the generic scorer validates canonical specialty-case identity and rejects cross-model event-bank signature mismatches.
- **Credential/path scan:** no populated API key, common secret-key prefix, or home-directory-specific absolute path was found in the packaged text files. `.env.example` contains blank placeholders only.
- **Payload scan:** no large file over 5 MB, symlink, real PMC article file, real case event bank, or real interaction trace is included.
- **Archive smoke test:** the final ZIP was extracted to a temporary directory and its tests and synthetic quick start were rerun successfully.

The offline smoke environment reused the host's installed scientific packages to avoid dependency downloads. `pip check` consequently reported a pre-existing `forestplot`/`matplotlib-inline` constraint conflict in the host environment. DiagTrace imports, tests, scorer, and plotting command were unaffected; a fully isolated dependency-resolution test remains advisable before publication.

## Corrected during package validation

The first smoke test found two release-interface issues: the synthetic test called a keyword-only function positionally, and the example manifest repeated its artifact-root prefix. Both were corrected before final validation. These fixes affect only the new release harness, not the original project or experiment outputs.

## Not performed / skipped

- **End-to-end Table 3 recomputation from real cases:** skipped because the real event banks and episodes are intentionally excluded pending redistribution review.
- **Fixed-checklist episode regeneration:** skipped because it would require real event banks and an external report-agent API.
- **Reference-resource extraction/audit and FVCU regeneration:** skipped because these require external LLM APIs and would create new, potentially version-dependent outputs.
- **Full-scale sensitivity recomputation from raw episodes:** skipped because the saved real traces are excluded. Included aggregate outputs and exact analysis provenance code were inspected instead.
- **Data payload:** intentionally absent from this code package. Redistribution approval was subsequently confirmed and the public data repository was audited separately; the camera-ready addendum was not yet present at the audited public commit.

## Result reproduction status

- The corrected canonical aggregates, fixed-checklist aggregates, and full-scale sensitivity aggregates are preserved unchanged.
- The offline code path and regression values were validated.
- Figure 2 is reproducible from the included aggregate CSV.
- The reported real-case metrics were **not independently recomputed end to end inside this redacted package**, because the necessary case-level artifacts are not presently releasable.

## Release assessment

- **Private GitHub review:** safe, provided the repository remains private and access controlled.
- **Public code release:** the repository now carries the MIT License. The reviewed camera-ready data addendum still needs to be added to the data repository before claiming end-to-end raw reproduction.
