# Public release checklist

The package is suitable for private collaborator review. Complete every unchecked item before public release.

- [ ] Confirm the software license with all code contributors and add the approved `LICENSE`.
- [ ] Confirm author names, order, paper title, venue wording, and final citation metadata.
- [ ] Replace the pending citation note only after the ACL Anthology identifier/DOI is official.
- [ ] Complete an article-level PMC license ledger for every real case proposed for release.
- [ ] Obtain institutional/legal review for redistributed excerpts, derived event banks, and traces.
- [ ] Decide whether model conversations may be distributed under provider terms and remove provider metadata.
- [ ] If real artifacts are approved, regenerate a public manifest and rerun all deterministic checks from a clean clone.
- [ ] Verify that every model is scored against the same frozen reference snapshot for each canonical `(specialty, PMCID)` identity.
- [ ] Resolve/document any remaining mismatch between construction-stage events and Setting 2 evidence items.
- [ ] Pin final dependency versions or publish a tested lock file/container.
- [ ] Run secret scanning and large-file scanning on the final Git history, not only the working tree.
- [ ] Review all prompts and examples for accidental article quotations or personal/institutional information.
- [ ] Add repository governance/contact information and archival release metadata.
- [ ] Re-run `python -m pytest -q` and the README commands in a clean environment.
- [ ] Review `RELEASE_VALIDATION_REPORT.md` and resolve all skipped checks that are required for the intended release claim.

Do not describe the current package as fully end-to-end reproducible from public raw data while the real event banks and trajectories remain excluded.
