# Public release checklist

The package is suitable for private collaborator review. Complete every unchecked item before public release.

- [x] Confirm and add the MIT software license.
- [ ] Confirm author names, order, paper title, venue wording, and final citation metadata.
- [ ] Replace the pending citation note only after the ACL Anthology identifier/DOI is official.
- [x] Confirm benchmark-data approval for public redistribution.
- [ ] Retain the approval/provenance ledger with the public data release records.
- [ ] Decide whether model conversations may be distributed under provider terms and remove provider metadata.
- [ ] Add the reviewed camera-ready data addendum without overwriting the original public release, then rerun all deterministic checks from a clean clone.
- [ ] Verify that every model is scored against the same frozen reference snapshot for each canonical `(specialty, PMCID)` identity.
- [ ] Resolve/document any remaining mismatch between construction-stage events and Setting 2 evidence items.
- [ ] Pin final dependency versions or publish a tested lock file/container.
- [ ] Run secret scanning and large-file scanning on the final Git history, not only the working tree.
- [ ] Review all prompts and examples for accidental article quotations or personal/institutional information.
- [ ] Add repository governance/contact information and archival release metadata.
- [ ] Re-run `python -m pytest -q` and the README commands in a clean environment.
- [ ] Review `RELEASE_VALIDATION_REPORT.md` and resolve all skipped checks that are required for the intended release claim.

Do not describe the current package as fully end-to-end reproducible from public raw data while the real event banks and trajectories remain excluded.
