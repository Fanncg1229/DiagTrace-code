# Data access and redistribution

## What is excluded

This code repository excludes:

- PMC XML, article PDFs, figures, captions copied from articles, and source quotations;
- real extracted event banks and hidden reference resources;
- real clinician/report-agent conversations and per-case trajectories;
- credentials, provider responses containing account metadata, and institutional files.

The benchmark data have been approved for public redistribution and are hosted separately at <https://github.com/Fanncg1229/Diagtrace>. The synthetic example in this code repository contains invented clinical text and IDs and must not be described as benchmark data.

## Why public access is not enough

PMC is a repository containing works under different copyright and license terms. Free availability in PMC does not mean that every article can be redistributed in a derived benchmark. Reuse permission must be evaluated per article and per included content type.

The release owner has confirmed redistribution approval. Future additions should retain the underlying approval and provenance records, including PMCID, article URL, license/permission basis, retrieval date, included content types, and reviewer decision.

## Lawful reconstruction workflow

1. Obtain PMC content through NCBI-approved APIs or the PMC Open Access mechanisms; do not scrape the website.
2. Verify the article-specific license and terms before storing or redistributing text or figures.
3. Run the provided construction code locally on eligible material.
4. Keep a frozen content hash and provenance record for every generated event bank.
5. Do not distribute generated excerpts when the underlying license does not permit it.

Relevant official guidance:

- PMC Open Access Subset: <https://pmc.ncbi.nlm.nih.gov/tools/openftlist/>
- PMC Copyright Notice: <https://pmc.ncbi.nlm.nih.gov/about/copyright/>
- PMC OAI service: <https://pmc.ncbi.nlm.nih.gov/tools/oai/>

This document is a release-safety procedure, not legal advice.
