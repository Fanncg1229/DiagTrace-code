# Data repository audit

The public data repository is <https://github.com/Fanncg1229/Diagtrace>.

At audited commit `13e23ca48b775993cbbb103f3db0917c14582ebf`, it contains the 510 construction resources and four matched sets of 225 traces, but it does not yet contain the canonical Setting 2 event banks, canonical replacements for six Claude traces, full-scale episodes, fixed-checklist episodes, or saved FVCU outputs required for end-to-end camera-ready reproduction.

The existing `data/diagtrace_510.jsonl` contains 5,911 construction-stage `reference_resource.evidence_events`. These are not the report-agent `evidence_items` used by Setting 2. Do not pass the construction JSONL to `evaluation/score_manifest.py` as a substitute for the frozen Setting 2 case banks.

The six canonical identity overrides are:

| PMCID | Canonical specialty |
|---|---|
| PMC10008168 | radiology |
| PMC10254956 | neurology |
| PMC10441080 | oncology |
| PMC11391149 | oncology |
| PMC12255965 | nephrology |
| PMC4938284 | nephrology |

Until the reviewed camera-ready data addendum is added, the public repository supports inspection of the original resources and traces but not raw end-to-end reproduction of the corrected camera-ready metrics.
