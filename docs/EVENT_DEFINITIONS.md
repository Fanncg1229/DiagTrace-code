# Event definitions and denominators

Two related but non-identical resource layers exist in the project.

## Construction-stage source events

The corpus summary in Table 1 counts `narrowing_data_construction.evidence_events`: 5,911 events across 510 cases, or 11.59 per case. These objects describe source-derived findings used during benchmark construction.

## Setting 2 report-agent items

Interactive evaluation reads the separate `evidence_items` list in the frozen Setting 2 case resource. `subset_characteristics.csv` reports `reference_event_count` from this bank. For the matched 225 cases, the count is 1,790 items, or 7.96 per case. In the available full-scale completed set, the corresponding average is about 8.19.

Event-level recall applies a further protocol filter: only items with `phase_index > 0` are eligible because phase 0 is already visible in the initial presentation. In the matched set this leaves 1,439 eligible items across 225 cases (6.40 per case); one case has no eligible item, so the reported macro event recall is over 224 defined cases.

Therefore the approximately eight-item `reference_event_count` and the 11.59-event Table 1 count are not contradictory arithmetic. They refer to different schemas and evaluation stages. They must retain distinct labels in the manuscript and release documentation. No result should be changed merely to force the denominators to match.
