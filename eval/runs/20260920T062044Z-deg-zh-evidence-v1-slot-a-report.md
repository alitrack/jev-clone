# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v1.jsonl`
- items sha256: `2f0415d903ad588cd392d1f48ec940a17f0e3b2b300885a42dd5e99a3af10b4b`
- prompt-set sha256: `91c3d2e1efa5668ac194efb54f7c88c126a0b7747b3c5e25b7e6a3faec14329b`
- predictions: `eval/runs/20260920T062044Z-deg-zh-evidence-v1-slot-a-predictions.jsonl`
- predictions sha256: `16af8c9930313ab2594a97d3092ed4dbb7503a129b0c0a5d0c3eacf409baeead`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.426667 | 0.454545 | 15.841785 | 1.146667 | 0.546154 | 0.573333 |
| candidate_selection | 38 | 38 | 38 | 0.289474 | 0.346154 | 19.632568 | 1.421053 | 0.710526 | 0.710526 |
| evidence_judgment | 38 | 38 | 38 | 0.500000 | 0.500000 | 13.815511 | 1.000000 | 0.500000 | 0.500000 |
| missing_evidence | 36 | 36 | 36 | 0.583333 | 0.727273 | 11.512925 | 0.833333 | 0.393939 | 0.416667 |
| rule_application | 38 | 38 | 38 | 0.342105 | 0.428571 | 18.178303 | 1.315789 | 0.571429 | 0.657895 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.289474 | 0.346154 | 19.632568 | 1.421053 | 0.710526 | 0.710526 |
| evidence_judgment | 38 | 38 | 38 | 0.500000 | 0.500000 | 13.815511 | 1.000000 | 0.500000 | 0.500000 |
| missing_evidence | 36 | 36 | 36 | 0.583333 | 0.727273 | 11.512925 | 0.833333 | 0.393939 | 0.416667 |
| rule_application | 38 | 38 | 38 | 0.342105 | 0.428571 | 18.178303 | 1.315789 | 0.571429 | 0.657895 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.2895 | 0.7105 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.5000 | 0.5000 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 36 | 1.0000 | 0.5833 | 0.4167 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.3421 | 0.6579 |
