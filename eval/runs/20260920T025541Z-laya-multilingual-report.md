# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v1.jsonl`
- items sha256: `2f0415d903ad588cd392d1f48ec940a17f0e3b2b300885a42dd5e99a3af10b4b`
- prompt-set sha256: `91c3d2e1efa5668ac194efb54f7c88c126a0b7747b3c5e25b7e6a3faec14329b`
- predictions: `eval/runs/20260920T025541Z-laya-multilingual-predictions.jsonl`
- predictions sha256: `55419cc4482440038f417582b15a0ee8dbc49dc089380884347032ccfb835d79`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.546667 | 0.583725 | 1.093239 | 0.656204 | 0.337826 | 0.221233 |
| candidate_selection | 38 | 38 | 38 | 0.552632 | 0.615385 | 1.496039 | 0.775898 | 0.429205 | 0.354302 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.513333 | 0.927704 | 0.637743 | 0.318872 | 0.271263 |
| missing_evidence | 36 | 36 | 36 | 0.638889 | 0.659091 | 0.919537 | 0.537922 | 0.286717 | 0.187614 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.484524 | 1.020534 | 0.667028 | 0.287084 | 0.222541 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.552632 | 0.615385 | 1.496039 | 0.775898 | 0.429205 | 0.354302 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.513333 | 0.927704 | 0.637743 | 0.318872 | 0.271263 |
| missing_evidence | 36 | 36 | 36 | 0.638889 | 0.659091 | 0.919537 | 0.537922 | 0.286717 | 0.187614 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.484524 | 1.020534 | 0.667028 | 0.287084 | 0.222541 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 1 | 0.3985 | 1.0000 | 0.6015 |
| [0.40, 0.50) | 7 | 0.4431 | 0.7143 | 0.2711 |
| [0.50, 0.60) | 4 | 0.5717 | 0.5000 | 0.0717 |
| [0.60, 0.70) | 1 | 0.6583 | 1.0000 | 0.3417 |
| [0.70, 0.80) | 5 | 0.7480 | 0.4000 | 0.3480 |
| [0.80, 0.90) | 6 | 0.8469 | 0.5000 | 0.3469 |
| [0.90, 1.00) | 14 | 0.9653 | 0.5000 | 0.4653 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 10 | 0.5456 | 0.7000 | 0.1543 |
| [0.60, 0.70) | 6 | 0.6534 | 0.3333 | 0.3201 |
| [0.70, 0.80) | 6 | 0.7693 | 0.5000 | 0.2693 |
| [0.80, 0.90) | 9 | 0.8600 | 0.6667 | 0.1934 |
| [0.90, 1.00) | 7 | 0.9268 | 0.4286 | 0.4983 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.40, 0.50) | 3 | 0.4836 | 0.6667 | 0.1831 |
| [0.50, 0.60) | 2 | 0.5194 | 0.5000 | 0.0194 |
| [0.60, 0.70) | 6 | 0.6353 | 0.6667 | 0.0313 |
| [0.70, 0.80) | 5 | 0.7464 | 0.4000 | 0.3464 |
| [0.80, 0.90) | 10 | 0.8494 | 0.7000 | 0.1494 |
| [0.90, 1.00) | 10 | 0.9752 | 0.7000 | 0.2752 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 1 | 0.3922 | 0.0000 | 0.3922 |
| [0.40, 0.50) | 3 | 0.4777 | 0.3333 | 0.1444 |
| [0.50, 0.60) | 8 | 0.5352 | 0.3750 | 0.1602 |
| [0.60, 0.70) | 8 | 0.6457 | 0.6250 | 0.0207 |
| [0.70, 0.80) | 10 | 0.7351 | 0.4000 | 0.3351 |
| [0.80, 0.90) | 6 | 0.8351 | 0.5000 | 0.3351 |
| [0.90, 1.00) | 2 | 0.9112 | 0.5000 | 0.4112 |
