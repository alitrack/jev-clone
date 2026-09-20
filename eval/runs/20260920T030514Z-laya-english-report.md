# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v1.jsonl`
- items sha256: `2f0415d903ad588cd392d1f48ec940a17f0e3b2b300885a42dd5e99a3af10b4b`
- prompt-set sha256: `91c3d2e1efa5668ac194efb54f7c88c126a0b7747b3c5e25b7e6a3faec14329b`
- predictions: `eval/runs/20260920T030514Z-laya-english-predictions.jsonl`
- predictions sha256: `b1b0785ca811045172ccbbc9a60272a80e737113f998437f3fd16e7ae35ed416`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.453333 | 0.553929 | 0.978837 | 0.637972 | 0.332573 | 0.150255 |
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.475641 | 1.121274 | 0.682099 | 0.427458 | 0.120151 |
| evidence_judgment | 38 | 38 | 38 | 0.473684 | 0.493333 | 0.872012 | 0.608740 | 0.304370 | 0.200108 |
| missing_evidence | 36 | 36 | 36 | 0.500000 | 0.626623 | 0.941564 | 0.632484 | 0.317427 | 0.229636 |
| rule_application | 38 | 38 | 38 | 0.421053 | 0.573214 | 0.978536 | 0.628276 | 0.235712 | 0.177872 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.475641 | 1.121274 | 0.682099 | 0.427458 | 0.120151 |
| evidence_judgment | 38 | 38 | 38 | 0.473684 | 0.493333 | 0.872012 | 0.608740 | 0.304370 | 0.200108 |
| missing_evidence | 36 | 36 | 36 | 0.500000 | 0.626623 | 0.941564 | 0.632484 | 0.317427 | 0.229636 |
| rule_application | 38 | 38 | 38 | 0.421053 | 0.573214 | 0.978536 | 0.628276 | 0.235712 | 0.177872 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 17 | 0.3744 | 0.3529 | 0.0214 |
| [0.40, 0.50) | 14 | 0.4376 | 0.5000 | 0.0624 |
| [0.50, 0.60) | 1 | 0.5876 | 1.0000 | 0.4124 |
| [0.60, 0.70) | 1 | 0.6433 | 0.0000 | 0.6433 |
| [0.70, 0.80) | 3 | 0.7718 | 0.3333 | 0.4385 |
| [0.90, 1.00) | 2 | 0.9783 | 0.5000 | 0.4783 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 11 | 0.5484 | 0.3636 | 0.1847 |
| [0.60, 0.70) | 16 | 0.6435 | 0.5625 | 0.0810 |
| [0.70, 0.80) | 4 | 0.7605 | 0.5000 | 0.2605 |
| [0.80, 0.90) | 4 | 0.8414 | 0.5000 | 0.3414 |
| [0.90, 1.00) | 3 | 0.9560 | 0.3333 | 0.6227 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 2 | 0.3873 | 0.5000 | 0.1127 |
| [0.40, 0.50) | 3 | 0.4310 | 0.3333 | 0.0976 |
| [0.50, 0.60) | 7 | 0.5631 | 0.5714 | 0.0083 |
| [0.60, 0.70) | 5 | 0.6701 | 0.4000 | 0.2701 |
| [0.70, 0.80) | 6 | 0.7542 | 0.3333 | 0.4209 |
| [0.80, 0.90) | 6 | 0.8636 | 0.6667 | 0.1970 |
| [0.90, 1.00) | 7 | 0.9475 | 0.5714 | 0.3761 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 7 | 0.3806 | 0.2857 | 0.0949 |
| [0.40, 0.50) | 6 | 0.4285 | 0.3333 | 0.0952 |
| [0.50, 0.60) | 13 | 0.5309 | 0.3077 | 0.2232 |
| [0.60, 0.70) | 5 | 0.6462 | 0.8000 | 0.1538 |
| [0.70, 0.80) | 3 | 0.7223 | 0.6667 | 0.0556 |
| [0.80, 0.90) | 1 | 0.8676 | 0.0000 | 0.8676 |
| [0.90, 1.00) | 3 | 0.9394 | 0.6667 | 0.2727 |
