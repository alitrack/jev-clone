# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v1.jsonl`
- items sha256: `2f0415d903ad588cd392d1f48ec940a17f0e3b2b300885a42dd5e99a3af10b4b`
- prompt-set sha256: `91c3d2e1efa5668ac194efb54f7c88c126a0b7747b3c5e25b7e6a3faec14329b`
- predictions: `eval/runs/20260920T062044Z-deg-zh-evidence-v1-slot-prior-predictions.jsonl`
- predictions sha256: `96c2f5c89ff585eb0b9c457a9462f14d907e7d61cea98acd66e459bda16e95af`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.480000 | 0.440000 | 0.825907 | 0.549739 | 0.303178 | 0.000000 |
| candidate_selection | 38 | 38 | 38 | 0.394737 | 0.384615 | 1.027942 | 0.636111 | 0.410404 | 0.016630 |
| evidence_judgment | 38 | 38 | 38 | 0.500000 | 0.500000 | 0.693147 | 0.500000 | 0.250000 | 0.000000 |
| missing_evidence | 36 | 36 | 36 | 0.555556 | 0.545455 | 0.756143 | 0.520831 | 0.274783 | 0.070917 |
| rule_application | 38 | 38 | 38 | 0.473684 | 0.442857 | 0.822724 | 0.540494 | 0.250000 | 0.050554 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.394737 | 0.384615 | 1.027942 | 0.636111 | 0.410404 | 0.016630 |
| evidence_judgment | 38 | 38 | 38 | 0.500000 | 0.500000 | 0.693147 | 0.500000 | 0.250000 | 0.000000 |
| missing_evidence | 36 | 36 | 36 | 0.555556 | 0.545455 | 0.756143 | 0.520831 | 0.274783 | 0.070917 |
| rule_application | 38 | 38 | 38 | 0.473684 | 0.442857 | 0.822724 | 0.540494 | 0.250000 | 0.050554 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 32 | 0.3611 | 0.3750 | 0.0139 |
| [0.50, 0.60) | 6 | 0.5312 | 0.5000 | 0.0312 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 38 | 0.5000 | 0.5000 | 0.0000 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 4 | 0.3611 | 0.2500 | 0.1111 |
| [0.50, 0.60) | 32 | 0.5279 | 0.5938 | 0.0659 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 37 | 0.5114 | 0.4595 | 0.0519 |
| [0.90, 1.00) | 1 | 1.0000 | 1.0000 | 0.0000 |
