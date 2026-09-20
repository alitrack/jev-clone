# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/en-evidence-v1.jsonl`
- items sha256: `c8a59e511d4520351c3b04b8868504b30b8bb9deeaa196de7b6b17f733ee1c6a`
- prompt-set sha256: `8aa53b61f09f6fb0bcd5450d76a531c373090511696a14788a77abff36097b5c`
- predictions: `eval/runs/20260920T062044Z-deg-en-evidence-v1-slot-a-predictions.jsonl`
- predictions sha256: `5b89918e39605fe954c2f0c1dc19dc54c708342eb300de035edaf94ad1ba89f6`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-en-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.426667 | 0.452830 | 15.841785 | 1.146667 | 0.546154 | 0.573333 |
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

### `authored-en-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.2895 | 0.7105 |

### `authored-en-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.5000 | 0.5000 |

### `authored-en-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 36 | 1.0000 | 0.5833 | 0.4167 |

### `authored-en-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.3421 | 0.6579 |
