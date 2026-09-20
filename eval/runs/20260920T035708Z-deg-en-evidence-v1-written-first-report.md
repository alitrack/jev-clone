# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/en-evidence-v1.jsonl`
- items sha256: `c8a59e511d4520351c3b04b8868504b30b8bb9deeaa196de7b6b17f733ee1c6a`
- prompt-set sha256: `8aa53b61f09f6fb0bcd5450d76a531c373090511696a14788a77abff36097b5c`
- predictions: `eval/runs/deg-en-evidence-v1-written-first-predictions.jsonl`
- predictions sha256: `19067c543288977ee6cef86a0ed678e1f02cfb36df505dbb554f75a61a5fcd3c`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-en-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.480000 | 0.528302 | 14.368131 | 1.040000 | 0.484615 | 0.520000 |
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.384615 | 15.996907 | 1.157895 | 0.578947 | 0.578947 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.626042 | 12.361246 | 0.894737 | 0.447368 | 0.447368 |
| missing_evidence | 36 | 36 | 36 | 0.500000 | 0.636364 | 13.815511 | 1.000000 | 0.484848 | 0.500000 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.642857 | 15.269775 | 1.105263 | 0.380952 | 0.552632 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.384615 | 15.996907 | 1.157895 | 0.578947 | 0.578947 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.626042 | 12.361246 | 0.894737 | 0.447368 | 0.447368 |
| missing_evidence | 36 | 36 | 36 | 0.500000 | 0.636364 | 13.815511 | 1.000000 | 0.484848 | 0.500000 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.642857 | 15.269775 | 1.105263 | 0.380952 | 0.552632 |

## Reliability curve (per stratum)

### `authored-en-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.4211 | 0.5789 |

### `authored-en-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.5526 | 0.4474 |

### `authored-en-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 36 | 1.0000 | 0.5000 | 0.5000 |

### `authored-en-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.4474 | 0.5526 |
