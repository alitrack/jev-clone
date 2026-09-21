# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/en-evidence-v3.jsonl`
- items sha256: `3c24d86101c509f5ae3d397087ad0d084b1d37e552b5e792efc025efafe4fa20`
- prompt-set sha256: `91c904e11dc601c0eaf99e73023a13aeebad81cad08d33c43c9bb01684e73828`
- predictions: `eval/runs/20260921T1110Z-local-m4-seq-en-predictions.jsonl`
- predictions sha256: `4b62d1ddd5fc00909365e327482a9790bd88d7295cd861490633ab2245fe13e6`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-en-v2-mt` (n = 134)

| overall | 134 | 134 | 134 | 0.619403 | 0.761151 | 0.713799 | 0.469431 | 0.223015 | 0.093888 |
| candidate_selection | 37 | 37 | 37 | 0.837838 | 0.886848 | 0.584452 | 0.345151 | 0.196865 | 0.175243 |
| evidence_judgment | 33 | 33 | 33 | 0.636364 | 0.744898 | 0.642812 | 0.428388 | 0.214194 | 0.138941 |
| missing_evidence | 31 | 31 | 31 | 0.580645 | 0.674603 | 0.678088 | 0.487450 | 0.245307 | 0.158846 |
| rule_application | 33 | 33 | 33 | 0.393939 | 0.510256 | 0.963358 | 0.632892 | 0.256408 | 0.239528 |

### source `authored-en-v3-mt` (n = 33)

| overall | 33 | 33 | 33 | 0.575758 | 0.687302 | 0.796108 | 0.506811 | 0.205696 | 0.109954 |
| rule_application | 33 | 33 | 33 | 0.575758 | 0.687302 | 0.796108 | 0.506811 | 0.205696 | 0.109954 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 37 | 37 | 37 | 0.837838 | 0.886848 | 0.584452 | 0.345151 | 0.196865 | 0.175243 |
| evidence_judgment | 33 | 33 | 33 | 0.636364 | 0.744898 | 0.642812 | 0.428388 | 0.214194 | 0.138941 |
| missing_evidence | 31 | 31 | 31 | 0.580645 | 0.674603 | 0.678088 | 0.487450 | 0.245307 | 0.158846 |
| rule_application | 66 | 66 | 66 | 0.484848 | 0.600463 | 0.879733 | 0.569852 | 0.233225 | 0.151531 |

## Reliability curve (per stratum)

### `authored-en-v2-mt` / `candidate_selection` (n_scored = 37)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.40, 0.50) | 5 | 0.4656 | 0.8000 | 0.3344 |
| [0.50, 0.60) | 8 | 0.5464 | 0.7500 | 0.2036 |
| [0.60, 0.70) | 7 | 0.6388 | 0.8571 | 0.2184 |
| [0.70, 0.80) | 10 | 0.7506 | 0.9000 | 0.1494 |
| [0.80, 0.90) | 7 | 0.8342 | 0.8571 | 0.0229 |

### `authored-en-v2-mt` / `evidence_judgment` (n_scored = 33)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 10 | 0.5273 | 0.6000 | 0.0727 |
| [0.60, 0.70) | 6 | 0.6309 | 0.1667 | 0.4642 |
| [0.70, 0.80) | 3 | 0.7132 | 0.6667 | 0.0465 |
| [0.80, 0.90) | 5 | 0.8523 | 0.8000 | 0.0523 |
| [0.90, 1.00) | 9 | 0.9635 | 0.8889 | 0.0746 |

### `authored-en-v2-mt` / `missing_evidence` (n_scored = 31)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 6 | 0.5539 | 0.5000 | 0.0539 |
| [0.60, 0.70) | 6 | 0.6418 | 0.5000 | 0.1418 |
| [0.70, 0.80) | 10 | 0.7575 | 0.4000 | 0.3575 |
| [0.80, 0.90) | 6 | 0.8497 | 0.8333 | 0.0163 |
| [0.90, 1.00) | 3 | 0.9743 | 1.0000 | 0.0257 |

### `authored-en-v2-mt` / `rule_application` (n_scored = 33)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 4 | 0.3780 | 0.0000 | 0.3780 |
| [0.40, 0.50) | 4 | 0.4320 | 0.5000 | 0.0680 |
| [0.50, 0.60) | 8 | 0.5442 | 0.2500 | 0.2942 |
| [0.60, 0.70) | 8 | 0.6425 | 0.3750 | 0.2675 |
| [0.70, 0.80) | 4 | 0.7437 | 0.5000 | 0.2437 |
| [0.80, 0.90) | 3 | 0.8701 | 0.6667 | 0.2034 |
| [0.90, 1.00) | 2 | 0.9789 | 1.0000 | 0.0211 |

### `authored-en-v3-mt` / `rule_application` (n_scored = 33)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 6 | 0.3701 | 0.3333 | 0.0368 |
| [0.40, 0.50) | 9 | 0.4558 | 0.5556 | 0.0998 |
| [0.50, 0.60) | 5 | 0.5623 | 0.4000 | 0.1623 |
| [0.60, 0.70) | 5 | 0.6468 | 0.8000 | 0.1532 |
| [0.70, 0.80) | 3 | 0.7607 | 0.6667 | 0.0940 |
| [0.80, 0.90) | 3 | 0.8269 | 0.6667 | 0.1603 |
| [0.90, 1.00) | 2 | 0.9154 | 1.0000 | 0.0846 |
