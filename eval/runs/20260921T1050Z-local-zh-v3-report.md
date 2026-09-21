# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v3.jsonl`
- items sha256: `a2e938837bebb2297d2168c3e366d0e9c20d535155656f79325dc846dd425ff7`
- prompt-set sha256: `4570212d8946ea1cf50fb08d79ecaf789d8896aed576bf2c339719e1d8262d2b`
- predictions: `eval/runs/20260921T1050Z-local-zh-v3-predictions.jsonl`
- predictions sha256: `143bd869510307711d1ad1d47707427341203dc278f0bfd8c5d3e65241d4d73f`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.733333 | 0.863792 | 0.573987 | 0.371109 | 0.165137 | 0.081705 |
| candidate_selection | 38 | 38 | 38 | 0.894737 | 0.955128 | 0.364301 | 0.199282 | 0.109745 | 0.186670 |
| evidence_judgment | 38 | 38 | 38 | 0.684211 | 0.733333 | 0.605973 | 0.427252 | 0.213626 | 0.185816 |
| missing_evidence | 36 | 36 | 36 | 0.777778 | 0.772727 | 0.530155 | 0.350536 | 0.172762 | 0.161032 |
| rule_application | 38 | 38 | 38 | 0.578947 | 0.742857 | 0.793211 | 0.506283 | 0.165649 | 0.177802 |

### source `authored-zh-v3` (n = 45)

| overall | 45 | 45 | 45 | 0.666667 | 0.776557 | 0.646085 | 0.403192 | 0.160259 | 0.088015 |
| rule_application | 45 | 45 | 45 | 0.666667 | 0.776557 | 0.646085 | 0.403192 | 0.160259 | 0.088015 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.894737 | 0.955128 | 0.364301 | 0.199282 | 0.109745 | 0.186670 |
| evidence_judgment | 38 | 38 | 38 | 0.684211 | 0.733333 | 0.605973 | 0.427252 | 0.213626 | 0.185816 |
| missing_evidence | 36 | 36 | 36 | 0.777778 | 0.772727 | 0.530155 | 0.350536 | 0.172762 | 0.161032 |
| rule_application | 83 | 83 | 83 | 0.626506 | 0.765106 | 0.713444 | 0.450390 | 0.162719 | 0.116628 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.40, 0.50) | 3 | 0.4692 | 1.0000 | 0.5308 |
| [0.50, 0.60) | 5 | 0.5598 | 0.4000 | 0.1598 |
| [0.60, 0.70) | 6 | 0.6529 | 1.0000 | 0.3471 |
| [0.70, 0.80) | 5 | 0.7637 | 0.8000 | 0.0363 |
| [0.80, 0.90) | 12 | 0.8457 | 1.0000 | 0.1543 |
| [0.90, 1.00) | 7 | 0.9162 | 1.0000 | 0.0838 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.60) | 8 | 0.5406 | 0.7500 | 0.2094 |
| [0.60, 0.70) | 5 | 0.6390 | 0.4000 | 0.2390 |
| [0.70, 0.80) | 7 | 0.7633 | 0.2857 | 0.4776 |
| [0.80, 0.90) | 6 | 0.8689 | 0.8333 | 0.0355 |
| [0.90, 1.00) | 12 | 0.9695 | 0.9167 | 0.0529 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.40, 0.50) | 1 | 0.4676 | 0.0000 | 0.4676 |
| [0.50, 0.60) | 5 | 0.5530 | 0.8000 | 0.2470 |
| [0.60, 0.70) | 9 | 0.6480 | 0.8889 | 0.2409 |
| [0.70, 0.80) | 9 | 0.7538 | 0.5556 | 0.1982 |
| [0.80, 0.90) | 6 | 0.8359 | 0.8333 | 0.0026 |
| [0.90, 1.00) | 6 | 0.9789 | 1.0000 | 0.0211 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 4 | 0.3859 | 0.7500 | 0.3641 |
| [0.40, 0.50) | 7 | 0.4566 | 0.1429 | 0.3138 |
| [0.50, 0.60) | 7 | 0.5487 | 0.5714 | 0.0228 |
| [0.60, 0.70) | 8 | 0.6681 | 0.3750 | 0.2931 |
| [0.70, 0.80) | 1 | 0.7498 | 1.0000 | 0.2502 |
| [0.80, 0.90) | 8 | 0.8424 | 0.8750 | 0.0326 |
| [0.90, 1.00) | 3 | 0.9706 | 1.0000 | 0.0294 |

### `authored-zh-v3` / `rule_application` (n_scored = 45)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.30, 0.40) | 3 | 0.3756 | 0.3333 | 0.0423 |
| [0.40, 0.50) | 13 | 0.4515 | 0.4615 | 0.0100 |
| [0.50, 0.60) | 8 | 0.5627 | 0.7500 | 0.1873 |
| [0.60, 0.70) | 7 | 0.6307 | 0.5714 | 0.0593 |
| [0.70, 0.80) | 5 | 0.7495 | 1.0000 | 0.2505 |
| [0.80, 0.90) | 4 | 0.8467 | 0.7500 | 0.0967 |
| [0.90, 1.00) | 5 | 0.9699 | 1.0000 | 0.0301 |
