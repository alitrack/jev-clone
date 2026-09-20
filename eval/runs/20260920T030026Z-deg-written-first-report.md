# jev-eval report (`jev-eval-report/1`)

- items: `eval/items/zh-evidence-v1.jsonl`
- items sha256: `2f0415d903ad588cd392d1f48ec940a17f0e3b2b300885a42dd5e99a3af10b4b`
- prompt-set sha256: `91c3d2e1efa5668ac194efb54f7c88c126a0b7747b3c5e25b7e6a3faec14329b`
- predictions: `eval/runs/deg-written-first-predictions.jsonl`
- predictions sha256: `6cc0b395115944f6898a8d71eabd8740bb852b1c61c47a57388bcb0df68eba2c`
- confidence bins: 10
- NLL probability floor: 0.000000000001
- verify tolerance: 0.000000000001

Stratified by source and by category. Every metric is reported per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not collapsed into one score.

## By source

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|

### source `authored-zh-v1` (n = 150)

| overall | 150 | 150 | 150 | 0.473333 | 0.528095 | 14.552338 | 1.053333 | 0.492308 | 0.526667 |
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.423077 | 15.996907 | 1.157895 | 0.578947 | 0.578947 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.606667 | 12.361246 | 0.894737 | 0.447368 | 0.447368 |
| missing_evidence | 36 | 36 | 36 | 0.472222 | 0.545455 | 14.583039 | 1.055556 | 0.515152 | 0.527778 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.642857 | 15.269775 | 1.105263 | 0.380952 | 0.552632 |

## By category (across sources)

| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |
|---|---|---|---|---|---|---|---|---|---|
| candidate_selection | 38 | 38 | 38 | 0.421053 | 0.423077 | 15.996907 | 1.157895 | 0.578947 | 0.578947 |
| evidence_judgment | 38 | 38 | 38 | 0.552632 | 0.606667 | 12.361246 | 0.894737 | 0.447368 | 0.447368 |
| missing_evidence | 36 | 36 | 36 | 0.472222 | 0.545455 | 14.583039 | 1.055556 | 0.515152 | 0.527778 |
| rule_application | 38 | 38 | 38 | 0.447368 | 0.642857 | 15.269775 | 1.105263 | 0.380952 | 0.552632 |

## Reliability curve (per stratum)

### `authored-zh-v1` / `candidate_selection` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.4211 | 0.5789 |

### `authored-zh-v1` / `evidence_judgment` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.5526 | 0.4474 |

### `authored-zh-v1` / `missing_evidence` (n_scored = 36)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 36 | 1.0000 | 0.4722 | 0.5278 |

### `authored-zh-v1` / `rule_application` (n_scored = 38)

| bin | n | mean confidence | accuracy | gap |
|---|---|---|---|---|
| [0.90, 1.00) | 38 | 1.0000 | 0.4474 | 0.5526 |
