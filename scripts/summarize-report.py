#!/usr/bin/env python3
"""Print the headline numbers of a `jev-eval run` report.

The report is deliberately stratified and has no merged total (AGENTS.md 铁律 7 /
specs/M1.md §6), so this prints one block per source, then the metric table per
category. Kept as a file rather than an inline heredoc so the acceptance script
embeds no interpreter and so this stays usable on its own.

Usage: summarize-report.py <report.json>
"""
import json
import sys

METRICS = [
    ("n", "n_items"),
    ("n_scored", "n_scored"),
    ("acc", "accuracy"),
    ("bal_acc", "balanced_accuracy"),
    ("nll", "nll"),
    ("brier_mc", "brier_multiclass"),
    ("brier_bin", "brier_binary"),
    ("ece", "ece"),
]
FLOAT_KEYS = {"accuracy", "balanced_accuracy", "nll", "brier_multiclass", "brier_binary", "ece"}


def row(label, m):
    cells = []
    for _, key in METRICS:
        v = m.get(key)
        if v is None:
            cells.append("-")
        elif key in FLOAT_KEYS:
            cells.append(f"{v:.6f}")
        else:
            cells.append(str(v))
    return f"{label:<22} " + " ".join(f"{c:>10}" for c in cells)


report = json.load(open(sys.argv[1], encoding="utf-8"))
strata = report.get("strata", {})
print(f"items sha256       : {report.get('items_sha256')}")
print(f"prompt-set sha256  : {report.get('prompt_set_sha256')}")
print(f"predictions sha256 : {report.get('predictions_sha256')}")
print("strata: stratified by source and by category; no merged total by design")
print()
header = f"{'stratum':<22} " + " ".join(f"{n:>10}" for n, _ in METRICS)
for src, entry in strata.get("by_source", {}).items():
    print(f"== source {src}")
    print(header)
    print(row("overall", entry.get("overall", {})))
    for cat, m in sorted(entry.get("by_category", {}).items()):
        print(row(cat, m))
    print()
print("== by category (across sources)")
print(header)
for cat, m in sorted(strata.get("by_category", {}).items()):
    print(row(cat, m))
