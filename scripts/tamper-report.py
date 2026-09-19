#!/usr/bin/env python3
"""Write a tampered copy of a `jev-eval` report, to be caught by `verify`.

Two tampers at once, because each alone is a different attack:

1. a published number is moved (the ECE of the first source's overall stratum), so
   the report no longer matches the raw evidence;
2. the report declares a huge `verify_tolerance`, so a verifier that trusted the
   artifact's own tolerance would call the moved number a match.

Before 2026-09-19 the second tamper was enough on its own: `cmd_verify` read the
tolerance out of the report, so `"verify_tolerance": 1000000000` made a rewritten
report print `0 mismatches` and exit 0. A verifier that still refuses this file is
a verifier that picked its own threshold.

Usage: tamper-report.py <report.json> <out.json>
"""

import copy
import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: tamper-report.py <report.json> <out.json>", file=sys.stderr)
        return 2
    source_path, out_path = sys.argv[1], sys.argv[2]

    with open(source_path, encoding="utf-8") as handle:
        report = json.load(handle)
    tampered = copy.deepcopy(report)

    by_source = tampered["strata"]["by_source"]
    if not by_source:
        print("report has no strata to tamper with", file=sys.stderr)
        return 1
    source = next(iter(by_source))
    overall = by_source[source]["overall"]
    before = overall.get("ece")
    overall["ece"] = (before or 0.0) + 0.5
    tampered["verify_tolerance"] = 1e9

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(tampered, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"tampered copy: {out_path} "
        f"(ece {before} -> {overall['ece']}, verify_tolerance -> 1e9)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
