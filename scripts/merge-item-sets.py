#!/usr/bin/env python3
"""Merge item sets into one frozen set, with a single `source` stratum value.

Why this is a script and not an editing pass: the output is a *frozen artifact*
whose sha256 goes into `eval/items/manifest.json` and into every report's
`items_sha256`. The same inputs must therefore produce the same bytes, and a
hand-edited line is how a set quietly stops being reproducible. It also lets the
older set file stay untouched, so reports generated against it keep verifying.

Ordering is by `id`, not by argument order, so the output does not depend on how
the inputs were passed. Duplicate ids are a hard error.

Usage:
  merge-item-sets.py --source <stratum> --out <out.jsonl> [--header <file>] <input.jsonl>...

`#` comment lines and blank lines in the inputs are dropped; the header (if given)
is copied verbatim to the top, because the numbers it documents — sizes, balance —
only exist after the merge has been measured.
"""

import argparse
import json
import pathlib
import sys


def read_items(path):
    items = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            items.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from None
    return items


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="source value for every item")
    parser.add_argument("--out", required=True)
    parser.add_argument("--header", help="comment block copied to the top of the output")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv[1:])

    by_id = {}
    for name in args.inputs:
        path = pathlib.Path(name)
        for item in read_items(path):
            if "id" not in item:
                raise SystemExit(f"{path}: item without id: {item!r}")
            if item["id"] in by_id:
                raise SystemExit(f"{path}: duplicate id {item['id']}")
            item["source"] = args.source
            by_id[item["id"]] = item

    lines = []
    if args.header:
        header = pathlib.Path(args.header).read_text(encoding="utf-8").rstrip("\n")
        lines.extend(header.splitlines())
        lines.append("")
    for item_id in sorted(by_id):
        lines.append(json.dumps(by_id[item_id], ensure_ascii=False, separators=(",", ":")))

    pathlib.Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(by_id)} items from {len(args.inputs)} input(s), source={args.source}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
