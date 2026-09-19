#!/usr/bin/env python3
"""Digest a `jev-eval` run into per-item errors: what was answered, by stratum.

The report is deliberately stratified and has no merged total, so it tells you
*that* `rule_application` is weak — not *which* items failed or what the model
reached for instead. This prints exactly that, because "the model got 12 wrong"
is not actionable while "it answered 支持 on 11 assertions whose材料 says the
opposite" is.

Groups errors by category and by question type, and — the part that matters for
a decision task — counts how often the wrong answer was the *other* slot rather
than a structurally different mistake.

Usage: error-digest.py <items.jsonl> <predictions.jsonl> [--max 40]
"""

import argparse
import collections
import json
import pathlib
import sys


def load_jsonl(path):
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(json.loads(s))
    return out


def slot_keys(question):
    """The item's declared slot order, as `crates/jev-eval/src/items.rs` defines it.

    choice: the criteria keys in dictionary order (the contract's `criteria` is a
    BTreeMap); score: the level indices as decimal strings; noul: true then false.
    Reimplemented rather than guessed — these three lines are asserted by unit
    tests in that file, so a divergence here shows up as a wrong digest.
    """
    kind = question["type"]
    if kind == "choice":
        return sorted(question["criteria"].keys())
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["true", "false"]


def predicted_slot(pred, slots):
    """Argmax of the probability vector, ties to the earliest slot; else `label`."""
    probs = pred.get("probabilities")
    if probs:
        values = [float(probs.get(slot, 0.0)) for slot in slots]
        best = 0
        for i, value in enumerate(values):
            if value > values[best]:
                best = i
        return slots[best]
    return pred.get("label")


def normalise(gold, kind):
    """Slot key as a string. noul golds are JSON booleans, score golds integers,
    but the contract scores on slot *keys* (`"true"/"false"`, `"0".."n-1"`) — a
    digest that compares `1` to `"1"` reports every score item as wrong."""
    if kind == "noul":
        return str(gold).lower()
    return str(gold)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("items")
    parser.add_argument("predictions")
    parser.add_argument("--max", type=int, default=40, help="max error lines to print")
    args = parser.parse_args(argv[1:])

    items = {i["id"]: i for i in load_jsonl(args.items)}
    preds = {p["id"]: p for p in load_jsonl(args.predictions)}

    wrong = []
    missing = []
    by_cat = collections.Counter()
    by_type = collections.Counter()
    by_cat_total = collections.Counter()
    by_type_total = collections.Counter()
    noul_confusion = collections.Counter()
    score_off_by = collections.Counter()
    flips = collections.Counter()

    for item_id, item in items.items():
        kind = item["question"]["type"]
        cat = item["category"]
        by_cat_total[cat] += 1
        by_type_total[kind] += 1
        pred = preds.get(item_id)
        slots = slot_keys(item["question"])
        got = None if pred is None else predicted_slot(pred, slots)
        if got is None:
            missing.append(item_id)
            continue
        gold = normalise(item["gold"], kind)
        got = normalise(got, kind)
        if got == gold:
            continue
        wrong.append((cat, kind, item_id, gold, got, item))
        by_cat[cat] += 1
        by_type[kind] += 1
        if kind == "noul":
            noul_confusion[(str(gold), str(got))] += 1
        if kind == "score":
            try:
                score_off_by[int(got) - int(gold)] += 1
            except (TypeError, ValueError):
                pass
        # Direction of the miss, for the denial assertions this set leans on
        # (`不支持` / `false` as the negative side). A model that answers
        # `支持` where the material says the opposite is making one mistake, not
        # six — worth separating from genuinely scattered errors.
        neg_gold = "不" in gold or gold == "false"
        neg_got = "不" in got or got == "false"
        if neg_gold and not neg_got:
            flips["negation missed (gold 否定 -> answered 肯定)"] += 1
        elif neg_got and not neg_gold:
            flips["negation over-applied (gold 肯定 -> answered 否定)"] += 1

    total = len(items)
    print(f"items {total} / predictions {len(preds)} / wrong {len(wrong)} / no-answer {len(missing)}")
    print(f"accuracy {1 - len(wrong) / total:.4f}" if total else "accuracy n/a")
    if missing:
        print(f"no-answer ids: {' '.join(missing[:20])}")
    print()
    print("wrong by category (err/total):")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<22} {n:>3} / {by_cat_total[cat]:<3} = {n / by_cat_total[cat]:.3f}")
    print("wrong by type (err/total):")
    for kind, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:<22} {n:>3} / {by_type_total[kind]:<3} = {n / by_type_total[kind]:.3f}")
    if noul_confusion:
        print(f"noul confusion (gold -> answered): {dict(noul_confusion)}")
    if score_off_by:
        print(f"score distance (answered - gold): {dict(sorted(score_off_by.items()))}")
    if flips:
        for label, n in flips.items():
            print(f"direction: {label}: {n}")
    print()
    for cat, kind, item_id, gold, got, item in wrong[: args.max]:
        print(f"[{cat}/{kind}] {item_id}  gold={gold}  answered={got}")
        print(f"    Q: {item['question']['instructions']}")
        if item.get("provenance"):
            print(f"    why gold: {item['provenance']}")
    if len(wrong) > args.max:
        print(f"... {len(wrong) - args.max} more errors not printed (--max)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
