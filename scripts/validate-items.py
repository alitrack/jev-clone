#!/usr/bin/env python3
"""Validate a frozen item set the way `jev-eval` will, before the crate does.

Checks, in the order they can actually bite:

1. every non-comment line is valid JSON — an ASCII quote inside a Chinese string
   silently broke a whole set the first time round;
2. ids are unique;
3. required fields are present and `category` is one of the four strata the report
   stratifies by (a fifth value would silently create a stratum nobody reads);
4. `gold` is one of the item's own declared slots — the most dangerous field in the
   file, because a wrong gold is not a crash, it is a wrong number;
5. `positive`, when given, is a declared slot too (it decides `brier_binary`);
6. `provenance` is present: a gold nobody can re-derive from `state` is a claim, and
   this set exists precisely to measure the model, not the author.

It then prints the balance report the campaign needs — because a degenerate strategy
("always answer the first slot", "always answer true") must not score well, which
means the *slot* of the gold matters, not just the label:

* category and question-type distributions;
* for each choice item, which letter slot the gold lands on (slot order is the
  criteria keys sorted by code point, i.e. exactly what the renderer does);
* the noul true/false split;
* the spread of score golds.

Finally it prints the sha256 that belongs in `eval/items/manifest.json`.

Usage: validate-items.py <items.jsonl>
Exit code is non-zero on any structural problem, so it can gate a commit.
"""

import collections
import hashlib
import json
import pathlib
import sys

CATEGORIES = {
    "evidence_judgment",
    "rule_application",
    "candidate_selection",
    "missing_evidence",
}
REQUIRED = {"id", "category", "source", "state", "question", "gold", "provenance"}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def slots_of(item):
    """The declared slots of an item, in the order the renderer will use.

    `match`-free by design: choice criteria are keys sorted by code point, score
    criteria are already an ordered list, noul is fixed by the contract.
    """
    question = item["question"]
    kind = question.get("type")
    if kind == "choice":
        return sorted(question["criteria"].keys())
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    if kind == "noul":
        return ["true", "false"]
    return []


def normalise(gold, kind):
    """The slot label a gold value refers to.

    noul golds are JSON booleans in the item file but `"true"`/`"false"` in the
    contract's slot space; everything else is already its own label.
    """
    return str(gold).lower() if kind == "noul" else gold


def main(argv):
    if len(argv) != 2:
        print("usage: validate-items.py <items.jsonl>", file=sys.stderr)
        return 2
    path = pathlib.Path(argv[1])
    raw = path.read_bytes()
    text = raw.decode("utf-8")

    items, problems, seen = [], [], {}
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            problems.append(f"line {lineno}: invalid JSON: {exc}")
            continue
        where = item.get("id", f"line {lineno}")
        missing = REQUIRED - set(item)
        if missing:
            problems.append(f"{where}: missing field(s) {sorted(missing)}")
        if item.get("id") in seen:
            problems.append(f"{where}: duplicate id (also on line {seen[item['id']]})")
        seen[item.get("id")] = lineno
        if item.get("category") not in CATEGORIES:
            problems.append(f"{where}: category {item.get('category')!r} is not a stratum")
        kind = item.get("question", {}).get("type")
        if kind not in {"choice", "score", "noul"}:
            problems.append(f"{where}: question.type {kind!r} is not in the contract")
        slots = slots_of(item) if kind in {"choice", "score", "noul"} else []
        gold = item.get("gold")
        if kind == "score":
            if not isinstance(gold, int) or isinstance(gold, bool) or not 0 <= gold < len(slots):
                problems.append(f"{where}: score gold {gold!r} is not a level index in range")
        elif kind in {"choice", "noul"}:
            # noul is the odd one: the contract's slot labels are the *strings*
            # "true"/"false" (that is what the renderer writes and what `positive`
            # names), while the item's `gold` is a JSON boolean. Normalise before
            # comparing, and use the same normalisation for the balance report so
            # the two can never disagree.
            if normalise(gold, kind) not in slots:
                problems.append(f"{where}: gold {gold!r} is not one of the declared slots {slots}")
        positive = item.get("positive")
        if positive is not None and positive not in slots:
            problems.append(f"{where}: positive {positive!r} is not a declared slot")
        if not str(item.get("provenance", "")).strip():
            problems.append(f"{where}: empty provenance — the gold must be re-derivable")
        items.append(item)

    if problems:
        print(f"{len(problems)} problem(s) in {path}:")
        for problem in problems:
            print("  -", problem)
        return 1

    by_category = collections.Counter(i["category"] for i in items)
    by_type = collections.Counter(i["question"]["type"] for i in items)
    gold_slot = collections.Counter()
    noul_split = collections.Counter()
    score_golds = collections.Counter()
    written_first = 0
    for item in items:
        slots = slots_of(item)
        kind = item["question"]["type"]
        if kind == "choice":
            slot = LETTERS[slots.index(item["gold"])]
            gold_slot[slot] += 1
            # The letter slot is decided by code-point order, which is not the order
            # a human reads the item file in. The strategy that actually needs
            # defusing is "always take the option as written first", so measure that
            # too rather than assuming the two orders agree.
            written = list(item["question"]["criteria"].keys())[0]
            written_first += item["gold"] == written
        elif kind == "noul":
            noul_split[normalise(item["gold"], kind)] += 1
        else:
            score_golds[item["gold"]] += 1

    print(f"{path}: {len(items)} items, no structural problems")
    print(f"  sha256  : {hashlib.sha256(raw).hexdigest()}")
    print(f"  by category : {dict(sorted(by_category.items()))}")
    print(f"  by type     : {dict(sorted(by_type.items()))}")
    if gold_slot:
        total_choice = sum(gold_slot.values())
        width = LETTERS.index(max(gold_slot)) + 1
        shares = " ".join(
            f"{letter}={gold_slot.get(letter, 0)}({gold_slot.get(letter, 0) / total_choice:.0%})"
            for letter in LETTERS[:width]
        )
        print(f"  choice gold slot (n={total_choice}): {shares}")
        print(
            f"  gold == first option as written (n={total_choice}): {written_first}"
            f" ({written_first / total_choice:.0%})"
            + (
                "  OK"
                if written_first / total_choice <= 0.55
                else "  ⚠️ 'always answer the first option listed' would score well"
            )
        )
        worst = max(gold_slot.values()) / total_choice
        print(
            f"  degenerate-strategy check (slot): most common slot {worst:.0%} of choice items"
            + ("" if worst <= 0.55 else "  ⚠️ a constant slot answer would score well")
        )
    if noul_split:
        total_noul = sum(noul_split.values())
        minority = min(noul_split.values()) / total_noul
        print(
            f"  noul split (n={total_noul}): {dict(noul_split)} — minority {minority:.0%}"
            + ("" if minority >= 0.4 else " ⚠️ a constant answer would score too well")
        )
    if score_golds:
        print(f"  score golds (n={sum(score_golds.values())}): {dict(sorted(score_golds.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
