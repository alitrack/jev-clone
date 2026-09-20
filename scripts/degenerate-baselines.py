#!/usr/bin/env python3
"""Generate the degenerate-strategy baselines `eval/README.md` §5 says are missing.

§5 puts it plainly: the sets have only *item-side* balance metrics (always-answer-slot-X
tops out at ~44%, always-take-the-written-first-option at ~52%) and the strategies
themselves were never run, so **no single ECE/Brier number means anything yet** —
"until the baselines are run, a single ECE/Brier number cannot show a model is good or
bad". This script closes exactly that hole.

Three strategies, produced from the item set alone (no model, no network):

| name | what it does | probability vector |
|---|---|---|
| `slot-a` | always takes slot **A** (code-point-first key — the renderer's slot order) | 1.0 on that slot |
| `written-first` | always takes the option **as written first** in the item file — a different order from code-point order, which is why both are generated | 1.0 on that slot |
| `slot-prior` | always predicts the **most common gold slot**, but spreads its probability over the item set's own slot prior | the set's marginal gold distribution |

`slot-prior` is the interesting one: it is a constant predictor whose confidence equals
the base rate, so its ECE lands near zero while its accuracy stays at the base rate.
It is the concrete counter-example to "low ECE ⇒ calibrated model".

Usage: degenerate-baselines.py --items <items.jsonl> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_items(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(json.loads(s))
    return out


def slots_of(item: dict) -> list[str]:
    """Slot order = the order the renderer assigns letters in.

    `choice` — code-point sort of the criteria keys (the contract's criteria is a
    BTreeMap, so this is what the renderer sees).
    `score` — the criteria list is already ordered.
    `noul` — fixed by the contract to `["true", "false"]`: the canonical positive
    class takes the first letter. NOT code-point order — `"false" < "true"`, so
    sorting here would hand slot A to `false`. Authority:
    `crates/jev-eval/src/items.rs:75-80` (`slots_of`) and its test at `:439`.
    """
    q = item["question"]
    if q["type"] == "choice":
        return sorted(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["true", "false"]


def gold_key(item: dict) -> str:
    q = item["question"]
    if q["type"] == "score":
        return str(item["gold"])
    if q["type"] == "noul":
        return "true" if item["gold"] is True else "false"
    return item["gold"]


def written_first_slot(item: dict) -> str:
    """The slot a "always take the first option" strategy would pick.

    `choice` — the key written first in the item file. That is NOT the letter slot
    (letters follow code-point order), which is exactly what makes this baseline
    worth measuring separately.
    `score` / `noul` — there is no written option list in the question, so this falls
    back to the renderer's first slot (for `noul` that is `true`).
    """
    q = item["question"]
    if q["type"] == "choice":
        return next(iter(q["criteria"]))  # JSON key order == order written in the file
    return slots_of(item)[0]


def uniform_zero(slots: list[str]) -> dict:
    return {s: 0.0 for s in slots}


def build(items: list[dict]) -> dict[str, list[dict]]:
    # The prior is over SLOT POSITIONS (A/B/C, level index, false/true), not over label
    # strings: a label's base rate from other items is meaningless here (item 26's options
    # are 甲/乙/丙 while item 1's are 支持/不支持), and a per-label prior would hand this
    # item a vector of zeros — which is not a probability distribution at all.
    #
    # Grouped by (type, number of slots) for the same reason: a 2-option item scored
    # against a 3-option position prior sums to ~0.87, which the contract rejects outright.
    # It is the item's own cohort — same type, same width — that the base rate means.
    pos_prior: dict[tuple, collections.Counter] = {}
    for item in items:
        slots = slots_of(item)
        g = gold_key(item)
        idx = slots.index(g) if g in slots else 0
        key = (item["question"]["type"], len(slots))
        pos_prior.setdefault(key, collections.Counter())[str(idx)] += 1
    priors = {k: {p: v / sum(c.values()) for p, v in c.items()} for k, c in pos_prior.items()}

    strategies = {"slot-a": [], "written-first": [], "slot-prior": []}
    for item in items:
        slots = slots_of(item)
        t = item["question"]["type"]

        p = uniform_zero(slots)
        p[slots[0]] = 1.0
        strategies["slot-a"].append({"id": item["id"], "probabilities": p})

        p = uniform_zero(slots)
        wf = written_first_slot(item)
        p[wf] = 1.0
        strategies["written-first"].append({"id": item["id"], "probabilities": p})

        # Constant predictor whose confidence equals the base rate of the slot it picks.
        cohort = priors[(t, len(slots))]
        p = {s: cohort.get(str(i), 0.0) for i, s in enumerate(slots)}
        strategies["slot-prior"].append({"id": item["id"], "probabilities": p})
    return strategies


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="", help="文件名前缀标签（题集名），避免不同题集的基线互相覆盖")
    ap.add_argument(
        "--stamp",
        default="",
        help="运行时间戳前缀。带上它，同一题集重跑就不会覆盖上一次的预测文件"
        "（否则旧报告的 predictions_sha256 会对不上，报告再也无法重算）",
    )
    args = ap.parse_args(argv[1:])

    items = read_items(pathlib.Path(args.items))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategies = build(items)
    tag = f"{args.tag}-" if args.tag else ""
    stamp = f"{args.stamp}-" if args.stamp else ""

    print(f"题集 {args.items}（{len(items)} 条）\n")
    for name, rows in strategies.items():
        path = out_dir / f"{stamp}deg-{tag}{name}-predictions.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                                for r in rows), encoding="utf-8")
        # item-side accuracy, so the number can be checked against validate-items.py's
        # printed degenerate ceilings before any scoring happens
        ok = sum(1 for it, r in zip(items, rows)
                 if max(r["probabilities"], key=lambda k: r["probabilities"][k]) == gold_key(it))
        print(f"{name:14s} 题集侧命中 {ok}/{len(items)} = {ok/len(items):.4f}   -> {path}")
    print("\n（`slot-a` / `written-first` 的命中率应与 scripts/validate-items.py 打印的退化上限一致；"
          "不一致说明槽位或书写序口径复现错了）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
