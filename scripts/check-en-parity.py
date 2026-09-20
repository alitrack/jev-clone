#!/usr/bin/env python3
"""Gate the English control set against its Chinese original (M3-1).

The English set exists to make ONE comparison valid: same content, two languages,
scored by the same harness.  Two things can silently destroy that:

1. **Slot drift.**  A `choice` item's slots are its `criteria` keys *sorted by code
   point* (see `crates/jev-core/src/render.rs`), and the readout reads the letter
   after `Answer:`.  Translate the labels and the sort order can flip, so the
   correct answer moves from slot B to slot A.  The pair then still looks fine in
   both files while comparing a model's slot bias instead of its language skill.
2. **Answer drift.**  Every `gold` must remain the correct option under translation;
   `score` golds and `noul` booleans must be *identical*, and `choice` golds must
   point at the same slot as the Chinese item.

It also prints the same balance report `validate-items.py` prints, so the two sets
can be compared stratum by stratum (a set whose degenerate-strategy ceiling differs
wildly from its counterpart is not a control set).

Usage: check-en-parity.py <zh.jsonl> <en.jsonl>
Exit code is non-zero on any pairing/slot/answer problem, so it can gate a commit.
"""

import collections
import hashlib
import json
import pathlib
import sys

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_items(path: pathlib.Path) -> list[dict]:
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


def numeric_id(item: dict) -> str:
    """`zh-ev-0031` / `en-ev-0031` -> `0031` (the pairing key)."""
    raw = item["id"]
    return raw.split("-")[-1]


def slot_index(item: dict) -> int:
    """0-based position of `gold` among the item's slots, as the renderer orders them."""
    q = item["question"]
    kind = q["type"]
    if kind == "choice":
        keys = sorted(q["criteria"].keys())
        return keys.index(item["gold"])
    if kind == "score":
        return int(item["gold"])
    if kind == "noul":
        return 0 if item["gold"] is True else 1
    raise SystemExit(f"{item['id']}: unknown question type {kind!r}")


def balance(items: list[dict]) -> dict:
    by_cat = collections.Counter(i["category"] for i in items)
    by_type = collections.Counter(i["question"]["type"] for i in items)
    choice = [i for i in items if i["question"]["type"] == "choice"]
    slots = collections.Counter(LETTERS[slot_index(i)] for i in choice)
    first = sum(1 for i in choice if slot_index(i) == 0)
    noul = collections.Counter(bool(i["gold"]) for i in items if i["question"]["type"] == "noul")
    score = collections.Counter(int(i["gold"]) for i in items if i["question"]["type"] == "score")
    return {
        "by_category": dict(by_cat), "by_type": dict(by_type),
        "choice_slots": dict(slots), "choice_first": first, "choice_n": len(choice),
        "noul": dict(noul), "score": dict(score),
    }


def print_balance(tag: str, items: list[dict]) -> None:
    b = balance(items)
    print(f"[{tag}] n={len(items)}  by_category={b['by_category']}  by_type={b['by_type']}")
    if b["choice_n"]:
        slot_pct = {k: f"{v}({v * 100 // b['choice_n']}%)" for k, v in sorted(b["choice_slots"].items())}
        print(f"        choice gold slot={slot_pct}  恒取首选项上限={b['choice_first']}"
              f"({b['choice_first'] * 100 // b['choice_n']}%)")
    print(f"        noul={b['noul']}  score={b['score']}")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    zh_path, en_path = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    zh, en = read_items(zh_path), read_items(en_path)
    problems: list[str] = []
    notes: list[str] = []

    zh_by, en_by = {}, {}
    for tag, items, bucket in (("zh", zh, zh_by), ("en", en, en_by)):
        for it in items:
            key = numeric_id(it)
            if key in bucket:
                problems.append(f"{tag} 集 id 重复：{key}")
            bucket[key] = it

    missing = sorted(set(zh_by) - set(en_by))
    extra = sorted(set(en_by) - set(zh_by))
    if missing:
        problems.append(f"英文集缺 {len(missing)} 条：{missing[:12]}")
    if extra:
        problems.append(f"英文集多 {len(extra)} 条：{extra[:12]}")

    slot_drift, answer_drift = [], []
    for key in sorted(set(zh_by) & set(en_by)):
        z, e = zh_by[key], en_by[key]
        if z["category"] != e["category"]:
            problems.append(f"{key}: category 不一致 {z['category']} -> {e['category']}")
        if z["question"]["type"] != e["question"]["type"]:
            problems.append(f"{key}: 题型不一致 {z['question']['type']} -> {e['question']['type']}")
            continue
        # Structural parity: a control set must differ in LANGUAGE only. If the translation
        # added or dropped a question field, the pair compares wording plus structure, and
        # the comparison stops being language-vs-language. (This is the check that the
        # pilot's "noul 不该有 criteria" rule got backwards — `NoulQuestion.criteria` is
        # `Option<NoulCriteria>` per contract.rs:50-56, so what matters is that the field is
        # there on both sides, identical in shape.)
        #
        # A key whose value is `null` counts as ABSENT: the zh set spells the field out as
        # `"criteria": null` on 24 noul items and omits it on 8, and both are the same
        # `None` to Rust. Comparing raw key sets would call that pair a mismatch.
        zq, eq = z["question"], e["question"]
        zk = {k for k, v in zq.items() if v is not None}
        ek = {k for k, v in eq.items() if v is not None}
        if zk != ek:
            problems.append(f"{key}: question 键集合不一致（忽略 null 后）zh{sorted(zk)} -> en{sorted(ek)}")
        if zq["type"] == "noul":
            zc, ec = zq.get("criteria"), eq.get("criteria")
            if (zc is None) != (ec is None):
                problems.append(f"{key}: noul.criteria 有无不一致（一方 null/缺、一方有值）")
            elif zc is not None and (
                not isinstance(zc, dict) or not isinstance(ec, dict) or set(zc) != set(ec)
            ):
                problems.append(f"{key}: noul.criteria 的 true/false 键不一致 {zc!r} -> {ec!r}")
        t = zq["type"]
        if t == "choice":
            zl, el = sorted(z["question"]["criteria"].keys()), sorted(e["question"]["criteria"].keys())
            if len(zl) != len(el):
                problems.append(f"{key}: 选项数不一致 {len(zl)} -> {len(el)}")
                continue
            if e["gold"] not in e["question"]["criteria"]:
                problems.append(f"{key}: 英文 gold 不是声明过的槽位：{e['gold']!r}")
                continue
            zs, es = slot_index(z), slot_index(e)
            if zs != es:
                slot_drift.append((key, LETTERS[zs], LETTERS[es], z["gold"], e["gold"]))
        elif t == "score":
            if z["gold"] != e["gold"]:
                answer_drift.append((key, z["gold"], e["gold"]))
            if len(z["question"]["criteria"]) != len(e["question"]["criteria"]):
                problems.append(f"{key}: score 档位数不一致")
        elif t == "noul":
            if bool(z["gold"]) != bool(e["gold"]):
                answer_drift.append((key, z["gold"], e["gold"]))
        # A translated `state` that still contains CJK is suspicious, not fatal.
        if any("\u4e00" <= c <= "\u9fff" for c in e["state"]):
            notes.append(f"{key}: 英文 state 里仍有中文字符")

    if slot_drift:
        problems.append(f"⛔ 槽位漂移 {len(slot_drift)} 条（中英 gold 落点不同的字母槽）")
    if answer_drift:
        problems.append(f"⛔ 答案漂移 {len(answer_drift)} 条（score/noul gold 被改动）")

    print_balance("zh", zh)
    print_balance("en", en)
    print()

    if slot_drift:
        print(f"### 槽位漂移明细（{len(slot_drift)} 条）—— 这些配对不能用于中英对照，必须修")
        for key, zs, es, zgold, egold in slot_drift:
            print(f"  {key}: zh {zs}({zgold}) -> en {es}({egold})")
        print()
    if answer_drift:
        print(f"### 答案漂移明细（{len(answer_drift)} 条）—— 译文改了答案，必须修")
        for key, zg, eg in answer_drift:
            print(f"  {key}: zh {zg!r} -> en {eg!r}")
        print()
    if notes:
        print(f"### 备注（{len(notes)} 条，人工看）")
        for n in notes[:15]:
            print("  -", n)
        print()

    sha = hashlib.sha256(en_path.read_bytes()).hexdigest()
    print(f"en sha256: {sha}")
    print(f"配对: {len(set(zh_by) & set(en_by))}/{len(zh)}")

    if problems:
        print(f"\n❌ {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
        return 1
    print("\n✅ 配对完整、槽位一致、答案未漂移 —— 可以作为对照集")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
