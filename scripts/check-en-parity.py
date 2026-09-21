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
3. **Fact drift.**  The translation must not change a number or an identifier.
   A control set that says `181 至 260 立方米` on one side and "161 to 260 cubic
   meters" on the other is not the same question twice, it is two different
   questions in two languages.  Measured on the M4-1 MT run: the tiered water-price
   items came back with `181`→"161" and `260`→"250".  The predicate lives in
   `scripts/qc_common.py` and is shared with the translator, so both agree what
   counts as a lost fact (numbers are compared as *quantities*, so `1,400`≡`1400`
   and `180 万`≡"1.8 million" are not false alarms).

It also prints the same balance report `validate-items.py` prints, so the two sets
can be compared stratum by stratum (a set whose degenerate-strategy ceiling differs
wildly from its counterpart is not a control set).

Usage: check-en-parity.py <zh.jsonl> <en.jsonl>
       check-en-parity.py <zh.jsonl> <en.jsonl> --allow-missing <translate-mt.py 的 --report json>

M4 的 v2 集是**专用 MT 重译**的（`en-evidence-v2.jsonl`），specs/M4.md §S3 规定：译文不合用的
题**只允许整条剔除**（不许人工改译文），剔除要计入 `n_missing` 并逐条列名。所以 v2 是 zh 集的
**子集**。`--allow-missing` 把「英文集缺 id」拆成两类：报告里**申报过的剔除**（记 n_missing，放行）
与**没申报的缺失**（仍是硬错——它意味着译文悄悄少了几条）。两边必须逐条对齐，多报少报都算问题。
Exit code is non-zero on any pairing/slot/answer problem, so it can gate a commit.
"""

import collections
import hashlib
import json
import pathlib
import re
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


def dropped_ids_from_report(path: pathlib.Path) -> set[str]:
    """从 `translate-mt.py --report` 的 JSON 里取出「整条剔除」的 id（M4 §S3）。

    条目形如 `zh-ev-0003(evidence_judgment): 槽位漂移：…`；取不出 id 直接报错，
    免得闸门拿到一份对不上的清单还以为核对过了。
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for entry in report.get("dropped", []):
        m = re.search(r"(?:zh|en)-ev-(\d+)", str(entry))
        if m is None:
            raise SystemExit(f"报告 {path} 的 dropped 条目没带 id，无法核对：{entry!r}")
        ids.add(m.group(1))
    n_report = report.get("n_dropped")
    if n_report is not None and n_report != len(ids):
        raise SystemExit(f"报告 {path} 自相矛盾：n_dropped={n_report} 但 dropped 有 {len(ids)} 条")
    return ids


def main(argv: list[str]) -> int:
    rest = argv[1:]
    allowed_missing: set[str] | None = None
    if "--allow-missing" in rest:
        i = rest.index("--allow-missing")
        if i + 1 >= len(rest):
            print("--allow-missing 需要 <translate-mt.py 的 --report json>")
            return 2
        allowed_missing = dropped_ids_from_report(pathlib.Path(rest[i + 1]))
        del rest[i : i + 2]
    if len(rest) != 2:
        print(__doc__)
        return 2
    zh_path, en_path = pathlib.Path(rest[0]), pathlib.Path(rest[1])
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
    if allowed_missing is None:
        if missing:
            problems.append(f"英文集缺 {len(missing)} 条：{missing[:12]}")
    else:
        # M4 §S3：允许「整条剔除」，但**剔了哪些必须与报告逐条对齐**（多报少报都是错）。
        undeclared = [k for k in missing if k not in allowed_missing]
        stale = sorted(k for k in allowed_missing if k not in set(missing))
        if undeclared:
            problems.append(f"英文集缺 {len(undeclared)} 条，但报告未申报剔除：{undeclared[:12]}")
        if stale:
            problems.append(f"报告申报剔除、英文集里却仍在：{stale[:12]}")
        if not undeclared and not stale:
            notes.append(f"按 §S3 整条剔除 {len(missing)} 条（与报告 dropped 逐条对齐）：{missing[:12]}")
    if extra:
        problems.append(f"英文集多 {len(extra)} 条：{extra[:12]}")

    # 事实核对 / 退化检测的谓词来自 scripts/qc_common.py（与 translate-mt.py 共用同一实现；
    # 单实现是刻意的：这两处判据必须一致，两套代码迟早分叉）。
    from qc_common import degenerate_reason, discount_lost, facts_lost

    slot_drift, answer_drift, fact_drift, broken, prov_drift = [], [], [], [], []
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
        # 事实核对——**只查模型能看到的字段**（state / instructions / criteria 标签）：这些漂了就是
        # 「同一道题的两个版本」不成立（specs/M4.md §S3：不合用只允许整条剔除）。
        lost = [f"state:{x}" for x in facts_lost(z["state"], e["state"])]
        lost += [f"instr:{x}" for x in facts_lost(z["question"].get("instructions") or "",
                                                  e["question"].get("instructions") or "")]
        if t == "choice":
            en_crit = json.dumps(e["question"]["criteria"], ensure_ascii=False)
            lost += [f"crit:{k}" for k in z["question"]["criteria"] if facts_lost(k, en_crit)]
        if lost:
            fact_drift.append((key, lost))
        # provenance 是出题人写的理由，**不进模型输入**（只在报告里给人看）。它译错不影响这一对的
        # 对照语义，故记 note 并单独计数（M4-4 里如实披露），不判死。实测：0090 的中文「不足 30 件」
        # 在英文 provenance 里成了 "the 35"——真错，但错在解释文字，不在题面。
        prov_lost = facts_lost(z.get("provenance") or "", e.get("provenance") or "")
        if prov_lost:
            prov_drift.append((key, prov_lost))
        deg = degenerate_reason(z["state"], e["state"])
        if deg:
            broken.append((key, deg))
        # A translated `state` that still contains CJK is suspicious, not fatal.
        if any("\u4e00" <= c <= "\u9fff" for c in e["state"]):
            notes.append(f"{key}: 英文 state 里仍有中文字符")
        # 软判据：「N 折」没被译出折扣字样（写法约定问题，不进致命清单）。
        if discount_lost(z["state"], e["state"]):
            notes.append(f"{key}: 中文写「N 折」，英文里没有折扣字样")

    if slot_drift:
        problems.append(f"⛔ 槽位漂移 {len(slot_drift)} 条（中英 gold 落点不同的字母槽）")
    if answer_drift:
        problems.append(f"⛔ 答案漂移 {len(answer_drift)} 条（score/noul gold 被改动）")
    if fact_drift:
        problems.append(f"⛔ 事实漂移 {len(fact_drift)} 条（译文改了数字/编号，须整条剔除）")
    if broken:
        problems.append(f"⛔ 译文损坏 {len(broken)} 条（复读机 / 长度爆炸）")
    if prov_drift:
        notes.append(f"provenance 事实漂移 {len(prov_drift)} 条（不进模型输入，非致命，单列披露）："
                     f"{[k for k, _ in prov_drift][:12]}")

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
    if fact_drift:
        print(f"### 事实漂移明细（{len(fact_drift)} 条）—— 这些配对的英文侧事实与中文不同，"
              f"按 §S3 整条剔除（不许手工改译文）")
        for key, lost in fact_drift:
            print(f"  {key}: 丢失 {lost}")
        print()
    if broken:
        print(f"### 译文损坏明细（{len(broken)} 条）")
        for key, why in broken:
            print(f"  {key}: {why}")
        print()
    if prov_drift:
        print(f"### provenance 事实漂移（{len(prov_drift)} 条，非致命）——理由文字里的数字漂了，"
              f"题面不受影响，但 M4-4 要如实披露")
        for key, lost in prov_drift:
            print(f"  {key}: {lost}")
        print()
    if notes:
        print(f"### 备注（{len(notes)} 条，人工看）")
        for n in notes[:15]:
            print("  -", n)
        print()

    sha = hashlib.sha256(en_path.read_bytes()).hexdigest()
    print(f"en sha256: {sha}")
    tail = f"（n_missing={len(missing)}，逐条见报告 dropped）" if allowed_missing is not None else ""
    print(f"配对: {len(set(zh_by) & set(en_by))}/{len(zh)}{tail}")

    if problems:
        print(f"\n❌ {len(problems)} 个问题：")
        for p in problems:
            print("  -", p)
        return 1
    print("\n✅ 配对完整、槽位一致、答案未漂移 —— 可以作为对照集")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
