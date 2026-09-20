#!/usr/bin/env python3
"""Paired zh-vs-en comparison for the M3 control pair.

Two kinds of number appear here and they come from different places on purpose:

* **Per-stratum aggregates** (accuracy / balanced accuracy / NLL / Brier / ECE) are
  read straight out of the two `jev-eval` reports. That crate is the authority on
  the metric definitions (`crates/jev-eval/src/metrics.rs`); recomputing them here
  would create a second implementation of the same formula and a second number.
* **The paired test** (2x2 discordance table + McNemar exact p-value) is computed
  here from `items` + `predictions`, because jev-eval reports per-stratum
  aggregates, not item-level pairing. The correctness rule mirrors the crate's:
  argmax over the reported probabilities with first-max tie-breaking, compared
  against `gold` rendered the way the contract reports it (label for `choice`,
  index for `score`, "true"/"false" for `noul`).

McNemar with n < 25 discordant pairs is reported as the exact binomial two-sided
p-value rather than chi-square, because chi-square is not trustworthy at that size
- and at n=150 per arm the discordant count *will* be small.

Usage:
  pair-diff.py --zh-report <r.json> --en-report <r.json> \
               [--zh-items <i.jsonl> --en-items <i.jsonl> \
                --zh-preds <p.jsonl> --en-preds <p.jsonl>]
"""

import argparse
import json
import math
import pathlib
import sys

METRICS = [
    ("accuracy", "acc", 4),
    ("balanced_accuracy", "bal_acc", 4),
    ("nll", "nll", 4),
    ("brier_multiclass", "brier_mc", 4),
    ("brier_binary", "brier_bin", 4),
    ("ece", "ece", 4),
]


def read_jsonl(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        doc = json.loads(s)
        out[doc["id"]] = doc
    return out


def gold_key(item: dict) -> str:
    """`gold` as the contract reports it (that is how prediction keys are named)."""
    kind = item["question"]["type"]
    if kind == "score":
        return str(item["gold"])
    if kind == "noul":
        return "true" if item["gold"] is True else "false"
    return item["gold"]


def argmax_first_max(probs: dict) -> str | None:
    """Mirrors `metrics.rs::argmax_first_max`: first maximum in key order."""
    best, best_v = None, None
    for k, v in probs.items():
        if best_v is None or v > best_v:
            best, best_v = k, v
    return best


def num(d: dict, key: str):
    v = d.get(key)
    return v if isinstance(v, (int, float)) else None


def fmt(v, nd: int) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def delta(a, b, nd: int) -> str:
    if a is None or b is None:
        return "-"
    d = b - a
    return f"{d:+.{nd}f}"


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact binomial p for McNemar (H0: P(zh right, en wrong) = P(en right, zh wrong))."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def numeric_key(item_id: str) -> str:
    """`zh-ev-0042` and `en-ev-0042` are the same item; pair on the numeric tail.

    The two sets carry different language prefixes on purpose (a report that mixed them up
    should be impossible), which makes an exact-id join empty — the same pairing rule as
    scripts/check-en-parity.py.
    """
    return item_id.rsplit("-", 1)[-1]


def stratum_table(a: dict, b: dict, label: str, tag_a: str = "zh", tag_b: str = "en") -> list[str]:
    lines = [f"| `{label}` | n | " + " | ".join(h for _, h, _ in METRICS) + " |",
             "|" + "---|" * (len(METRICS) + 2)]
    for tag, rep in ((tag_a, a), (tag_b, b), (f"Δ {tag_b}-{tag_a}", None)):
        if rep is None:
            cells = []
            for key, _, nd in METRICS:
                cells.append(delta(num(a, key), num(b, key), nd))
            lines.append(f"| **Δ** | {a.get('n_scored', '-')} | " + " | ".join(cells) + " |")
        else:
            cells = [fmt(num(rep, key), nd) for key, _, nd in METRICS]
            lines.append(f"| {tag} | {rep.get('n_scored', '-')} | " + " | ".join(cells) + " |")
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-report", required=True)
    ap.add_argument("--en-report", required=True)
    ap.add_argument("--zh-items")
    ap.add_argument("--en-items")
    ap.add_argument("--zh-preds")
    ap.add_argument("--en-preds")
    ap.add_argument("--label-a", default="zh", help="A 侧显示名（默认 zh）；比两个模型时用它改标注")
    ap.add_argument("--label-b", default="en", help="B 侧显示名（默认 en）")
    ap.add_argument("--title", default="中英对照（同一端点、同一条消息流）")
    args = ap.parse_args(argv[1:])
    A, B = args.label_a, args.label_b

    zr = json.loads(pathlib.Path(args.zh_report).read_text(encoding="utf-8"))
    er = json.loads(pathlib.Path(args.en_report).read_text(encoding="utf-8"))
    zsrc = next(iter(zr["strata"]["by_source"]))
    esrc = next(iter(er["strata"]["by_source"]))
    zb, eb = zr["strata"]["by_source"][zsrc], er["strata"]["by_source"][esrc]

    print(f"# {args.title}\n")
    print(f"- {A}: `{zr['items_file']}` sha256 `{zr['items_sha256'][:12]}…` "
          f"source `{zsrc}` (n={zb['n_items']})")
    print(f"- {B}: `{er['items_file']}` sha256 `{er['items_sha256'][:12]}…` "
          f"source `{esrc}` (n={eb['n_items']})")
    print("- 逐层聚合数字来自两份 `jev-eval` 报告（口径权威 = `crates/jev-eval/src/metrics.rs`）\n")

    print("## 总体（每份报告自己的 source 层）\n")
    print("\n".join(stratum_table(zb["overall"], eb["overall"], "overall", A, B)))
    print()

    print("## 分题型/category\n")
    cats = sorted(set(zb.get("by_category", {})) | set(eb.get("by_category", {})))
    for cat in cats:
        zc, ec = zb.get("by_category", {}).get(cat), eb.get("by_category", {}).get(cat)
        if not zc or not ec:
            print(f"⚠️ `{cat}` 只在一侧存在（{A}={bool(zc)} {B}={bool(ec)}）\n")
            continue
        print("\n".join(stratum_table(zc, ec, cat, A, B)))
        print()

    if args.zh_items and args.en_items and args.zh_preds and args.en_preds:
        # Pair by the numeric tail, NOT by the full id: the two sets are deliberately named
        # `zh-ev-0042` / `en-ev-0042`, so an exact-id intersection is empty. (First M3-2 run
        # reported "可配对条目：0" for 298 successful requests because of this.)
        zi = {numeric_key(k): v for k, v in read_jsonl(pathlib.Path(args.zh_items)).items()}
        ei = {numeric_key(k): v for k, v in read_jsonl(pathlib.Path(args.en_items)).items()}
        zp = {numeric_key(k): v for k, v in read_jsonl(pathlib.Path(args.zh_preds)).items()}
        ep = {numeric_key(k): v for k, v in read_jsonl(pathlib.Path(args.en_preds)).items()}

        def scorable(row: dict) -> bool:
            # A prediction row with no probability vector — `import` writes `{"id": …}` for a
            # question the backend never answered — cannot be compared here: there is no
            # distribution to read a slot off. jev-eval still counts those rows in n_missing
            # (coverage is never hidden); the paired test simply cannot use them, so they are
            # listed rather than silently dropped.
            return isinstance(row.get("probabilities"), dict) and bool(row["probabilities"])

        shared = set(zi) & set(ei) & set(zp) & set(ep)
        keys = sorted(k for k in shared if scorable(zp[k]) and scorable(ep[k]))
        unscorable = sorted(shared - set(keys))

        def quad(subset: list[str]) -> tuple[int, int, int, int]:
            br = zo = eo = bw = 0
            for k in subset:
                z_ok = argmax_first_max(zp[k]["probabilities"]) == gold_key(zi[k])
                e_ok = argmax_first_max(ep[k]["probabilities"]) == gold_key(ei[k])
                if z_ok and e_ok:
                    br += 1
                elif z_ok:
                    zo += 1
                elif e_ok:
                    eo += 1
                else:
                    bw += 1
            return br, zo, eo, bw

        both_right, zh_only, en_only, both_wrong = quad(keys)
        served = len(keys)
        p = mcnemar_exact(zh_only, en_only)
        print("## 配对检验（逐条配对，由 items+predictions 直接算）\n")
        print(f"- 可配对条目：{served}（两侧都有题 + 都拿到可评分的概率向量）")
        if unscorable:
            print(f"- 因**没有概率向量**退出配对：{len(unscorable)} 条 {unscorable}"
                  f"（后端没给出该槽位的 logprob；仍计入各报告的 n_missing）")
        print(f"- 两者都对：{both_right}｜只有 {A} 对：{zh_only}｜只有 {B} 对：{en_only}｜都对错：{both_wrong}")
        if served:
            acc_a = (both_right + zh_only) / served
            acc_b = (both_right + en_only) / served
            print(f"- **同一批 {served} 条上的正确率**：{A} {acc_a:.4f}｜{B} {acc_b:.4f}"
                  f"（差 {acc_b - acc_a:+.4f}）—— 与上面的 Δ 行口径不同：Δ 行是两份报告各自的 n")
        print(f"- McNemar 精确检验（双侧）：p = {p:.4f}"
              f"{'  ⚠️ 不一致对数 < 25，用精确二项而非常规卡方' if zh_only + en_only < 25 else ''}")
        diff = zh_only - en_only
        if p >= 0.05:
            print(f"- 判读：**两侧差异在 n={served} 上不显著**（不一致 {zh_only + en_only} 条，"
                  f"净差 {diff:+d}）⇒ 不足以支撑任何「{A} 更好/更差」的断言")
        else:
            better = A if diff > 0 else B
            print(f"- 判读：差异显著（净差 {diff:+d}，p={p:.4f}）⇒ 在 n={served} 上 {better} 正确率更高")
        print()

        print("### 逐分层配对（整体不显著时，差异可能藏在一层里）\n")
        print(f"| category | n_paired | {A} acc | {B} acc | 只有 {A} 对 | 只有 {B} 对 | McNemar p |")
        print("|---|---|---|---|---|---|---|")
        cats = sorted({zi[k]["category"] for k in keys})
        for cat in cats:
            sub = [k for k in keys if zi[k]["category"] == cat]
            br, zo, eo, bw = quad(sub)
            pa = mcnemar_exact(zo, eo)
            accs = "—" if not sub else f"{(br + zo) / len(sub):.4f}"
            acce = "—" if not sub else f"{(br + eo) / len(sub):.4f}"
            star = " ⭐" if pa < 0.05 else ""
            print(f"| `{cat}` | {len(sub)} | {accs} | {acce} | {zo} | {eo} | {pa:.4f}{star} |")
        print("\n（⭐ = p<0.05。多重比较提醒：4 个分层各测一次，未做校正，单层显著值应当作线索而非结论。）")
        # Failing strata (both sides wrong) are where the whole question lives: if a stratum is
        # hard in BOTH languages, that is a task-difficulty floor, not a language effect.
        print()
    else:
        print("（未给 items/predictions，跳过配对检验）\n")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
