#!/usr/bin/env python3
"""置信门控曲线：在给定阈值下只接受模型自己"有把握"的答案，能换到多少正确率？

这是 specs/M3.md 的 Q2（"中文上置信度还能不能当门控"）的直接读数，也是决策相关的：
门控的价值不在 ECE 本身，而在 *拒绝掉一部分之后剩下那部分的正确率*。

口径声明（避免与 jev-eval 抢权威）：
* 逐层聚合的 acc/ECE/NLL/Brier 一律以 `jev-eval` 报告为准（`crates/jev-eval/src/metrics.rs`）；
* 本脚本只算 `metrics.rs` 没有的那两个量：**覆盖率**与**覆盖内正确率**，
  以及厂商自己报告里用的 "accuracy at 50% coverage" 同类指标。
* 置信度取该题概率向量的最大值（noul 取 max(p, 1-p)），与 ECE 的置信定义一致。

Usage: gate-curve.py --items <items.jsonl> --preds <predictions.jsonl> [--label 名字]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def gold_key(item: dict) -> str:
    q = item["question"]
    if q["type"] == "score":
        return str(item["gold"])
    if q["type"] == "noul":
        return "true" if item["gold"] is True else "false"
    return item["gold"]


def argmax_first_max(probs: dict) -> str | None:
    best, best_v = None, None
    for k, v in probs.items():
        if best_v is None or v > best_v:
            best, best_v = k, v
    return best


def read_jsonl(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            row = json.loads(line)
            out[row["id"]] = row
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--label", default="model")
    args = ap.parse_args(argv[1:])

    items, preds = read_jsonl(pathlib.Path(args.items)), read_jsonl(pathlib.Path(args.preds))
    rows = []
    skipped = []
    for iid in sorted(set(items) & set(preds)):
        probs = preds[iid].get("probabilities")
        if not isinstance(probs, dict) or not probs:
            # `import` writes `{"id": …}` when the backend never returned a distribution for
            # that question — no confidence to threshold on. Count it, name it, move on.
            skipped.append(iid)
            continue
        conf = max(probs.values())
        rows.append((conf, argmax_first_max(probs) == gold_key(items[iid])))

    n = len(rows)
    if not n:
        print("没有可配对的条目")
        return 1
    acc_all = sum(1 for _, ok in rows if ok) / n
    rows.sort(key=lambda r: -r[0])

    print(f"# 置信门控曲线 — {args.label}（n={n}）\n")
    if skipped:
        print(f"⚠️ {len(skipped)} 条无概率向量、未纳入：{skipped[:8]}{' …' if len(skipped) > 8 else ''}\n")
    print(f"全量正确率（不门控）：**{acc_all:.4f}**\n")
    print("| 阈值 | 覆盖率 | 覆盖内正确率 | 覆盖数 | 被拒数与其中错的比例 |")
    print("|---|---|---|---|---|")
    for t in THRESHOLDS:
        kept = [(c, ok) for c, ok in rows if c >= t]
        if not kept:
            print(f"| ≥{t:.2f} | 0.0% | - | 0 | - |")
            continue
        cov = len(kept) / n
        acc = sum(1 for _, ok in kept if ok) / len(kept)
        rejected = [(c, ok) for c, ok in rows if c < t]
        r_wrong = sum(1 for _, ok in rejected if not ok)
        r_str = f"{len(rejected)} 条，其中错 {r_wrong} 条（{r_wrong/len(rejected):.1%}）" if rejected else "-"
        print(f"| ≥{t:.2f} | {cov:.1%} | **{acc:.4f}** | {len(kept)} | {r_str} |")

    # accuracy at 50% coverage (sorted by confidence, take the most confident half)
    half = max(1, n // 2)
    acc_half = sum(1 for _, ok in rows[:half] if ok) / half
    print(f"\n50% 覆盖率下的正确率（取最有把握的一半）：**{acc_half:.4f}**")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
