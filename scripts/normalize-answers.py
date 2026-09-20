#!/usr/bin/env python3
"""Make a foreign model's answer body satisfy our frozen contract, **visibly**.

The contract (specs/M2.md §0 invariant 3) requires `Σp = 1 ± 1e-6` per question, and
`jev-eval run` refuses anything else. An external model can violate it for a mundane
reason: Laya rounds its probabilities to four decimals, so a three-way choice like
`0.3334 + 0.3333 + 0.3334` already sums to 1.0001.

Two things this script deliberately does NOT do:
* it does not touch the verbatim answer file — `answers-<mode>.json` stays exactly what
  the model emitted, so the raw numbers remain checkable;
* it does not silently rescale. Every item's before/after sum is written to a report,
  and items whose deviation exceeds `--max-shift` (default 1e-3, i.e. far beyond
  rounding) are a hard error — a real readout bug must not be papered over by
  normalization.

Usage:
  normalize-answers.py --in answers-multilingual.json --out answers-multilingual.norm.json \
                       --report normalization-multilingual.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def normalize_body(body: dict, max_shift: float) -> tuple[dict, dict]:
    answers = body.get("answers") or {}
    out_answers: dict = {}
    report: dict = {"n_items": len(answers), "max_abs_deviation": 0.0,
                    "n_normalized": 0, "items": {}, "max_shift_allowed": max_shift}

    for qid, ans in answers.items():
        if not isinstance(ans, dict):
            out_answers[qid] = ans
            continue
        kind = ans.get("type")
        probs = ans.get("probabilities")
        if kind not in ("choice", "score") or not isinstance(probs, dict) or not probs:
            out_answers[qid] = ans
            continue
        s = sum(float(v) for v in probs.values())
        dev = abs(s - 1.0)
        report["max_abs_deviation"] = max(report["max_abs_deviation"], dev)
        if dev > max_shift:
            raise SystemExit(
                f"⛔ {qid}: Σp = {s!r}（偏差 {dev:.2e}）超出容差 {max_shift:g} —— "
                f"这不像舍入误差，不允许用归一化掩盖读数缺陷"
            )
        if dev == 0.0:
            out_answers[qid] = ans
            continue
        new = dict(ans)
        new["probabilities"] = {k: float(v) / s for k, v in probs.items()}
        out_answers[qid] = new
        report["n_normalized"] += 1
        report["items"][qid] = {"sum_before": s, "deviation": dev}

    out_body = dict(body)
    out_body["answers"] = out_answers
    return out_body, report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--max-shift", type=float, default=1e-3)
    args = ap.parse_args(argv[1:])

    body = json.loads(pathlib.Path(args.src).read_text(encoding="utf-8"))
    out_body, report = normalize_body(body, args.max_shift)
    pathlib.Path(args.dst).write_text(
        json.dumps(out_body, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pathlib.Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"输入 {args.src}")
    print(f"  条目 {report['n_items']}｜被重标定 {report['n_normalized']}｜"
          f"最大偏差 {report['max_abs_deviation']:.2e}（容差 {args.max_shift:g}）")
    if report["items"]:
        worst = max(report["items"].items(), key=lambda kv: kv[1]["deviation"])
        print(f"  最差一条：{worst[0]} Σp={worst[1]['sum_before']!r}")
    print(f"  原件未改：{args.src}（逐字）｜契约合规版：{args.dst}｜偏差记录：{args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
