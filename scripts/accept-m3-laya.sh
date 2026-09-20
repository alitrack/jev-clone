#!/usr/bin/env bash
# M3-3 acceptance: score Laya's answers with the SAME frozen metrics + tamper gate the
# Chinese baseline went through. Nothing about the numbers is Laya-specific — that is
# the point: the item set is frozen, the metrics are frozen, only the model changes.
#
#   bash scripts/accept-m3-laya.sh multilingual
#   MODE=english bash scripts/accept-m3-laya.sh english      # ablation: English checkpoint on Chinese items
#
# Expects answers at $WORK/laya/answers-<mode>.json (written by scripts/run-laya-items.py).
set -euo pipefail
cd /mnt/d/wsl2/jev-clone

MODE="${1:-${MODE:-multilingual}}"
WORK="${JEV_WORK:-/mnt/d/wsl2/tmp/jev-m3}"
L="$WORK/laya"
ITEMS="${JEV_ITEMS:-eval/items/zh-evidence-v1.jsonl}"
ANSWERS="$L/answers-$MODE.json"
RUN_DIR="eval/runs"
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
PREDS="$RUN_DIR/$DATE-laya-$MODE-predictions.jsonl"
REPORT="$RUN_DIR/$DATE-laya-$MODE-report.json"
MD="$RUN_DIR/$DATE-laya-$MODE-report.md"

[ -s "$ANSWERS" ] || { echo "缺答案体：$ANSWERS（先跑 scripts/run-laya-items.py --mode $MODE）"; exit 1; }
mkdir -p "$RUN_DIR"

echo "== 1/6 题集与答案体自检 =="
python3 scripts/validate-items.py "$ITEMS"
echo "answers: $ANSWERS ($(wc -c < "$ANSWERS") bytes)"

echo
echo "== 2/6 归一化（Laya 四位小数舍入会撞契约不变量 Σp=1±1e-6；原件不动，偏差留痕）=="
NORM="$L/answers-$MODE.norm.json"
NORM_REPORT="$L/normalization-$MODE.json"
python3 scripts/normalize-answers.py --in "$ANSWERS" --out "$NORM" --report "$NORM_REPORT"

echo
echo "== 3/6 import（答案体 -> predictions，转换口径在 Rust 侧）=="
./target/debug/jev-eval import --items "$ITEMS" --answers "$NORM" --out "$PREDS"

echo
echo "== 4/6 run（manifest 哈希门：题集哈希不符就拒绝算）=="
./target/debug/jev-eval run --items "$ITEMS" --predictions "$PREDS" \
  --manifest eval/items/manifest.json --out "$REPORT" --md "$MD"

echo
echo "== 5/6 verify（从原始证据重算每一个数）=="
./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" --report "$REPORT"

echo
echo "== 6/6 tamper gate（改写过的报告必须重算不干净）=="
TAMPERED="$WORK/$DATE-laya-$MODE-report-tampered.json"
python3 scripts/tamper-report.py "$REPORT" "$TAMPERED"
if ./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" \
     --report "$TAMPERED" > "$WORK/$DATE-laya-$MODE-tamper.log" 2>&1; then
  echo "TAMPER GATE FAILED: 一个自报容差的改写报告竟然校验通过"; head -20 "$WORK/$DATE-laya-$MODE-tamper.log"; exit 1
fi
echo "tamper gate passed"

echo
echo "== 摘要（分层，禁止合并总分）=="
python3 scripts/summarize-report.py "$REPORT"
echo
echo "产物：$REPORT  $MD  $PREDS"
