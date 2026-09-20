#!/usr/bin/env bash
# Score the degenerate baselines through the same frozen metrics + verify path as any model.
# This is the bar `eval/README.md` §5 says must exist before any ECE/Brier is read.
#
#   bash scripts/score-degenerate-baselines.sh [items.jsonl]
set -euo pipefail
cd /mnt/d/wsl2/jev-clone

ITEMS="${1:-eval/items/zh-evidence-v1.jsonl}"
RUN_DIR="eval/runs"
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(sha256sum "$ITEMS" | cut -d' ' -f1)"
# Tag the prediction files with the set, not just the strategy: running this for the English
# companion set used to overwrite the Chinese baselines' prediction files (same names).
SET="$(basename "$ITEMS" .jsonl)"

echo "== 生成三条退化策略的预测（无模型、无网络；题集 $SET）=="
python3 scripts/degenerate-baselines.py --items "$ITEMS" --out-dir "$RUN_DIR" --tag "$SET"
echo

for NAME in slot-a written-first slot-prior; do
  P="$RUN_DIR/deg-$SET-$NAME-predictions.jsonl"
  R="$RUN_DIR/$DATE-deg-$SET-$NAME-report.json"
  M="$RUN_DIR/$DATE-deg-$SET-$NAME-report.md"
  echo "== [$NAME] run + verify =="
  ./target/debug/jev-eval run --items "$ITEMS" --predictions "$P" \
    --expect-sha256 "$SHA" --out "$R" --md "$M" > /dev/null
  ./target/debug/jev-eval verify --items "$ITEMS" --predictions "$P" --report "$R" | tail -2
  python3 scripts/summarize-report.py "$R" | grep -E "^(overall|items)" || true
  echo
done

echo "产物：$RUN_DIR/$DATE-deg-*-report.{json,md}  $RUN_DIR/deg-*-predictions.jsonl"
