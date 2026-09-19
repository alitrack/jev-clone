#!/usr/bin/env bash
# End-to-end M1-2 acceptance: frozen item set -> real server -> predictions ->
# report -> verify. Everything lands in eval/runs/ so the numbers are auditable.
set -euo pipefail
cd /mnt/d/wsl2/jev-clone

PORT="${JEV_PORT:-18080}"
BASE="http://127.0.0.1:$PORT"
ITEMS="eval/items/zh-evidence-v0.jsonl"
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="eval/runs"
ANSWERS="$RUN_DIR/$DATE-answers.json"
PREDS="$RUN_DIR/$DATE-predictions.jsonl"
REPORT="$RUN_DIR/$DATE-report.json"
MD="$RUN_DIR/$DATE-report.md"
WORK="/mnt/d/wsl2/tmp/jev-m0"
mkdir -p "$RUN_DIR"

echo "== 1/5 starting jev-server on :$PORT (backend $( [ -n "${JEV_BASE_URL:-}" ] && echo "$JEV_BASE_URL" || echo http://10.10.10.115:8014/v1 ))"
JEV_BASE_URL="${JEV_BASE_URL:-http://10.10.10.115:8014/v1}" \
JEV_MODEL="${JEV_MODEL:-qwen3.8-27b}" \
JEV_LISTEN="127.0.0.1:$PORT" \
  setsid ./target/debug/jev-server > "$WORK/eval-server.log" 2>&1 &
SRV=$!
trap 'kill '"$SRV"' 2>/dev/null || true' EXIT

for i in $(seq 1 40); do
  if curl -s --noproxy '*' "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then break; fi
  sleep 0.5
done
curl -s --noproxy '*' "http://127.0.0.1:$PORT/health" | head -c 200; echo

echo "== 2/5 driving $ITEMS through /v1/systemone"
python3 scripts/run-eval-items.py "$ITEMS" "$BASE" "$ANSWERS"

echo "== 3/5 import answers -> predictions"
./target/debug/jev-eval import --items "$ITEMS" --answers "$ANSWERS" --out "$PREDS"

echo "== 4/5 report (hash-gated by the manifest)"
./target/debug/jev-eval run --items "$ITEMS" --predictions "$PREDS" \
  --manifest eval/items/manifest.json --out "$REPORT" --md "$MD"

echo "== 5/5 verify (recompute every number from raw evidence)"
./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" --report "$REPORT"

echo
echo "== summary"
python3 scripts/summarize-report.py "$REPORT"
echo "report: $REPORT"
