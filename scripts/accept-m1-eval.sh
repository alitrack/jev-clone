#!/usr/bin/env bash
# End-to-end M1-2 acceptance: frozen item set -> real server -> predictions ->
# report -> verify. Everything lands in eval/runs/ so the numbers are auditable.
set -euo pipefail
cd /mnt/d/wsl2/jev-clone

PORT="${JEV_PORT:-18080}"
BASE="http://127.0.0.1:$PORT"
# The frozen set under test. Default is v1 (150 items, reached 2026-09-19); v0
# stays on disk with its own manifest hash because the M1 reports were computed
# against it. Override to re-run against either: JEV_ITEMS=eval/items/zh-evidence-v0.jsonl
ITEMS="${JEV_ITEMS:-eval/items/zh-evidence-v1.jsonl}"
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="eval/runs"
ANSWERS="$RUN_DIR/$DATE-answers.json"
PREDS="$RUN_DIR/$DATE-predictions.jsonl"
REPORT="$RUN_DIR/$DATE-report.json"
MD="$RUN_DIR/$DATE-report.md"
WORK="/mnt/d/wsl2/tmp/jev-m0"
mkdir -p "$RUN_DIR" "$WORK"

echo "== 1/6 starting jev-server on :$PORT (backend $( [ -n "${JEV_BASE_URL:-}" ] && echo "$JEV_BASE_URL" || echo http://10.10.10.115:8014/v1 ))"
JEV_BASE_URL="${JEV_BASE_URL:-http://10.10.10.115:8014/v1}" \
JEV_MODEL="${JEV_MODEL:-qwen3.8-27b}" \
JEV_LISTEN="127.0.0.1:$PORT" \
  setsid ./target/debug/jev-server > "$WORK/eval-server.log" 2>&1 &
SRV=$!
trap 'kill '"$SRV"' 2>/dev/null || true' EXIT

# The readiness probe must hit a route the server actually serves: it is
# `/healthz` (`api.rs`), and polling `/health` (which 404s) can never break the
# loop — the script would just sleep through its whole budget and then act as if
# the server were up. Fail loudly instead.
READY=0
for i in $(seq 1 40); do
  if curl -s -m 3 --noproxy '*' "http://127.0.0.1:$PORT/healthz" | grep -q '"ok"'; then READY=1; break; fi
  sleep 0.5
done
if [ "$READY" -ne 1 ]; then
  echo "server never became ready on $BASE/healthz; last log lines:"
  tail -20 "$WORK/eval-server.log"
  exit 1
fi
curl -s -m 3 --noproxy '*' "http://127.0.0.1:$PORT/healthz" | head -c 200; echo

echo "== 2/6 driving $ITEMS through /v1/systemone"
python3 scripts/run-eval-items.py "$ITEMS" "$BASE" "$ANSWERS"

echo "== 3/6 import answers -> predictions"
./target/debug/jev-eval import --items "$ITEMS" --answers "$ANSWERS" --out "$PREDS"

echo "== 4/6 report (hash-gated by the manifest)"
./target/debug/jev-eval run --items "$ITEMS" --predictions "$PREDS" \
  --manifest eval/items/manifest.json --out "$REPORT" --md "$MD"

echo "== 5/6 verify (recompute every number from raw evidence)"
./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" --report "$REPORT"

echo "== 6/6 tamper gate (the audit threshold is ours, not the artifact's)"
TAMPERED="$WORK/$DATE-report-tampered.json"
python3 scripts/tamper-report.py "$REPORT" "$TAMPERED"
if ./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" \
     --report "$TAMPERED" > "$WORK/tamper.log" 2>&1; then
  echo "TAMPER GATE FAILED: a rewritten report declaring its own tolerance verified clean"
  head -20 "$WORK/tamper.log"
  exit 1
fi
grep -m3 -E "mismatch|verify_tolerance|ece" "$WORK/tamper.log" || true
echo "tamper gate passed: verify refused the rewritten report"

echo
echo "== summary"
python3 scripts/summarize-report.py "$REPORT"
echo "report: $REPORT"
