#!/usr/bin/env bash
# M3-2 acceptance: the zh/en control pair, driven interleaved through the real
# server, reported per language (never merged), verified, tamper-gated, and
# finally compared with the paired test.
#
#   JEV_ZH=eval/items/zh-evidence-v1.jsonl \
#   JEV_EN=eval/items/en-evidence-v1.jsonl \
#   bash scripts/accept-m3-paired.sh
#
# Same skeleton as scripts/accept-m1-eval.sh on purpose: one set of commands the
# whole project already knows how to read.
set -euo pipefail
cd /mnt/d/wsl2/jev-clone

PORT="${JEV_PORT:-18081}"
BASE="http://127.0.0.1:$PORT"
ZH="${JEV_ZH:-eval/items/zh-evidence-v1.jsonl}"
EN="${JEV_EN:-eval/items/en-evidence-v1.jsonl}"
RUN_DIR="eval/runs"
WORK="/mnt/d/wsl2/tmp/jev-m3"
mkdir -p "$RUN_DIR" "$WORK"

echo "== 0/7 先过闸门：题集冻结哈希 + 中英配对/槽位一致 =="
python3 scripts/validate-items.py "$EN"
# M4 起英文集是 MT 重译的，specs/M4.md §S3 允许「整条剔除」⇒ 英文集可能是中文集的**子集**。
# 传 JEV_ALLOW_MISSING=<translate-mt.py 的 report.json> 时，闸门改判「缺的 id 必须与报告
# 申报的剔除清单逐条对齐」（多报少报都算错）；不传则与 M3 一样，缺一条即失败。
PARITY_ARGS=()
if [ -n "${JEV_ALLOW_MISSING:-}" ]; then
  PARITY_ARGS=(--allow-missing "$JEV_ALLOW_MISSING")
  echo "   JEV_ALLOW_MISSING=$JEV_ALLOW_MISSING —— 英文集允许按 §S3 剔除，缺 id 须与报告对齐"
fi
python3 scripts/check-en-parity.py "$ZH" "$EN" ${PARITY_ARGS[@]+"${PARITY_ARGS[@]}"}

echo "== 1/7 启动 jev-server on :$PORT (backend ${JEV_BASE_URL:-http://127.0.0.1:8014/v1}) =="
JEV_BASE_URL="${JEV_BASE_URL:-http://127.0.0.1:8014/v1}" \
JEV_MODEL="${JEV_MODEL:-qwen3.8-27b}" \
JEV_LISTEN="127.0.0.1:$PORT" \
  setsid ./target/debug/jev-server > "$WORK/paired-server.log" 2>&1 &
SRV=$!
trap 'kill '"$SRV"' 2>/dev/null || true' EXIT

READY=0
for _ in $(seq 1 40); do
  if curl -s -m 3 --noproxy '*' "http://127.0.0.1:$PORT/healthz" | grep -q '"ok"'; then READY=1; break; fi
  sleep 0.5
done
if [ "$READY" -ne 1 ]; then
  echo "server never became ready; last log lines:"; tail -20 "$WORK/paired-server.log"; exit 1
fi

echo "== 2/7 交错驱动两集（zh 0001, en 0001, zh 0002, …）=="
if [ "${JEV_SKIP_DRIVE:-0}" = "1" ]; then
  echo "   JEV_SKIP_DRIVE=1 —— 跳过驱动，复用最近一次 run 的 answers（用于只补跑 3–7 步）"
else
  # A per-item request failure must NOT abort the acceptance run. eval/README.md §4.1 is
  # explicit that coverage may never be hidden — it is to be counted and listed, with the
  # missing rows excluded from NLL/Brier/ECE, not used as a reason to throw away the 298
  # answers that did come back. (First M3-2 run, 2026-09-20: 2 en items got HTTP 422 from the
  # backend readout, the driver exited non-zero, `set -e` killed the script before import, and
  # the whole 400 s of driving produced no report at all.)
  set +e
  python3 scripts/run-eval-paired.py "$ZH" "$EN" "$BASE" "$RUN_DIR"
  DRIVE_RC=$?
  set -e
  if [ "$DRIVE_RC" -ne 0 ]; then
    echo "⚠️  驱动器返回 $DRIVE_RC（有请求失败）—— 继续，覆盖率由 import/report 单列计数并逐条列名"
  fi
fi
META="$(ls -t "$RUN_DIR"/*-run-meta.json | head -1)"
DATE="$(basename "$META" | sed 's/-run-meta.json//')"
echo "run date: $DATE"

for LANG in zh en; do
  ITEMS="$ZH"; [ "$LANG" = "en" ] && ITEMS="$EN"
  ANSWERS="$RUN_DIR/$DATE-answers-$LANG.json"
  PREDS="$RUN_DIR/$DATE-$LANG-predictions.jsonl"
  REPORT="$RUN_DIR/$DATE-$LANG-report.json"
  MD="$RUN_DIR/$DATE-$LANG-report.md"

  echo "== 3/7 [$LANG] import =="
  ./target/debug/jev-eval import --items "$ITEMS" --answers "$ANSWERS" --out "$PREDS"

  echo "== 4/7 [$LANG] report（manifest 哈希门）=="
  ./target/debug/jev-eval run --items "$ITEMS" --predictions "$PREDS" \
    --manifest eval/items/manifest.json --out "$REPORT" --md "$MD"

  echo "== 5/7 [$LANG] verify（从原始证据重算每一个数）=="
  ./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" --report "$REPORT"

  echo "== 6/7 [$LANG] tamper gate =="
  TAMPERED="$WORK/$DATE-$LANG-report-tampered.json"
  python3 scripts/tamper-report.py "$REPORT" "$TAMPERED"
  if ./target/debug/jev-eval verify --items "$ITEMS" --predictions "$PREDS" \
       --report "$TAMPERED" > "$WORK/$DATE-$LANG-tamper.log" 2>&1; then
    echo "TAMPER GATE FAILED [$LANG]: a rewritten report declaring its own tolerance verified clean"
    head -20 "$WORK/$DATE-$LANG-tamper.log"; exit 1
  fi
  echo "tamper gate passed [$LANG]"
done

echo "== 7/7 中文/英文分开报（禁止合并总分）+ 配对检验 =="
echo "--- zh ---"; python3 scripts/summarize-report.py "$RUN_DIR/$DATE-zh-report.json"
echo "--- en ---"; python3 scripts/summarize-report.py "$RUN_DIR/$DATE-en-report.json"

PAIR_MD="$RUN_DIR/$DATE-pair-diff.md"
python3 scripts/pair-diff.py \
  --zh-report "$RUN_DIR/$DATE-zh-report.json" \
  --en-report "$RUN_DIR/$DATE-en-report.json" \
  --zh-items "$ZH" --en-items "$EN" \
  --zh-preds "$RUN_DIR/$DATE-zh-predictions.jsonl" \
  --en-preds "$RUN_DIR/$DATE-en-predictions.jsonl" | tee "$PAIR_MD"

echo
echo "产物：$RUN_DIR/$DATE-{zh,en}-report.{json,md}  $RUN_DIR/$DATE-{zh,en}-predictions.jsonl  $PAIR_MD"
