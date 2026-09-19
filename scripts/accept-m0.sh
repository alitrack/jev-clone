#!/bin/bash
# M0 acceptance: build + unit tests (offline) + live end-to-end contract checks.
#
#   usage: scripts/accept-m0.sh [--no-live]
#
# Offline cargo only: cargo's TLS client cannot reach any crates.io mirror from
# this host (see .cargo/config.toml). Missing deps: scripts/fetch-deps-offline.sh
set -u
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
LIVE=1
[ "${1:-}" = "--no-live" ] && LIVE=0

echo "########## 1/3 build + unit tests (offline)"
cargo build --offline --workspace 2>&1 | tail -5
cargo test --offline --workspace 2>&1 | tail -25
TEST_RC=${PIPESTATUS[0]}
echo "unit tests exit=$TEST_RC"

[ "$LIVE" = "0" ] && exit "$TEST_RC"

echo
echo "########## 2/3 start server against the live endpoint"
JEV_BASE_URL="${JEV_BASE_URL:-http://10.10.10.115:8014/v1}" \
JEV_MODEL="${JEV_MODEL:-qwen3.8-27b}" \
JEV_LISTEN=127.0.0.1:8080 \
  ./target/debug/jev-server > /mnt/d/wsl2/tmp/jev-m0/server.log 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID log=/mnt/d/wsl2/tmp/jev-m0/server.log"

cleanup() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT

for i in $(seq 1 30); do
  if curl -s --noproxy '*' -m 3 http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
    echo "server ready after ${i}s"
    break
  fi
  sleep 1
done

if ! curl -s --noproxy '*' -m 3 http://127.0.0.1:8080/healthz > /dev/null 2>&1; then
  echo "!! server never became ready — last log lines:"
  tail -20 /mnt/d/wsl2/tmp/jev-m0/server.log
  exit 1
fi

echo
echo "########## 3/3 contract acceptance (live model)"
python3 "$REPO/scripts/accept-m0.py" --url http://127.0.0.1:8080
ACCEPT_RC=$?

echo
echo "server log tail:"
tail -12 /mnt/d/wsl2/tmp/jev-m0/server.log

echo
echo "========== unit_tests=$TEST_RC acceptance=$ACCEPT_RC"
[ "$TEST_RC" = "0" ] && [ "$ACCEPT_RC" = "0" ]
