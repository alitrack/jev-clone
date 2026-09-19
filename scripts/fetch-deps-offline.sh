#!/bin/bash
# Fill the cargo registry cache for jev-clone.
#
# Cargo's TLS client is broken on this host (see .cargo/config.toml in the repo),
# so `cargo build --offline` is used and the missing `.crate` files are fetched
# with curl — which works — from the official CDN. Loop: try to build, parse the
# crate cargo says it is missing, fetch it, repeat.
#
# Checksums still come from Cargo.lock / the sparse index, so a wrong or partial
# download fails the build rather than silently poisoning the cache.
set -u
cd /mnt/d/wsl2/jev-clone || exit 1

CACHE="$HOME/.cargo/registry/cache/rsproxy.cn-e3de039b2554c837"
mkdir -p "$CACHE"

for i in $(seq 1 120); do
  out=$(cargo build --offline 2>&1)
  missing=$(printf '%s\n' "$out" | grep -oP 'failed to download `\K[^`]+' | head -1)

  if [ -z "$missing" ]; then
    echo "=== no further downloads needed after $((i-1)) fetches (iter $i) ==="
    printf '%s\n' "$out" | tail -25
    exit 0
  fi

  name=$(printf '%s' "$missing" | awk '{print $1}')
  ver=$(printf '%s' "$missing" | awk '{print $2}' | sed 's/^v//')
  dest="$CACHE/$name-$ver.crate"

  if [ -f "$dest" ]; then
    echo "[$i] $name-$ver already present but rejected -> removing and refetching"
    rm -f "$dest"
  fi

  # Reuse a copy from another registry cache dir when available (same bytes).
  other=$(find "$HOME/.cargo/registry/cache" -name "$name-$ver.crate" -not -path "*$(basename "$CACHE")*" 2>/dev/null | head -1)
  if [ -n "$other" ]; then
    cp -f "$other" "$dest"
    echo "[$i] $name-$ver <- copied from $(dirname "$other")"
    continue
  fi

  code=$(curl -sSL -o "$dest" -w "%{http_code}" --max-time 90 \
    "https://static.crates.io/crates/$name/$name-$ver.crate")
  size=$(stat -c '%s' "$dest" 2>/dev/null || echo 0)
  echo "[$i] $name-$ver <- curl $code ($size bytes)"
  if [ "$code" != "200" ]; then rm -f "$dest"; fi
done

echo "=== hit the 120-iteration cap without converging ==="
cargo build --offline 2>&1 | tail -15
exit 1
