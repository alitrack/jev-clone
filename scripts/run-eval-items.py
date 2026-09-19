#!/usr/bin/env python3
"""Drive the frozen item set through the real jev-server and collect its answers.

Every item has its own `state`, so nothing is shared between requests — this is the
honest "what does the model actually decide, and how sure is it" run, not a
throughput run. Answers are collected verbatim into a SystemOneResponse-shaped
body so `jev-eval import` sees exactly what the server said.

Usage: run-eval-items.py <items.jsonl> <base-url> <out-answers.json>
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

items_path, base, out_path = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]

items = []
for line in pathlib.Path(items_path).read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    items.append(json.loads(s))
print(f"loaded {len(items)} items from {items_path}")

answers, failures, usage = {}, [], {}
model = None
t0 = time.perf_counter()
for n, it in enumerate(items, 1):
    body = json.dumps(
        {"state": it["state"], "questions": {it["id"]: it["question"]}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}
    )
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:220]
            if attempt == 2:
                failures.append((it["id"], f"HTTP {e.code}: {detail}"))
                doc = None
        except Exception as e:  # timeout, connection reset, ...
            if attempt == 2:
                failures.append((it["id"], f"{type(e).__name__}: {e}"))
                doc = None
    if not doc:
        continue
    model = model or doc.get("model")
    got = doc.get("answers") or {}
    if it["id"] not in got:
        failures.append((it["id"], f"200 but no answer for {it['id']}: {list(got)}"))
        continue
    answers.update(got)
    for k, v in (doc.get("usage") or {}).items():
        if isinstance(v, (int, float)):
            usage[k] = usage.get(k, 0) + v
    if n % 10 == 0:
        print(f"  {n}/{len(items)} done ({time.perf_counter() - t0:.1f}s)")

out = {"model": model, "answers": answers, "usage": usage}
pathlib.Path(out_path).write_text(
    json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(f"\nanswered {len(answers)}/{len(items)} in {time.perf_counter() - t0:.1f}s -> {out_path}")
print(f"usage: {usage}")
if failures:
    print(f"\n{len(failures)} FAILURE(S):")
    for iid, why in failures:
        print(f"  - {iid}: {why}")
    sys.exit(1)
print("no failures")
