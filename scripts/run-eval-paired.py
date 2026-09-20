#!/usr/bin/env python3
"""Drive a zh/en item pair through jev-server **interleaved**, one item at a time.

Why interleaved and not "150 Chinese then 150 English": both sets run on the same
shared SGLang endpoint (115:8014) and the same GPU, so any drift — another job
arriving, a clock ramp, a thermal step — lands entirely on whichever set ran
second.  That is exactly how M1's first gate produced a 0.96x reading that turned
out to be an artifact of measurement order (see specs/M2.md, "cold-clock ramp").
Alternating the two sets spreads any drift across both, and the per-item latency
log written here makes the drift visible instead of assumed.

Outputs (into <out-dir>, all timestamped so runs never overwrite each other):
  * `<date>-answers-zh.json`  — SystemOneResponse-shaped, exactly what `jev-eval import` wants
  * `<date>-answers-en.json`
  * `<date>-run-meta.json`    — the interleave order, per-item latency, model, usage

Usage: run-eval-paired.py <zh.jsonl> <en.jsonl> <base-url> <out-dir>
Exit code is non-zero if ANY item failed after two attempts — a set that is 149/150
is not a result, it is a bug (specs/M3.md §4).
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request


def read_items(path: str) -> list[dict]:
    out = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(json.loads(s))
    return out


def drive(base: str, item: dict, attempts: int = 2) -> tuple[dict | None, float, str | None]:
    body = json.dumps(
        {"state": item["state"], "questions": {item["id"]: item["question"]}},
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        base + "/v1/systemone", data=body, headers={"Content-Type": "application/json"}
    )
    why = None
    dt = 0.0
    for attempt in range(1, attempts + 1):
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                doc = json.loads(resp.read().decode("utf-8"))
            dt = (time.perf_counter() - t0) * 1000.0
            got = doc.get("answers") or {}
            if item["id"] not in got:
                why = f"200 but no answer for {item['id']}: keys={list(got)}"
                continue
            return doc, dt, None
        except urllib.error.HTTPError as exc:
            dt = (time.perf_counter() - t0) * 1000.0
            why = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:220]}"
        except Exception as exc:  # noqa: BLE001 - timeout, reset, ...
            dt = (time.perf_counter() - t0) * 1000.0
            why = f"{type(exc).__name__}: {exc}"
    return None, dt, why


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(__doc__)
        return 2
    zh_path, en_path, base, out_dir = argv[1], argv[2], argv[3].rstrip("/"), pathlib.Path(argv[4])
    zh, en = read_items(zh_path), read_items(en_path)

    def key(item: dict) -> str:
        return item["id"].split("-")[-1]

    zh_by, en_by = {key(i): i for i in zh}, {key(i): i for i in en}
    if set(zh_by) != set(en_by):
        print(f"⛔ 两集 id 不成对：zh-only={sorted(set(zh_by) - set(en_by))[:5]} "
              f"en-only={sorted(set(en_by) - set(zh_by))[:5]}")
        return 2
    if len(zh_by) != len(zh) or len(en_by) != len(en):
        print("⛔ 集内 id 有重复")
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    # Interleave: zh 0001, en 0001, zh 0002, en 0002, ...
    plan = []
    for k in sorted(zh_by):
        plan.append(("zh", zh_by[k]))
        plan.append(("en", en_by[k]))

    answers = {"zh": {}, "en": {}}
    usage = {"zh": {}, "en": {}}
    order, failures = [], []
    model = None
    t_start = time.perf_counter()
    for n, (lang, item) in enumerate(plan, 1):
        doc, dt, why = drive(base, item)
        order.append({"n": n, "lang": lang, "id": item["id"], "latency_ms": round(dt, 1),
                      "ok": doc is not None})
        if doc is None:
            failures.append((item["id"], why))
            continue
        model = model or doc.get("model")
        answers[lang].update(doc.get("answers") or {})
        for k, v in (doc.get("usage") or {}).items():
            if isinstance(v, (int, float)):
                usage[lang][k] = usage[lang].get(k, 0) + v
        if n % 20 == 0:
            print(f"  {n}/{len(plan)} done ({time.perf_counter() - t_start:.1f}s)", flush=True)

    elapsed = time.perf_counter() - t_start
    date = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for lang in ("zh", "en"):
        doc = {"model": model, "answers": answers[lang], "usage": usage[lang]}
        p = out_dir / f"{date}-answers-{lang}.json"
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{lang}: {len(answers[lang])}/{len(plan) // 2} -> {p}")

    # Latency drift evidence: compare the two halves' medians per language.
    med = {}
    for lang in ("zh", "en"):
        xs = sorted(r["latency_ms"] for r in order if r["lang"] == lang and r["ok"])
        med[lang] = xs[len(xs) // 2] if xs else None
    meta = {"date": date, "model": model, "base_url": base, "elapsed_s": round(elapsed, 1),
            "order": order, "median_latency_ms": med,
            "zh_items": zh_path, "en_items": en_path}
    (out_dir / f"{date}-run-meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"\n{len(plan) - len(failures)}/{len(plan)} answered in {elapsed:.1f}s "
          f"(median latency zh={med['zh']}ms en={med['en']}ms)")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for iid, why in failures[:20]:
            print(f"  - {iid}: {why}")
        return 1
    print("no failures")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
