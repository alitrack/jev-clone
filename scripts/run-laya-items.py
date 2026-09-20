#!/usr/bin/env python3
"""M3-3: run the frozen Chinese item set through Laya and emit a scoreable answer doc.

Design constraints (from specs/M3.md):

* **No conversion logic of our own.** The harness already owns "answer -> number":
  `jev-eval import` reads a `SystemOneResponse` body (`{model, answers:{<id>:Answer}}`)
  and turns it into the prediction JSONL. So this script's only job is to *produce that
  body*. Metrics, stratification and the tamper gate stay in Rust.
* **No guessing about Laya's output shape.** The checkpoint's exact probability field
  name is not documented in the README's print examples, so we probe: the first result
  is dumped verbatim, candidate key names are tried in order, and if none matches the
  item's criteria keys the script fails loudly (and says which keys it *did* see)
  rather than inventing a mapping.
* **One forward pass per item, one question per item** — matching our item schema.

Usage (run with the venv that has laya installed):
  venv-laya/bin/python scripts/run-laya-items.py \
      --items eval/items/zh-evidence-v1.jsonl \
      --out-dir /mnt/d/wsl2/tmp/jev-m3/laya \
      --mode router            # router | multilingual | english | typed-decisions
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PROB_KEYS = ("probabilities", "distribution", "probs", "options", "scores", "logits")
CONF_KEYS = ("confidence", "conf", "probability")


def load_items(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(json.loads(line))
    return out


def questions_for(item: dict) -> dict:
    """Our item -> Laya's `questions` argument (same shape by design)."""
    q = item["question"]
    out: dict = {"type": q["type"], "instructions": q["instructions"]}
    if q["type"] == "choice":
        # criteria is {label: rubric|null}. Laya wants label -> description; a null
        # rubric becomes the label itself (keeps the prompt well-formed, adds nothing).
        out["criteria"] = {k: (v if v is not None else k) for k, v in q["criteria"].items()}
    elif q["type"] == "score":
        # criteria is an ordered list of level descriptions.
        out["criteria"] = [c if c is not None else f"level {i}" for i, c in enumerate(q["criteria"])]
    return out


def pick(d: dict, names) -> tuple[str | None, object]:
    for n in names:
        if n in d:
            return n, d[n]
    return None, None


def as_prob_map(raw: object) -> dict:
    if isinstance(raw, dict):
        return {str(a): float(b) for a, b in raw.items()}
    raise LookupError(f"expected a dict of label->p, got {type(raw).__name__}")


def as_prob_seq(raw: object) -> list:
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    if isinstance(raw, dict):
        return [float(v) for v in raw.values()]
    raise LookupError(f"expected a list/dict distribution, got {type(raw).__name__}")


def as_float(v: object) -> float:
    if isinstance(v, (int, float, str)):
        return float(v)
    raise LookupError(f"expected a number, got {type(v).__name__}")


def to_answer(item: dict, res: dict) -> dict:
    """Laya's per-question result -> contract `Answer` (contract.rs:76-100)."""
    q = item["question"]
    kind = q["type"]

    if kind == "choice":
        k, raw = pick(res, PROB_KEYS)
        if k is None:
            raise LookupError(f"no probability key in {sorted(res)}")
        probs = as_prob_map(raw)
        want = set(q["criteria"])
        if set(probs) != want:
            raise LookupError(f"probabilities keys {sorted(probs)} != criteria {sorted(want)}")
        label = str(res.get("choice") or max(probs, key=lambda x: probs[x]))
        _, conf = pick(res, CONF_KEYS)
        return {
            "type": "choice",
            "choice": label,
            "probabilities": probs,
            "confidence": as_float(conf) if conf is not None else max(probs.values()),
        }

    if kind == "score":
        k, raw = pick(res, PROB_KEYS)
        if k is None:
            raise LookupError(f"no distribution key in {sorted(res)}")
        seq = as_prob_seq(raw)
        levels = len(q["criteria"])
        if len(seq) != levels:
            raise LookupError(f"distribution length {len(seq)} != levels {levels}")
        probs = {str(i): p for i, p in enumerate(seq)}
        legend = {str(i): str(c) for i, c in enumerate(q["criteria"])}
        score = as_float(res["score"]) if "score" in res else sum(i * p for i, p in enumerate(seq))
        _, conf = pick(res, CONF_KEYS)
        return {
            "type": "score",
            "score": score,
            "legend": legend,
            "probabilities": probs,
            "confidence": as_float(conf) if conf is not None else max(probs.values()),
        }

    if kind == "noul":
        if "noul" not in res:
            raise LookupError(f"no noul field in {sorted(res)}")
        return {"type": "noul", "noul": as_float(res["noul"])}

    raise LookupError(f"unsupported question type {kind}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", default="router",
                    help="router | multilingual | english | typed-decisions")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv[1:])

    import laya  # noqa: PLC0415  (import here so --help works without the venv)

    items = load_items(pathlib.Path(args.items))
    if args.limit:
        items = items[: args.limit]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # preload=True: the README measures a 7.4 s median checkpoint REBUILD on CPU when
    # `max_loaded=1` and traffic alternates languages. Our set can mix scripts, so pay
    # once up front instead of per request.
    if args.mode == "router":
        engine = laya.Router(preload=True)
        model_name = "laya-router"
    elif args.mode in ("english", "root"):
        # The English checkpoint is the REPO ROOT — there is no `english/` subfolder
        # (verified against the HF file tree: root holds model.safetensors +
        #  encoder/ + tokenizer/; the other two live in multilingual/ and
        #  typed-decisions/). `subfolder="english"` raises FileNotFoundError.
        engine = laya.load("convaiinnovations/laya")
        model_name = "laya-english"
    else:
        engine = laya.load("convaiinnovations/laya", subfolder=args.mode)
        model_name = f"laya-{args.mode}"
    call = lambda st, qs: engine.predict(st, qs)  # noqa: E731

    answers: dict = {}
    routing: dict = {}
    problems: dict = {}
    first_dumped = False

    for i, item in enumerate(items, 1):
        qs = {"q": questions_for(item)}
        try:
            res = call(item["state"], qs)
        except Exception as exc:  # noqa: BLE001 - record and keep going; the run reports it
            problems[item["id"]] = f"predict failed: {type(exc).__name__}: {exc}"
            continue
        if not first_dumped:
            (out_dir / "laya-shape-probe.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            first_dumped = True
        if isinstance(res, dict) and "routing" in res:
            routing[item["id"]] = res["routing"]
        per_q = res.get("answers", res) if isinstance(res, dict) else {}
        qres = per_q.get("q", per_q)
        try:
            answers[item["id"]] = to_answer(item, dict(qres))
        except Exception as exc:  # noqa: BLE001
            problems[item["id"]] = f"extract failed: {type(exc).__name__}: {exc}"
        if i % 25 == 0:
            print(f"  {i}/{len(items)} …", flush=True)

    body = {
        "model": model_name,
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
    (out_dir / f"answers-{args.mode}.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if routing:
        (out_dir / f"routing-{args.mode}.json").write_text(
            json.dumps(routing, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if problems:
        (out_dir / f"problems-{args.mode}.json").write_text(
            json.dumps(problems, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\n{model_name}: {len(answers)}/{len(items)} 条产出可评分答案 -> {out_dir}/answers-{args.mode}.json")
    if problems:
        print(f"⛔ {len(problems)} 条失败（见 problems-{args.mode}.json）：")
        for k, v in list(problems.items())[:5]:
            print(f"   {k}: {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
