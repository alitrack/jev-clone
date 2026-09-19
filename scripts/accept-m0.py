#!/usr/bin/env python3
"""Contract acceptance checks for a running jev-server (M0).

Independent of the implementation: this only speaks HTTP and asserts the public
contract's observable properties. Run it against a live server:

    python3 scripts/accept-m0.py --url http://127.0.0.1:8080

Exit code 0 = all checks passed, 1 = at least one failure.
No third-party imports (urllib only) so it runs anywhere.
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.request

FAILS: list[str] = []
PASSES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    (PASSES if ok else FAILS).append(f"{label}{(' :: ' + detail) if detail else ''}")
    print(("  OK   " if ok else "  FAIL ") + label + ((" :: " + detail) if detail else ""))


def post(url: str, payload: dict, timeout: int = 300) -> tuple[int, dict]:
    """POST JSON through a deliberately proxy-free opener (loopback HTTP must not
    go through any configured proxy)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, {"raw": body[:300]}


STATE = (
    "Order A-104: the customer was charged twice for a monthly subscription. "
    "The customer writes: 'I see two identical charges on my card this month. "
    "Please refund the duplicate.' Support replied the same day and started a refund."
)


def q_choice():
    return {
        "type": "choice",
        "instructions": "Does the customer request a refund?",
        "criteria": {"yes": "asks for money back", "no": "does not ask for money back"},
    }


def q_score():
    return {
        "type": "score",
        "instructions": "How severe is this issue?",
        "criteria": [
            "low: cosmetic confusion",
            "medium: one duplicate charge",
            "high: repeated billing failure",
        ],
    }


def q_noul():
    return {
        "type": "noul",
        "instructions": "Did support reply the same day?",
        "criteria": {"true": "replied the same day", "false": "slower than that"},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    base = args.url.rstrip("/")

    print(f"== jev-server contract acceptance @ {base}")

    # 1) health
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            base + "/healthz", timeout=10
        ) as r:
            health = json.loads(r.read().decode())
        check(health.get("status") == "ok", "healthz responds ok")
    except Exception as e:  # noqa: BLE001
        check(False, "healthz responds ok", str(e))
        return report()

    # 2) the three primitives in ONE request (questions are independent)
    status, body = post(
        base + "/v1/systemone",
        {"state": STATE, "questions": {"refund": q_choice(), "severity": q_score(), "replied": q_noul()}},
    )
    check(status == 200, "POST /v1/systemone returns 200", f"status={status} body={str(body)[:200]}")
    if status != 200:
        return report()

    check(isinstance(body.get("model"), str) and body["model"], "response carries `model`")
    usage = body.get("usage") or {}
    check(
        isinstance(usage.get("input_tokens"), int) and usage["input_tokens"] > 0,
        "usage.input_tokens present and > 0",
        str(usage),
    )
    check(usage.get("output_tokens") == 3, "usage.output_tokens == question count", str(usage))

    answers = body.get("answers") or {}
    check(set(answers) == {"refund", "severity", "replied"}, "answers keyed by every question id", str(list(answers)))

    # 3) choice
    a = answers.get("refund") or {}
    check(a.get("type") == "choice", "choice answer has type=choice", str(a.get("type")))
    probs = a.get("probabilities") or {}
    check(set(probs) == {"yes", "no"}, "choice probabilities cover the criteria labels", str(list(probs)))
    if probs:
        total = sum(probs.values())
        check(abs(total - 1.0) <= 1e-6, "choice probabilities sum to 1", f"sum={total!r}")
        best = max(probs, key=lambda k: probs[k])
        check(a.get("choice") == best, "choice == argmax(probabilities)", f"choice={a.get('choice')} best={best}")
        check(0.0 <= float(a.get("confidence", -1)) <= 1.0, "choice confidence in [0,1]", str(a.get("confidence")))

    # 4) score
    s = answers.get("severity") or {}
    check(s.get("type") == "score", "score answer has type=score", str(s.get("type")))
    sprobs = s.get("probabilities") or {}
    check(set(sprobs) == {"0", "1", "2"}, "score probabilities keyed by level index", str(list(sprobs)))
    if sprobs:
        stotal = sum(sprobs.values())
        check(abs(stotal - 1.0) <= 1e-6, "score probabilities sum to 1", f"sum={stotal!r}")
        expect = sum(i * float(sprobs[str(i)]) for i in range(len(sprobs)))
        check(
            abs(float(s.get("score", -99)) - expect) <= 1e-3,
            "score == sum(i * p_i)",
            f"score={s.get('score')} expected={round(expect, 6)}",
        )
        legend = s.get("legend") or {}
        check(set(legend) == {"0", "1", "2"}, "score legend covers every level", str(list(legend)))

    # 5) noul
    n = answers.get("replied") or {}
    check(n.get("type") == "noul", "noul answer has type=noul", str(n.get("type")))
    noul = n.get("noul")
    check(isinstance(noul, (int, float)) and 0.0 <= float(noul) <= 1.0, "noul in [0,1]", str(noul))
    check("confidence" not in n, "noul answer carries no confidence (per contract)")

    # 6) contract violations must be 422, never a silently wrong answer
    status, body = post(base + "/v1/systemone", {"state": STATE, "questions": {}})
    check(status == 422, "empty questions rejected with 422", f"status={status}")

    too_few = {"type": "choice", "instructions": "pick", "criteria": {"only": "one"}}
    status, body = post(base + "/v1/systemone", {"state": STATE, "questions": {"q": too_few}})
    check(status == 422, "1-option choice rejected with 422", f"status={status}")

    many = {"type": "choice", "instructions": "pick", "criteria": {f"o{i}": None for i in range(300)}}
    status, body = post(base + "/v1/systemone", {"state": STATE, "questions": {"q": many}})
    check(status == 422, "256-option choice rejected with 422", f"status={status}")

    few_levels = {"type": "score", "instructions": "rate", "criteria": ["only one"]}
    status, body = post(base + "/v1/systemone", {"state": STATE, "questions": {"q": few_levels}})
    check(status == 422, "1-level score rejected with 422", f"status={status}")

    # 7) determinism: same request twice must agree (temperature 0 readout)
    _, again = post(
        base + "/v1/systemone",
        {"state": STATE, "questions": {"refund": q_choice()}},
    )
    again_choice = ((again.get("answers") or {}).get("refund") or {}).get("choice")
    check(
        again_choice == (a.get("choice") if a else None),
        "repeated request gives the same choice",
        f"{again_choice} vs {a.get('choice')}",
    )

    return report()


def report() -> int:
    print(f"\n== {len(PASSES)} passed, {len(FAILS)} failed")
    for f in FAILS:
        print("  FAILED: " + f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
