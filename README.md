# jev-clone

A local, **contract-compatible System One decision server**: give it a `state` and a set of
runtime-defined typed questions (`choice` / `score` / `noul`), get back a probability
distribution per question — **without generating a single token**.

> **Status: M0 and M1 landed, accepted against a real endpoint** (164 unit tests / clippy clean /
> contract assertions 24/24 / **150-item Chinese set** driven end to end + `verify` recomputing
> 572 checks with 0 mismatches, with a rewritten report rejected). **M1's "≥3× prefix reuse" gate
> measured 0.96× on the shared endpoint and is not met** — the reason and the restatement are in
> [`specs/M1.md`](specs/M1.md) §4.1; do not quote it as a result.
> Design: [`docs/design.md`](docs/design.md) (Chinese); source-level study of the three reference
> implementations: [`docs/blueprints.md`](docs/blueprints.md).

> **Independent project.** Not affiliated with, endorsed by, or derived from TypeSafe or Jev.
> "Jev", "TypeSafe", and "System One" are the property of their respective owners. No official
> weights, data, or code were used. See [Honesty](#honesty) below.

---

## What this is

A server that answers this shape of request:

```json
{
  "state": "My payouts have been failing for 3 days. Any idea what's going on?",
  "model": "local-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": { "billing": "Payments, invoicing, refunds", "technical": "Bugs, outages" }
    },
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?" },
    "frustration": { "type": "score", "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Frustrated", "Very angry"] }
  }
}
```

with this shape of reply (field-for-field compatible with the public TypeSafe HTTP contract):

```json
{
  "model": "local-latest",
  "answers": {
    "department": { "type": "choice", "choice": "technical",
                    "probabilities": { "billing": 0.07, "technical": 0.93 }, "confidence": 0.79 },
    "is_urgent": { "type": "noul", "noul": 0.94 },
    "frustration": { "type": "score", "score": 1.6, "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                     "probabilities": { "0": 0.05, "1": 0.3, "2": 0.65 }, "confidence": 0.31 }
  },
  "usage": { "input_tokens": 312, "output_tokens": 0 }
}
```

Every answer is **constrained to the options you supplied** — your code never parses prose.
Many questions in one request share one prefill and run in parallel.

## Errors

Two failure classes, and never mixed up:

| Status | Meaning | Examples |
|---|---|---|
| **422** | The request, or the readout it produced, does not hold up. Fix the input; retrying changes nothing. | no questions; fewer than 2 options; more than 26 options; fewer than 2 levels; prompt over the token budget; a declared slot missing from the readout (`SlotMissingFromTopK`) |
| **502** | The model endpoint we depend on failed. Retrying may help. | HTTP error from the endpoint; a body we cannot parse; a response that carries no `logprobs` |

A missing slot is deliberately a 422 and **never** a zero: an option whose probability the model
did not report is an option whose probability is *unknown*, and silently treating it as 0 — or
renormalising the rest — would invent a number in a system whose whole point is calibrated
probabilities.

## What this is not

- Not a chat model. It never writes prose. Anything that needs text generation is out of scope
  by construction.
- Not a claim of "cannot be wrong". The only guarantee is that the output cannot fall outside
  the option set. A forced typed output can still be semantically wrong (this is stated by the
  vendor's own failure-mode documentation, and is the reason this project's primary metric is
  **calibration**, not accuracy).
- Not a reproduction of the vendor's undisclosed architecture, training method, or numbers.

## Design in one line

Three open reimplementations already published the *interface shape*; none of them closed the
**calibration** gap, and none targets Chinese. So this project optimizes for (1) calibrated
probabilities with published ECE/Brier/reliability curves, (2) Chinese-first evaluation, and
(3) prefix reuse as an architectural requirement rather than an afterthought.

## Roadmap

| Stage | Deliverable | Gate |
|---|---|---|
| **M0** | Contract layer + slot verification + HTTP server (no GPU needed) | Field-level match on the public contract's example requests; slot-verification failure paths covered by tests |
| **M1** | Batched readout (`prompt` array) + calibrated evaluation crate | Prefix reuse **measured** on the shared endpoint: 0.96×, i.e. the gate is **not** met there (the server's own radix cache already gives the baseline the same reuse) — see `specs/M1.md` §4.1, which restates the gate for a backend whose cache we control |
| **M2** | Frozen evaluation matrix (Chinese-first) + reproducibility check | One command recomputes every published number from raw results (**crate + frozen 150-item Chinese set `zh-evidence-v1` landed 2026-09-19** — `eval/README.md`). The set is entirely self-authored, so its numbers are a relative comparison on one endpoint, not a capability claim |
| **M3** | Route A head → Route C LoRA + temperature calibration | ECE ≤ 0.05 on the Chinese set without losing accuracy. ⚠️ **This gate is already met before any calibration work** (uncalibrated ECE 0.04395 on the 150-item set), so it does not discriminate as written — the gate needs restating before it can be used as evidence (see `eval/README.md` §7) |
| **M4** | Optional: expose as an "auto answerer" for canvas/workflow contracts | Low confidence escalates to a human |

## Honesty

- The upstream service is closed-source and hosted; its architecture and parameter count are
  undisclosed. Nothing here is reverse-engineered from its weights.
- `confidence` is our own statistic computed from the returned distribution. The vendor does not
  publish its formula (and explicitly says you are not locked into theirs). We implement several
  candidates and select by expected calibration error.
- Vendor latency/cost multipliers (40–200×, 40–400×) come from the vendor's own workflow
  benchmarks and must not be reused as ours.
- Evaluation sets are declared with content hashes. Sources are never pooled into one score.

## License

TBD (private repository). Third-party reference implementations were read for design study only;
notably `daseinlabs/open-jev` ships no license and its code was **not** copied.
