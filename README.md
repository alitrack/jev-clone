# jev-clone

A local, **contract-compatible System One decision server**: give it a `state` and a set of
runtime-defined typed questions (`choice` / `score` / `noul`), get back a probability
distribution per question — **without generating a single token**.

> **Status: design phase.** No implementation code yet. See [`docs/design.md`](docs/design.md)
> (Chinese) and [`docs/blueprints.md`](docs/blueprints.md) for the source-level study of the
> reference implementations this design is based on.

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
| **M1** | llama.cpp backend + shared-state prefix reuse | Same-state N questions ≥ 3× decisions/s vs per-question re-encode (4090 / Apple silicon benchmark tables) |
| **M2** | Frozen evaluation matrix (Chinese-first) + reproducibility check | One command recomputes every published number from raw results |
| **M3** | Route A head → Route C LoRA + temperature calibration | ECE ≤ 0.05 on the Chinese set without losing accuracy |
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
