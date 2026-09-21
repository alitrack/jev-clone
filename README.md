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

### Endpoint dialect (measured 2026-09-21, llama.cpp on an M3 Ultra)

The OpenAI-compatible endpoint we were developed against is not the only one, and one of the
differences was a **silently wrong answer**, not a loud failure. All three are now negotiated in
`jev-backend`, so no proxy/shim is needed in front of the server:

| Difference | Reference endpoint | llama.cpp | How it is handled |
|---|---|---|---|
| `/tokenize` field | `prompt`; anything else → 400 | `content`; `prompt` → **`200 {"tokens": []}`** | both fields × both URL spellings are tried; a non-empty text that tokenizes to nothing is treated as "field not recognised", never cached as an empty prompt, and the working `(url, field)` pair is remembered |
| array `prompt` (batched readout) | accepted | **400** on build b9590; accepted on b11065 | only a **request-shape rejection** (400/404/405/415/422) falls back to sequential single prompts — 429, 401/403 and 5xx still propagate, so a throttled or unauthorised endpoint can never be laundered into a "successful" run — and the fallback is **counted**, so the bench report states that the shared-prefix number is then not one shared prefill |
| `logprobs` shape | `logprobs.top_logprobs[0]` — a token→logprob map | `logprobs.content[0].top_logprobs[]` — an array of `{token, logprob}` | the legacy map is read first (frozen behaviour), then the array; an empty distribution is `NoLogprobs`, a missing `token`/`logprob` is `Decode` |

The startup slot self-check is configurable with `JEV_SLOT_CHECK`:

- `strict` (default) — asserts the probed ids of the reference tokenizer
  (`"Answer:\n"` = `[15666, 25, 198]`, and `A` = `[32]`).
- `letters` — asserts only what the readout actually depends on, for **any** vocabulary: every slot
  letter is a single token and appending it does not re-tokenize the prompt tail. Needed because the
  strict check pins a whole tokenization the readout never uses (Qwen3-4B: `"Answer:\n"` =
  `[16141, 510, 32]`, and the old check refused to start a server that reads probabilities fine).

`usage.input_tokens` is always the **endpoint's own** prompt-token count, never a recomputation: for
a single request it is that request's number, and for a batched request it is whatever the endpoint
reported for the whole batch (attached to the first readout, the rest reporting 0, so summing the
vector reproduces the endpoint's number exactly). Backends disagree on what that number should be —
the reference endpoint counts `N × prefix` on a batch, llama.cpp b11065 counts the shared prefix once
(909 vs 18933 for the same 63 decisions) — so the bench report prints both counts and claims nothing
more. Do not use token counts as the speed measure anywhere.

Prefix reuse on llama.cpp is a property of the **model architecture as much as of the build**:
on build b11065 the server reports `cache_reuse is not supported by this context, it will be
disabled` for a hybrid (Gated DeltaNet) model such as Qwen3.5-4B, and the batched arm then measures
1.05× (154 ms/decision) — the same order as the dense Qwen3-4B's 1.74×/1.85× on the older build,
against 2.5–5.5× on the tuned direct probe. None of these three numbers is a ≥3× claim.

## License

TBD (private repository). Third-party reference implementations were read for design study only;
notably `daseinlabs/open-jev` ships no license and its code was **not** copied.
