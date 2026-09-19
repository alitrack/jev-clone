# bench/results/ — raw evidence, and how to read it

These files are the raw output of `jev-bench`. Every number in `specs/M1.md` §4.1 is
recomputable from them (each file carries both `raw_ms` arrays), which is why they
are committed at all: a benchmark number without its artifact is a claim.

**Read this before quoting a ratio.**

| Artifact | Protocol | top_k | Wall-clock ratio | Trustworthy as a speedup? |
|---|---|---|---|---|
| `interleaved/…json` | **interleaved A/B, 21 questions × 5 rounds** | 100 | **0.96×** | **Yes — this is the one to quote** |
| `topk20/…json` | block-timed (all of arm A, then all of arm B), 21 × 3 | 20 | 2.23× | **No** |
| `2026-09-19-DESKTOP-QD6AT84.json` (top level) | block-timed, 21 × 3 | 20 | 1.50× | **No** |
| `topk200/…json` | block-timed, 21 × 3 | 200 | 1.06× | **No** |
| `topk100/…json` | block-timed, 21 × 3 | 100 | 0.88× | **No** |

## Why four ratios for the same work are not four measurements of the same thing

The endpoint (`10.10.10.115:8014`) is a **shared** GPU. A block-timed run measures
the *sequence* "every fresh request, then every batched request", so any other
tenant's job that happens to land in one block is charged entirely to that arm. The
four block-timed ratios above span 0.88×–2.23× for identical work, which is the size
of that artifact — one single request in the block-timed fresh arm took **33,283 ms**.

The interleaved protocol (one round of fresh, immediately one round of shared,
repeated) cancels a slowly varying background load: its per-round ratios are
0.95 / 0.95 / 0.99 / 0.92 / 0.98, i.e. ±4% instead of ±150%.

`--top-k` was the only difference between the three `topk*` runs; the coverage counts
they recorded (118/126 declared slots present at `top_k=20` vs **126/126** at 100 and
200, with per-decision median latency 226 ms → 228 ms) are still valid evidence and
are what justified raising the floor to 100 (`specs/M1.md` §5②). **Only their
`speedup` field should be ignored.**

## Stale text inside the older artifacts

The `notes` array of every file committed before `top_k`'s floor moved from 20 to 100
still says the server rule is `max(n_slots + 5, 20)`. The rule is now
`max(n_slots + 5, 100)` (`crates/jev-server/src/api.rs`). The files were left byte-for-byte
as produced — editing a measurement artifact by hand is how evidence stops being
evidence — so the discrepancy is documented here instead.

The `topk*` runs remain reproducible with an explicit `--top-k 20|100|200`; they simply
no longer describe the default configuration.

## Revision note

All five artifacts were produced on 2026-09-19 against the working tree that became
commit `45553fa` (`interleaved/`) and the earlier state committed as `e3be413`
(the four block-timed runs). The interleaved protocol itself landed in `45553fa`.
