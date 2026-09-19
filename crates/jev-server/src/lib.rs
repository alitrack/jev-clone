//! jev-server library surface.
//!
//! The HTTP surface lives in [`api`]. It is compiled both into the `jev-server`
//! binary (`src/main.rs`) and into this library target; the library exists so the
//! no-GPU end-to-end tests in `tests/` can assemble the very same router with a
//! mock backend and a fake tokenizer (specs/M0.md §4 B5.3) instead of spawning the
//! binary and talking to a real model endpoint.
//!
//! Everything here is deliberately backend-agnostic: [`api::AppState`] holds a
//! `dyn DecisionBackend` and an `HttpTokenizer` behind `Arc`s, and nothing in the
//! request path assumes a particular server.
//!
//! [`bench`] is the measurement rig behind the `jev-bench` binary: it lives in
//! the library so its testable parts (arg parsing, workload generation,
//! percentiles, report assembly) are covered by `cargo test`.

pub mod api;
pub mod bench;
