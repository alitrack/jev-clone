//! `jev-eval` — see `cli.rs` for the subcommands and `lib.rs` for the design.
//!
//! Nothing in this binary talks to a network or to a model. It reads JSONL in,
//! writes JSON/Markdown out.

fn main() -> anyhow::Result<()> {
    jev_eval::cli::run()
}
