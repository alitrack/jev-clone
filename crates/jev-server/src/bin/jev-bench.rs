//! `jev-bench` — CLI shell for the prefix-reuse benchmark (specs/M1.md §4).
//!
//! All the logic lives in [`jev_server::bench`] so it has real tests; this file
//! only parses `argv`, runs the measurement, prints the table and lets the exit
//! code carry the outcome (`0` = the run completed and the JSON was written,
//! `2` = bad arguments, `1` = the run itself failed).
//!
//!     jev-bench --base-url http://127.0.0.1:8014/v1 --model qwen3.8-27b \
//!               --state-file bench/states/long-state.txt --questions 21 --repeat 3
//!
//! The command does **not** judge the acceptance gate: it prints the speedup it
//! measured. Whether that clears 3x is for the caller to read, not for the tool
//! to assert.

use jev_server::bench::{parse_args, render_table, run, USAGE};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();

    if argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return Ok(());
    }

    let args = match parse_args(&argv) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("jev-bench: {message}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    let report = run(&args).await?;
    println!("{}", render_table(&report));
    Ok(())
}
