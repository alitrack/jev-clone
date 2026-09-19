//! Command-line surface. Three subcommands, all of them offline:
//!
//! ```text
//! jev-eval run     --items <items.jsonl> --predictions <preds.jsonl> [--manifest m.json] [--bins 10] [--out r.json] [--md r.md]
//! jev-eval verify  --items <items.jsonl> --predictions <preds.jsonl> --report <r.json> [--tol 1e-12]
//! jev-eval import  --items <items.jsonl> --answers <response.json> --out <preds.jsonl>
//! ```
//!
//! `run` is the only writer. `verify` never writes anything and exits non-zero on
//! the first non-empty diff list, which is what makes it usable as a CI gate:
//! "re-run the command, the numbers must come back identical".

use crate::error::EvalError;
use crate::import::{import_answers, predictions_to_jsonl};
use crate::items::{check_expected_hash, check_manifest, load_item_set};
use crate::predictions::load_predictions;
use crate::report::{build_report, from_json, hash_file, to_json, to_markdown};
use crate::verify::verify;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

/// Default float tolerance for `verify`, in the spirit of SemIf's
/// `verify_published.py` (`close(..., tolerance=5e-10)`), tightened because a
/// recomputation here is bit-identical rather than merely close.
pub const DEFAULT_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Parser)]
#[command(
    name = "jev-eval",
    about = "Frozen item sets, frozen metrics, re-computable reports (offline; calls no model)",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Compute a stratified report from a frozen item set and a prediction file.
    Run(RunArgs),
    /// Recompute every number in a report from the raw evidence and diff it.
    Verify(VerifyArgs),
    /// Convert a server answers file into the prediction JSONL the metrics read.
    Import(ImportArgs),
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Frozen item set (JSONL). Also hash-pinned: see `--manifest` / `--expect-sha256`.
    #[arg(long)]
    pub items: PathBuf,
    /// Predictions (JSONL), one row per item id.
    #[arg(long)]
    pub predictions: PathBuf,
    /// `eval/items/manifest.json`: refuse to compute when the item file's hash is
    /// not the manifest's.
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    /// Refuse to compute unless the item file hashes to exactly this value.
    #[arg(long)]
    pub expect_sha256: Option<String>,
    /// Number of equal-width confidence bins for ECE / the reliability curve.
    #[arg(long, default_value_t = crate::report::default_bins())]
    pub bins: usize,
    /// Tolerance recorded in the report (and used by the matching `verify`).
    #[arg(long, default_value_t = DEFAULT_TOLERANCE)]
    pub tol: f64,
    /// Where to write the report JSON. Omit to print it to stdout.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Also write the human-readable stratified view here.
    #[arg(long)]
    pub md: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct VerifyArgs {
    #[arg(long)]
    pub items: PathBuf,
    #[arg(long)]
    pub predictions: PathBuf,
    /// The report to check.
    #[arg(long)]
    pub report: PathBuf,
    /// Tolerance for float comparison. Defaults to the report's own recorded value.
    #[arg(long)]
    pub tol: Option<f64>,
}

#[derive(Debug, Args)]
pub struct ImportArgs {
    #[arg(long)]
    pub items: PathBuf,
    /// A `SystemOneResponse`-shaped body: `{model?, answers: {id: Answer}, usage?}`.
    #[arg(long)]
    pub answers: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
}

/// Parse `argv` and dispatch. Returns `Err` for anything that should be a
/// non-zero exit — including a non-empty diff list from `verify`.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => cmd_run(args),
        Command::Verify(args) => cmd_verify(args),
        Command::Import(args) => cmd_import(args),
    }
}

fn cmd_run(args: RunArgs) -> anyhow::Result<()> {
    if args.bins == 0 {
        anyhow::bail!("--bins must be >= 1");
    }

    let set = load_item_set(&args.items)?;
    if let Some(manifest) = &args.manifest {
        check_manifest(manifest, &args.items, &set.sha256)?;
    }
    if let Some(expected) = &args.expect_sha256 {
        check_expected_hash(expected, &set.sha256, &args.items)?;
    }

    let predictions = load_predictions(&args.predictions, &set)?;
    let predictions_sha256 = hash_file(&args.predictions)?;
    let report = build_report(
        &set,
        &predictions,
        &args.predictions,
        &predictions_sha256,
        args.bins,
        args.tol,
    )?;

    let json = to_json(&report)?;
    match &args.out {
        Some(path) => {
            std::fs::write(path, &json).map_err(|source| EvalError::Io {
                path: path.display().to_string(),
                source,
            })?;
            println!("wrote report: {}", path.display());
        }
        None => print!("{json}"),
    }
    if let Some(path) = &args.md {
        std::fs::write(path, to_markdown(&report)).map_err(|source| EvalError::Io {
            path: path.display().to_string(),
            source,
        })?;
        println!("wrote markdown: {}", path.display());
    }

    println!(
        "items sha256 {}\nprompt-set sha256 {}\nitems {} / predictions {} / sources {} / categories {}",
        set.sha256,
        set.prompt_set_sha256,
        set.items.len(),
        predictions.len(),
        report.strata.by_source.len(),
        report.strata.by_category.len()
    );
    Ok(())
}

fn cmd_verify(args: VerifyArgs) -> anyhow::Result<()> {
    let set = load_item_set(&args.items)?;
    let predictions = load_predictions(&args.predictions, &set)?;
    let predictions_sha256 = hash_file(&args.predictions)?;
    let text = std::fs::read_to_string(&args.report).map_err(|source| EvalError::Io {
        path: args.report.display().to_string(),
        source,
    })?;
    let report = from_json(&text, &args.report)?;
    let tolerance = args.tol.unwrap_or(report.verify_tolerance);

    let outcome = verify(
        &set,
        &predictions,
        &predictions_sha256,
        &args.predictions.display().to_string(),
        &report,
        tolerance,
    )?;

    print!("{}", outcome.report());
    if !outcome.is_ok() {
        anyhow::bail!(
            "{} metric(s) in {} do not match the evidence",
            outcome.diffs.len(),
            args.report.display()
        );
    }
    println!("report verified against the raw evidence (tolerance {tolerance})");
    Ok(())
}

fn cmd_import(args: ImportArgs) -> anyhow::Result<()> {
    let set = load_item_set(&args.items)?;
    let outcome = import_answers(&args.answers, &set.items)?;
    let jsonl = predictions_to_jsonl(&outcome.predictions)?;
    std::fs::write(&args.out, jsonl).map_err(|source| EvalError::Io {
        path: args.out.display().to_string(),
        source,
    })?;

    if let Some(model) = &outcome.model {
        println!("model reported by the answers file: {model}");
    }
    println!(
        "wrote {} prediction row(s) to {}",
        outcome.predictions.len(),
        args.out.display()
    );
    if !outcome.missing_answers.is_empty() {
        eprintln!(
            "warning: {} item(s) had no answer and were written as no-answer rows: {:?}",
            outcome.missing_answers.len(),
            outcome.missing_answers
        );
    }
    if !outcome.contract_disagreements.is_empty() {
        eprintln!(
            "warning: {} item(s) where the server's own decision disagrees with the argmax of \
its own distribution:",
            outcome.contract_disagreements.len()
        );
        for line in &outcome.contract_disagreements {
            eprintln!("  - {line}");
        }
    }
    Ok(())
}
