//! # jev-eval — frozen item sets, frozen metrics, re-computable reports
//!
//! This crate is the reproducibility half of M1 (§6 of `specs/M1.md`). It does
//! not call any model: it reads a **frozen item set** (JSONL) and a
//! **prediction file** (JSONL) and reduces them to numbers a third party can
//! recompute by hand.
//!
//! Three ideas hold the whole thing together:
//!
//! 1. **The item set is frozen by content hash.** [`ItemSet::sha256`] is the
//!    SHA-256 of the item file's raw bytes; it is written into the report. The
//!    optional manifest check refuses to compute anything when the file no
//!    longer hashes to what the manifest recorded.
//! 2. **Every number is defined, not implied.** The definitions of balanced
//!    accuracy, NLL, Brier (two of them) and ECE + the reliability curve live in
//!    [`metrics`] and are restated in `eval/README.md`. See
//!    [`NLL_PROBABILITY_FLOOR`] for the one magic constant.
//! 3. **Nothing gets merged into a total.** [`report::Report`] is *stratified*
//!    only — by source, and by category within source. There is deliberately no
//!    top-level score anywhere in the structure (AGENTS.md 铁律 7, `specs/M1.md`
//!    §6). [`verify`] independently recomputes every field and reports diffs.
//!
//! Typical use:
//!
//! ```no_run
//! use jev_eval::{build_report, load_item_set, load_predictions};
//! # fn main() -> Result<(), jev_eval::EvalError> {
//! let set = load_item_set("eval/items/zh-evidence-v1.jsonl".as_ref())?;
//! let preds = load_predictions("eval/predictions/example-handwritten.jsonl".as_ref(), &set)?;
//! let report = build_report(&set, &preds, "eval/predictions/example-handwritten.jsonl".as_ref(), "", 10, 1e-12)?;
//! # let _ = report; Ok(()) }
//! ```

pub mod cli;
pub mod error;
pub mod import;
pub mod items;
pub mod metrics;
pub mod predictions;
pub mod report;
pub mod sha;
pub mod verify;

pub use error::EvalError;
pub use items::{load_item_set, Item, ItemSet, Manifest, KNOWN_CATEGORIES};
pub use metrics::{
    argmax_first_max, summarize, Sample, StratumMetrics, DEFAULT_BINS, NLL_PROBABILITY_FLOOR,
};
pub use predictions::{load_predictions, Prediction};
pub use report::{build_report, strata_from, Report, Strata};
pub use verify::{verify, Diff, VerifyOutcome};

/// Schema tag written into every report, so a reader can tell which revision of
/// the definitions produced the numbers.
pub const REPORT_SCHEMA: &str = "jev-eval-report/1";
/// Schema tag for the item-set manifest.
pub const MANIFEST_SCHEMA: &str = "jev-eval-manifest/1";
