//! `verify` — recompute a published report from the raw evidence and compare.
//!
//! The model here is SemIf's `benchmarks/verify_published.py`: the report is a
//! *claim*, the item file and prediction file are the *evidence*, and verification
//! means re-deriving every claimed number from the evidence rather than trusting
//! the report. Two consequences worth stating out loud:
//!
//! * The item-set hash in the report is checked first. If the item file changed,
//!   verification fails at that point instead of comparing numbers that were never
//!   comparable.
//! * Nothing is "repaired" silently. Every mismatch is emitted as a
//!   [`Diff`] with the JSON path, the claimed value and the recomputed one, and the
//!   CLI exits non-zero when the list is non-empty.
//!
//! `verify_tolerance` is deliberately **not** compared: it is metadata describing
//! how the comparison was made, not a number the report claims about the data.

use crate::error::EvalError;
use crate::items::ItemSet;
use crate::metrics::StratumMetrics;
use crate::predictions::Prediction;
use crate::report::{samples_from, strata_from, Report, Strata};

/// One disagreement between a report and a fresh recomputation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diff {
    /// JSON-ish path of the field, e.g. `strata.by_source.s1.overall.ece`.
    pub path: String,
    /// What the report claims.
    pub claimed: String,
    /// What the evidence recomputes to.
    pub recomputed: String,
}

impl Diff {
    fn new(path: impl Into<String>, claimed: impl Into<String>, recomputed: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            claimed: claimed.into(),
            recomputed: recomputed.into(),
        }
    }
}

/// Result of a verification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutcome {
    /// How many individual comparisons were made (so "0 diffs" is meaningful).
    pub checks: usize,
    pub diffs: Vec<Diff>,
}

impl VerifyOutcome {
    pub fn is_ok(&self) -> bool {
        self.diffs.is_empty()
    }

    /// One line per mismatch, for printing.
    pub fn report(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "{} checks; {} mismatch{}",
            self.checks,
            self.diffs.len(),
            if self.diffs.len() == 1 { "" } else { "es" }
        );
        for diff in &self.diffs {
            let _ = writeln!(
                out,
                "  - {}: claimed {} / recomputed {}",
                diff.path, diff.claimed, diff.recomputed
            );
        }
        out
    }
}

struct Comparator {
    tolerance: f64,
    checks: usize,
    diffs: Vec<Diff>,
}

impl Comparator {
    fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            checks: 0,
            diffs: Vec::new(),
        }
    }

    fn text(&mut self, path: &str, claimed: &str, recomputed: &str) {
        self.checks += 1;
        if claimed != recomputed {
            self.diffs.push(Diff::new(path, claimed, recomputed));
        }
    }

    fn count(&mut self, path: &str, claimed: usize, recomputed: usize) {
        self.checks += 1;
        if claimed != recomputed {
            self.diffs
                .push(Diff::new(path, claimed.to_string(), recomputed.to_string()));
        }
    }

    /// File-name comparison by basename: the recorded path is provenance, not a
    /// number, so verification must survive being run from another directory.
    fn file(&mut self, path: &str, claimed: &str, recomputed: &str) {
        self.checks += 1;
        if !same_file(claimed, recomputed) {
            self.diffs.push(Diff::new(path, claimed, recomputed));
        }
    }

    fn float(&mut self, path: &str, claimed: f64, recomputed: f64) {
        self.checks += 1;
        let delta = (claimed - recomputed).abs();
        // Written as `is_nan || delta > tol` rather than `!(delta <= tol)` so NaN
        // counts as a mismatch and the negation-of-comparison lint stays quiet.
        if delta.is_nan() || delta > self.tolerance {
            self.diffs.push(Diff::new(
                path,
                format!("{claimed:.17}"),
                format!("{recomputed:.17}"),
            ));
        }
    }

    fn optional_float(&mut self, path: &str, claimed: Option<f64>, recomputed: Option<f64>) {
        match (claimed, recomputed) {
            (Some(a), Some(b)) => self.float(path, a, b),
            (None, None) => self.checks += 1,
            (a, b) => {
                self.checks += 1;
                self.diffs
                    .push(Diff::new(path, format!("{a:?}"), format!("{b:?}")));
            }
        }
    }

    fn metrics(&mut self, path: &str, claimed: &StratumMetrics, recomputed: &StratumMetrics) {
        self.count(&format!("{path}.n_items"), claimed.n_items, recomputed.n_items);
        self.count(&format!("{path}.n_missing"), claimed.n_missing, recomputed.n_missing);
        self.count(
            &format!("{path}.n_predicted"),
            claimed.n_predicted,
            recomputed.n_predicted,
        );
        self.count(&format!("{path}.n_scored"), claimed.n_scored, recomputed.n_scored);
        self.optional_float(&format!("{path}.accuracy"), claimed.accuracy, recomputed.accuracy);
        self.optional_float(
            &format!("{path}.balanced_accuracy"),
            claimed.balanced_accuracy,
            recomputed.balanced_accuracy,
        );
        self.text(
            &format!("{path}.classes_in_gold"),
            &format!("{:?}", claimed.classes_in_gold),
            &format!("{:?}", recomputed.classes_in_gold),
        );

        let claimed_classes: Vec<&String> = claimed.per_class_recall.keys().collect();
        let recomputed_classes: Vec<&String> = recomputed.per_class_recall.keys().collect();
        self.text(
            &format!("{path}.per_class_recall.keys"),
            &format!("{claimed_classes:?}"),
            &format!("{recomputed_classes:?}"),
        );
        for (class, value) in &claimed.per_class_recall {
            match recomputed.per_class_recall.get(class) {
                Some(other) => self.float(&format!("{path}.per_class_recall.{class}"), *value, *other),
                None => self.diffs.push(Diff::new(
                    format!("{path}.per_class_recall.{class}"),
                    format!("{value:.17}"),
                    "<absent>".to_string(),
                )),
            }
        }

        self.optional_float(&format!("{path}.nll"), claimed.nll, recomputed.nll);
        self.optional_float(
            &format!("{path}.brier_multiclass"),
            claimed.brier_multiclass,
            recomputed.brier_multiclass,
        );
        self.count(&format!("{path}.n_binary"), claimed.n_binary, recomputed.n_binary);
        self.optional_float(
            &format!("{path}.brier_binary"),
            claimed.brier_binary,
            recomputed.brier_binary,
        );
        self.optional_float(&format!("{path}.ece"), claimed.ece, recomputed.ece);

        self.count(
            &format!("{path}.reliability.len"),
            claimed.reliability.len(),
            recomputed.reliability.len(),
        );
        for (index, bin) in claimed.reliability.iter().enumerate() {
            let Some(other) = recomputed.reliability.get(index) else {
                continue;
            };
            let at = format!("{path}.reliability[{index}]");
            self.float(&format!("{at}.lower"), bin.lower, other.lower);
            self.float(&format!("{at}.upper"), bin.upper, other.upper);
            self.count(&format!("{at}.n"), bin.n, other.n);
            self.float(
                &format!("{at}.mean_confidence"),
                bin.mean_confidence,
                other.mean_confidence,
            );
            self.float(&format!("{at}.accuracy"), bin.accuracy, other.accuracy);
        }
    }

    fn strata(&mut self, claimed: &Strata, recomputed: &Strata) {
        let claimed_sources: Vec<&String> = claimed.by_source.keys().collect();
        let recomputed_sources: Vec<&String> = recomputed.by_source.keys().collect();
        self.text(
            "strata.by_source.keys",
            &format!("{claimed_sources:?}"),
            &format!("{recomputed_sources:?}"),
        );
        for (source, stratum) in &claimed.by_source {
            let Some(other) = recomputed.by_source.get(source) else {
                continue;
            };
            let at = format!("strata.by_source.{source}");
            self.count(&format!("{at}.n_items"), stratum.n_items, other.n_items);
            self.text(
                &format!("{at}.categories"),
                &format!("{:?}", stratum.categories),
                &format!("{:?}", other.categories),
            );
            self.metrics(&format!("{at}.overall"), &stratum.overall, &other.overall);

            let claimed_categories: Vec<&String> = stratum.by_category.keys().collect();
            let recomputed_categories: Vec<&String> = other.by_category.keys().collect();
            self.text(
                &format!("{at}.by_category.keys"),
                &format!("{claimed_categories:?}"),
                &format!("{recomputed_categories:?}"),
            );
            for (category, metrics) in &stratum.by_category {
                if let Some(other) = other.by_category.get(category) {
                    self.metrics(&format!("{at}.by_category.{category}"), metrics, other);
                }
            }
        }

        let claimed_categories: Vec<&String> = claimed.by_category.keys().collect();
        let recomputed_categories: Vec<&String> = recomputed.by_category.keys().collect();
        self.text(
            "strata.by_category.keys",
            &format!("{claimed_categories:?}"),
            &format!("{recomputed_categories:?}"),
        );
        for (category, metrics) in &claimed.by_category {
            if let Some(other) = recomputed.by_category.get(category) {
                self.metrics(&format!("strata.by_category.{category}"), metrics, other);
            }
        }
    }
}

/// Compare two file names by basename, so verification survives being run from a
/// different working directory (the recorded path is provenance, not a number).
fn same_file(claimed: &str, actual: &str) -> bool {
    let claimed_name = std::path::Path::new(claimed).file_name();
    let actual_name = std::path::Path::new(actual).file_name();
    claimed_name == actual_name
}

/// Recompute every number in `report` from `set` + `predictions` and list the
/// disagreements.
///
/// `predictions_sha256` is the hash of the prediction file on disk. Pass an empty
/// string to skip that particular check (used by in-memory tests).
pub fn verify(
    set: &ItemSet,
    predictions: &[Prediction],
    predictions_sha256: &str,
    predictions_file: &str,
    report: &Report,
    tolerance: f64,
) -> Result<VerifyOutcome, EvalError> {
    let mut cmp = Comparator::new(tolerance);

    // 0. Identity and hashes first: everything below assumes the evidence has not
    //    moved underneath us.
    cmp.text("schema", &report.schema, crate::REPORT_SCHEMA);
    let items_file = set.path.display().to_string();
    cmp.file("items_file", &items_file, &report.items_file);
    cmp.file("predictions_file", predictions_file, &report.predictions_file);
    cmp.text("items_sha256", &set.sha256, &report.items_sha256);
    cmp.text(
        "prompt_set_sha256",
        &set.prompt_set_sha256,
        &report.prompt_set_sha256,
    );
    if !predictions_sha256.is_empty() {
        cmp.text(
            "predictions_sha256",
            predictions_sha256,
            &report.predictions_sha256,
        );
    }
    cmp.float(
        "nll_probability_floor",
        crate::metrics::NLL_PROBABILITY_FLOOR,
        report.nll_probability_floor,
    );

    if report.confidence_bins == 0 {
        return Err(EvalError::message(
            "report declares 0 confidence bins; cannot recompute",
        ));
    }

    // 1. Recompute, using the bin count the report itself declares.
    let samples = samples_from(&set.items, predictions)?;
    let recomputed = strata_from(&samples, report.confidence_bins);

    // 2. Compare every stratum.
    cmp.strata(&report.strata, &recomputed);

    Ok(VerifyOutcome {
        checks: cmp.checks,
        diffs: cmp.diffs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basename_comparison_ignores_the_directory() {
        assert!(same_file("a/b/items.jsonl", "./items.jsonl"));
        assert!(!same_file("a/items.jsonl", "a/other.jsonl"));
        assert!(!same_file("items.jsonl", "predictions.jsonl"));
    }

    #[test]
    fn a_reported_diff_renders_one_line_per_mismatch() {
        let outcome = VerifyOutcome {
            checks: 3,
            diffs: vec![
                Diff::new("a", "1", "2"),
                Diff::new("b", "3", "4"),
            ],
        };
        assert!(!outcome.is_ok());
        let text = outcome.report();
        assert!(text.contains("2 mismatches"), "{text}");
        assert!(text.contains("a: claimed 1 / recomputed 2"), "{text}");
    }

    #[test]
    fn the_float_comparator_uses_a_strict_tolerance() {
        let mut cmp = Comparator::new(1e-12);
        cmp.float("x", 1.0, 1.0 + 1e-13);
        assert!(cmp.diffs.is_empty());
        cmp.float("y", 1.0, 1.0 + 1e-6);
        assert_eq!(cmp.diffs.len(), 1);
        assert_eq!(cmp.checks, 2);
    }
}
