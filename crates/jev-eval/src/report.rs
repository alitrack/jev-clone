//! The report: a **stratified** reduction of `(items, predictions)` to numbers.
//!
//! ## There is deliberately no total
//!
//! The `specs/M1.md` §6 requirement is "按题目来源/类别分层上报，**禁止合并成一个
//! 总分**". This module enforces that structurally rather than by convention:
//! [`Report`] has no field that could hold a grand total, only `strata`. Every
//! number in it is attributed to a source, and (one level down) to a category
//! within that source, plus a category-across-sources view. `accuracy`,
//! `balanced_accuracy`, `nll`, `brier_*` and `ece` are kept as separate fields
//! and never collapsed into a single "score" (AGENTS.md 铁律 7).
//!
//! ## Everything needed to recompute is referenced, not inlined
//!
//! The report records the SHA-256 of both input files. [`crate::verify`] uses
//! those hashes to prove it is looking at the same evidence, then re-derives
//! every field with the very same code path that produced it (SemIf's
//! `verify_published.py` approach).

use crate::error::EvalError;
use crate::items::{Item, ItemSet};
use crate::metrics::{argmax_first_max, summarize, Sample, StratumMetrics, DEFAULT_BINS};
use crate::predictions::Prediction;
use crate::sha::sha256_hex;
use crate::REPORT_SCHEMA;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Structured so a reader cannot accidentally average two strata together:
/// metrics live *under* a source and *under* a category, never at the top.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Strata {
    /// source -> metrics for that source, broken down further by category.
    pub by_source: BTreeMap<String, SourceStratum>,
    /// category -> metrics across every source (still a stratification, not a total).
    pub by_category: BTreeMap<String, StratumMetrics>,
}

/// One source: its own numbers, plus the same numbers per category inside it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceStratum {
    pub n_items: usize,
    /// Categories present under this source.
    pub categories: Vec<String>,
    /// Metrics over the whole source. This is a stratum, not a merged total: it
    /// covers exactly one provenance group.
    pub overall: StratumMetrics,
    pub by_category: BTreeMap<String, StratumMetrics>,
}

/// The published report. Field order here is the field order on disk.
///
/// `deny_unknown_fields` is deliberate: an artifact is evidence, and a field that
/// the loader silently ignores is a field a tamperer can add for free (a fabricated
/// `"total_accuracy": 0.99` at the top level would otherwise survive a `verify`
/// that reports success — the exact thing specs/M1.md §6 forbids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Report {
    pub schema: String,
    /// The item file, as it was named on the command line (provenance only; the
    /// hash below is what is actually checked).
    pub items_file: String,
    /// Frozen content hash of the item set (SHA-256 of the raw bytes).
    pub items_sha256: String,
    /// SHA-256 over `(id, rendered prompt)` for the whole set.
    pub prompt_set_sha256: String,
    pub predictions_file: String,
    pub predictions_sha256: String,
    pub confidence_bins: usize,
    pub nll_probability_floor: f64,
    /// Tolerance `verify` used (and will use) when comparing floats. Metadata,
    /// not a computed number, so `verify` does not compare it.
    pub verify_tolerance: f64,
    /// Restates *why* there is no grand total, in the artifact itself.
    pub stratification_note: String,
    pub strata: Strata,
}

/// Restates *why* the report has no merged total. Kept as a constant because
/// `verify` compares the artifact's copy against it: a report whose note has been
/// rewritten (e.g. into a fabricated grand total) must not verify clean, and a
/// "verified" report is otherwise read as proof that it is stratified.
pub const STRATIFICATION_NOTE: &str = "Stratified by source and by category. Every metric is reported \
per stratum; there is deliberately no merged total (AGENTS.md 铁律 7 / specs/M1.md §6). \
accuracy / balanced_accuracy / nll / brier_* / ece are separate fields and are not \
collapsed into one score.";

/// Reduce items + predictions to per-item [`Sample`]s.
///
/// The prediction for an item is looked up by id; an item with no prediction row
/// becomes a sample with no predicted slot (counted in `n_missing`).
pub fn samples_from(items: &[Item], predictions: &[Prediction]) -> Result<Vec<Sample>, EvalError> {
    let by_id: BTreeMap<&str, &Prediction> = predictions
        .iter()
        .map(|prediction| (prediction.id.as_str(), prediction))
        .collect();

    let mut samples = Vec::with_capacity(items.len());
    for item in items {
        let slots = item.slot_labels();
        let gold = item.gold_label().map_err(|message| EvalError::Item {
            path: "<in-memory item set>".to_string(),
            line: 0,
            id: item.id.clone(),
            message,
        })?;

        let prediction = by_id.get(item.id.as_str()).copied();
        let probs = prediction
            .and_then(|p| p.probabilities.as_ref())
            .map(|map| {
                slots
                    .iter()
                    .map(|slot| {
                        // `load_predictions` guarantees the key set equals the slot
                        // set, so this cannot miss; a miss here would be a bug in
                        // the loader, not in this item.
                        *map.get(slot)
                            .expect("load_predictions guarantees every slot has a probability")
                    })
                    .collect::<Vec<f64>>()
            });
        let predicted = match &probs {
            Some(vector) => Some(slots[argmax_first_max(vector)].clone()),
            None => prediction.and_then(|p| p.label.clone()),
        };

        samples.push(Sample {
            id: item.id.clone(),
            source: item.source.clone(),
            category: item.category.clone(),
            gold,
            slots,
            predicted,
            probs,
            positive: item.positive_label(),
        });
    }
    Ok(samples)
}

/// Group samples and compute every stratum's metrics. Shared by `run` and
/// `verify` so the two can never disagree about the definitions.
pub fn strata_from(samples: &[Sample], bins: usize) -> Strata {
    let mut by_source: BTreeMap<String, SourceStratum> = BTreeMap::new();
    let mut by_category: BTreeMap<String, Vec<Sample>> = BTreeMap::new();

    for sample in samples {
        by_category
            .entry(sample.category.clone())
            .or_default()
            .push(sample.clone());
    }

    let mut per_source: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
    for sample in samples {
        per_source
            .entry(sample.source.clone())
            .or_default()
            .push(sample.clone());
    }

    for (source, group) in &per_source {
        let mut categories: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
        for sample in group {
            categories
                .entry(sample.category.clone())
                .or_default()
                .push(sample.clone());
        }
        let by_category_metrics: BTreeMap<String, StratumMetrics> = categories
            .iter()
            .map(|(category, group)| (category.clone(), summarize(group, bins)))
            .collect();
        by_source.insert(
            source.clone(),
            SourceStratum {
                n_items: group.len(),
                categories: categories.keys().cloned().collect(),
                overall: summarize(group, bins),
                by_category: by_category_metrics,
            },
        );
    }

    let by_category_metrics: BTreeMap<String, StratumMetrics> = by_category
        .iter()
        .map(|(category, group)| (category.clone(), summarize(group, bins)))
        .collect();

    Strata {
        by_source,
        by_category: by_category_metrics,
    }
}

/// Build a report from an item set and a prediction file already in memory.
///
/// `predictions_sha256` is the hash of the prediction file's raw bytes; pass an
/// empty string only when there is no file on disk (tests build reports from
/// in-memory data).
pub fn build_report(
    set: &ItemSet,
    predictions: &[Prediction],
    predictions_file: &Path,
    predictions_sha256: &str,
    bins: usize,
    tolerance: f64,
) -> Result<Report, EvalError> {
    if bins == 0 {
        return Err(EvalError::message("confidence bin count must be >= 1"));
    }
    let samples = samples_from(&set.items, predictions)?;
    let strata = strata_from(&samples, bins);
    Ok(Report {
        schema: REPORT_SCHEMA.to_string(),
        items_file: set.path.display().to_string(),
        items_sha256: set.sha256.clone(),
        prompt_set_sha256: set.prompt_set_sha256.clone(),
        predictions_file: predictions_file.display().to_string(),
        predictions_sha256: predictions_sha256.to_string(),
        confidence_bins: bins,
        nll_probability_floor: crate::metrics::NLL_PROBABILITY_FLOOR,
        verify_tolerance: tolerance,
        stratification_note: STRATIFICATION_NOTE.to_string(),
        strata,
    })
}

/// SHA-256 of a file's raw bytes, or an error naming the path.
pub fn hash_file(path: &Path) -> Result<String, EvalError> {
    let bytes = std::fs::read(path).map_err(|source| EvalError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(sha256_hex(&bytes))
}

/// Serialize a report to pretty JSON (trailing newline included).
pub fn to_json(report: &Report) -> Result<String, EvalError> {
    let mut text = serde_json::to_string_pretty(report)
        .map_err(|err| EvalError::message(format!("cannot serialize report: {err}")))?;
    text.push('\n');
    Ok(text)
}

/// Parse a report back from JSON.
pub fn from_json(text: &str, path: &Path) -> Result<Report, EvalError> {
    serde_json::from_str(text).map_err(|err| EvalError::Report {
        path: path.display().to_string(),
        message: err.to_string(),
    })
}

/// Human-readable, still stratified (no total row anywhere).
pub fn to_markdown(report: &Report) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "# jev-eval report (`{}`)", report.schema);
    let _ = writeln!(out);
    let _ = writeln!(out, "- items: `{}`", report.items_file);
    let _ = writeln!(out, "- items sha256: `{}`", report.items_sha256);
    let _ = writeln!(out, "- prompt-set sha256: `{}`", report.prompt_set_sha256);
    let _ = writeln!(out, "- predictions: `{}`", report.predictions_file);
    let _ = writeln!(out, "- predictions sha256: `{}`", report.predictions_sha256);
    let _ = writeln!(out, "- confidence bins: {}", report.confidence_bins);
    let _ = writeln!(
        out,
        "- NLL probability floor: {}",
        report.nll_probability_floor
    );
    let _ = writeln!(out, "- verify tolerance: {}", report.verify_tolerance);
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", report.stratification_note);
    let _ = writeln!(out);

    let _ = writeln!(out, "## By source");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", markdown_header());
    for (source, stratum) in &report.strata.by_source {
        let _ = writeln!(out);
        let _ = writeln!(out, "### source `{source}` (n = {})", stratum.n_items);
        let _ = writeln!(out);
        let _ = writeln!(out, "{}", metrics_row("overall", &stratum.overall));
        for (category, metrics) in &stratum.by_category {
            let _ = writeln!(out, "{}", metrics_row(category, metrics));
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "## By category (across sources)");
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", markdown_header());
    for (category, metrics) in &report.strata.by_category {
        let _ = writeln!(out, "{}", metrics_row(category, metrics));
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "## Reliability curve (per stratum)");
    for (source, stratum) in &report.strata.by_source {
        for (category, metrics) in &stratum.by_category {
            if metrics.reliability.is_empty() {
                continue;
            }
            let _ = writeln!(out);
            let _ = writeln!(out, "### `{source}` / `{category}` (n_scored = {})", metrics.n_scored);
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "| bin | n | mean confidence | accuracy | gap |"
            );
            let _ = writeln!(out, "|---|---|---|---|---|");
            for bin in &metrics.reliability {
                let _ = writeln!(
                    out,
                    "| [{:.2}, {:.2}) | {} | {:.4} | {:.4} | {:.4} |",
                    bin.lower,
                    bin.upper,
                    bin.n,
                    bin.mean_confidence,
                    bin.accuracy,
                    (bin.accuracy - bin.mean_confidence).abs()
                );
            }
        }
    }
    out
}

fn fmt_opt(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{v:.6}"),
        None => "—".to_string(),
    }
}

fn metrics_row(label: &str, metrics: &StratumMetrics) -> String {
    format!(
        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
        label,
        metrics.n_items,
        metrics.n_predicted,
        metrics.n_scored,
        fmt_opt(metrics.accuracy),
        fmt_opt(metrics.balanced_accuracy),
        fmt_opt(metrics.nll),
        fmt_opt(metrics.brier_multiclass),
        fmt_opt(metrics.brier_binary),
        fmt_opt(metrics.ece),
    )
}

/// The markdown table header that pairs with [`metrics_row`].
pub fn markdown_header() -> &'static str {
    "| stratum | n | n_pred | n_scored | accuracy | balanced_acc | nll | brier_mc | brier_bin | ece |\n|---|---|---|---|---|---|---|---|---|---|"
}

/// Convenience used by the CLI to pin the bin default in one place.
pub fn default_bins() -> usize {
    DEFAULT_BINS
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn item(id: &str, source: &str, category: &str, gold: &str) -> Item {
        Item {
            id: id.into(),
            category: category.into(),
            source: source.into(),
            state: json!("state"),
            question: serde_json::from_value(json!({
                "type": "choice",
                "instructions": "pick",
                "criteria": {"no": null, "yes": null}
            }))
            .unwrap(),
            gold: json!(gold),
            positive: Some("yes".into()),
            provenance: None,
        }
    }

    fn prediction(id: &str, yes: f64) -> Prediction {
        let mut probabilities = BTreeMap::new();
        probabilities.insert("no".to_string(), 1.0 - yes);
        probabilities.insert("yes".to_string(), yes);
        Prediction {
            id: id.into(),
            label: None,
            probabilities: Some(probabilities),
        }
    }

    #[test]
    fn source_and_category_are_kept_apart_never_summed() {
        let items = vec![
            item("a", "s1", "cat1", "yes"),
            item("b", "s1", "cat2", "no"),
            item("c", "s2", "cat1", "yes"),
        ];
        let predictions = vec![prediction("a", 0.9), prediction("b", 0.9), prediction("c", 0.1)];

        let samples = samples_from(&items, &predictions).unwrap();
        let strata = strata_from(&samples, 10);

        // No grand total exists anywhere: the top level only has the two maps.
        assert_eq!(strata.by_source.len(), 2);
        assert_eq!(strata.by_source["s1"].n_items, 2);
        assert_eq!(strata.by_source["s2"].n_items, 1);
        // s1 overall: a correct (yes), b wrong (no gold, pred yes) -> 1/2
        assert_eq!(strata.by_source["s1"].overall.accuracy, Some(0.5));
        // s2 overall: c wrong -> 0/1
        assert_eq!(strata.by_source["s2"].overall.accuracy, Some(0.0));
        // The two are never combined into 0.333...
        assert_eq!(strata.by_source["s1"].categories, vec!["cat1", "cat2"]);
        // category view spans sources: cat1 has a (correct) and c (wrong) -> 1/2
        assert_eq!(strata.by_category["cat1"].accuracy, Some(0.5));
        assert_eq!(strata.by_category["cat1"].n_items, 2);
    }

    #[test]
    fn an_item_without_a_prediction_row_is_counted_as_missing() {
        let items = vec![item("a", "s1", "cat1", "yes"), item("b", "s1", "cat1", "no")];
        let predictions = vec![prediction("a", 0.9)];
        let samples = samples_from(&items, &predictions).unwrap();
        let metrics = summarize(&samples, 10);
        assert_eq!(metrics.n_items, 2);
        assert_eq!(metrics.n_missing, 1);
        assert_eq!(metrics.n_predicted, 1);
    }

    #[test]
    fn zero_bins_is_rejected_before_anything_is_computed() {
        let items = vec![item("a", "s1", "cat1", "yes")];
        let set = ItemSet {
            path: "in-memory".into(),
            sha256: "deadbeef".into(),
            prompt_set_sha256: "deadbeef".into(),
            items,
        };
        let err = build_report(
            &set,
            &[prediction("a", 0.9)],
            Path::new("p.jsonl"),
            "0",
            0,
            1e-12,
        )
        .unwrap_err();
        assert!(err.to_string().contains("bin count must be >= 1"));
    }

    #[test]
    fn the_json_round_trips_exactly() {
        let items = vec![item("a", "s1", "cat1", "yes")];
        let set = ItemSet {
            path: "items.jsonl".into(),
            sha256: "abc".into(),
            prompt_set_sha256: "def".into(),
            items,
        };
        let report = build_report(
            &set,
            &[prediction("a", 0.7)],
            Path::new("p.jsonl"),
            "0123",
            10,
            1e-12,
        )
        .unwrap();
        let text = to_json(&report).unwrap();
        let parsed = from_json(&text, Path::new("r.json")).unwrap();
        assert_eq!(parsed, report);
    }

    #[test]
    fn the_markdown_view_has_no_totals_row() {
        let items = vec![item("a", "s1", "cat1", "yes")];
        let set = ItemSet {
            path: "items.jsonl".into(),
            sha256: "abc".into(),
            prompt_set_sha256: "def".into(),
            items,
        };
        let report = build_report(
            &set,
            &[prediction("a", 0.7)],
            Path::new("p.jsonl"),
            "0123",
            10,
            1e-12,
        )
        .unwrap();
        let md = to_markdown(&report);
        assert!(md.contains("### source `s1`"));
        assert!(md.contains("By category (across sources)"));
        assert!(md.contains("Reliability curve"));
        assert!(!md.to_lowercase().contains("| total |"));
    }
}
