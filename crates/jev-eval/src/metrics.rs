//! The metric definitions. Every number `jev-eval` emits is defined here, once.
//!
//! The same definitions are restated in prose in `eval/README.md`; if the two
//! ever disagree, this file is the truth and the README is the bug.
//!
//! ## Definitions (frozen)
//!
//! * **predicted slot** = `argmax` of the probability vector, ties broken by the
//!   first slot in the item's declared slot order; for rows without a
//!   probability vector it is the prediction file's `label`.
//! * **accuracy** = `#(predicted == gold) / #rows with a prediction`.
//! * **balanced accuracy** = mean of per-class recall over the classes that
//!   actually occur as `gold`. Classes with no gold row are *excluded* (their
//!   recall is `0/0`), which is the SemIf `evaluate.py` convention — so this is
//!   macro-recall, not "mean over the item's declared slots". With more than two
//!   classes it still means exactly that: mean recall over the gold classes.
//! * **NLL** = mean of `-ln(max(p_gold, NLL_PROBABILITY_FLOOR))` over scored rows.
//! * **Brier** comes in two flavours and both are reported, because "the Brier
//!   score" is ambiguous the moment there are more than two options:
//!   * `brier_multiclass` = mean of `Σ_k (p_k - 1[gold == slot_k])²` over scored
//!     rows. Defined for any number of slots; needs no binarisation.
//!   * `brier_binary` = mean of `(p_positive - 1[gold == positive])²` over the
//!     scored rows that declare a positive class. `n_binary` says how many rows
//!     that was, and it is `null` when the stratum has none.
//!   * For a two-slot row the two are related exactly: `multiclass = 2 × binary`,
//!     because the two per-class terms are symmetric. A test asserts this.
//! * **confidence** = `max(p)` (the probability mass behind the predicted slot).
//! * **ECE** = `Σ_b (n_b / N) · |acc_b − conf_b|` over `bins` equal-width bins in
//!   `[0,1]`; bin index = `min(bins-1, floor(conf · bins))`, so `conf == 1.0`
//!   lands in the last bin rather than overflowing. `N` is the number of scored
//!   rows, `acc_b` the mean correctness in the bin and `conf_b` the mean
//!   confidence in it.
//! * **reliability curve** = the per-bin `(lower, upper, n, mean_confidence,
//!   accuracy)` tuples that ECE is summed from. Empty bins are omitted, matching
//!   SemIf's `reliability_bins`.
//!
//! Rows that carry no probability vector are counted in `n_items`/`n_missing` and
//! contribute to accuracy, but they are excluded from NLL / Brier / ECE — and
//! `n_scored` names exactly how many rows did contribute, so coverage is never
//! hidden.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Probability floor for NLL. Keeps `-ln(0)` from becoming `+inf`; mirrors
/// SemIf's `nll_probability_floor`.
pub const NLL_PROBABILITY_FLOOR: f64 = 1e-12;

/// Default number of equal-width confidence bins for ECE / the reliability curve.
pub const DEFAULT_BINS: usize = 10;

/// One item reduced to exactly the quantities the metrics need.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub id: String,
    /// Which frozen set this item came from (stratification key, never merged).
    pub source: String,
    /// Item category (stratification key, never merged).
    pub category: String,
    /// Gold slot key. `choice`: the option label; `score`: the level index as a
    /// decimal string; `noul`: `"true"` / `"false"`.
    pub gold: String,
    /// The item's declared slot keys, in the order the contract lists them.
    pub slots: Vec<String>,
    /// `argmax(probs)` when probabilities are present, else the row's `label`.
    pub predicted: Option<String>,
    /// Probability vector aligned to `slots`. `None` ⇒ row is unscored.
    pub probs: Option<Vec<f64>>,
    /// Slot key used as the positive class for `brier_binary`. `None` ⇒ the row
    /// is excluded from that one metric (so the binarisation is never arbitrary).
    pub positive: Option<String>,
}

impl Sample {
    fn gold_index(&self) -> Option<usize> {
        self.slots.iter().position(|slot| *slot == self.gold)
    }
}

/// Which slot wins the argmax. Ties go to the earliest slot, so the result is a
/// pure function of the vector — the same guarantee the server makes.
pub fn argmax_first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// One bin of the reliability curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReliabilityBin {
    pub lower: f64,
    pub upper: f64,
    pub n: usize,
    pub mean_confidence: f64,
    pub accuracy: f64,
}

/// Every number computed for one stratum. Nothing here is summed with any other
/// stratum's version of it.
///
/// `deny_unknown_fields`: a fabricated extra metric (say `"total_accuracy"`) must
/// fail to load rather than ride along inside a report that `verify` then passes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StratumMetrics {
    pub n_items: usize,
    /// Rows with neither a probability vector nor a label.
    pub n_missing: usize,
    /// Rows that produced a predicted slot.
    pub n_predicted: usize,
    /// Rows that produced a full probability vector.
    pub n_scored: usize,
    pub accuracy: Option<f64>,
    pub balanced_accuracy: Option<f64>,
    /// The classes balanced accuracy averaged over (i.e. the gold classes present).
    pub classes_in_gold: Vec<String>,
    pub per_class_recall: BTreeMap<String, f64>,
    pub nll: Option<f64>,
    pub brier_multiclass: Option<f64>,
    /// How many scored rows declared a positive class (the `brier_binary` denominator).
    pub n_binary: usize,
    pub brier_binary: Option<f64>,
    pub ece: Option<f64>,
    pub reliability: Vec<ReliabilityBin>,
}

#[derive(Debug, Clone, Copy)]
struct BinAcc {
    confidence_sum: f64,
    n: usize,
    correct: usize,
}

fn bin_of(confidence: f64, bins: usize) -> usize {
    let raw = (confidence * bins as f64).floor();
    if raw.is_nan() || raw < 0.0 {
        0
    } else {
        (raw as usize).min(bins - 1)
    }
}

/// Reduce one stratum's samples to [`StratumMetrics`].
///
/// Deterministic: same samples in, bit-identical numbers out. That is what lets
/// [`crate::verify`] re-derive a published report and compare it exactly.
///
/// Panics only if `bins == 0`, which [`crate::report::build_report`] rejects first.
pub fn summarize(samples: &[Sample], bins: usize) -> StratumMetrics {
    assert!(bins > 0, "bin count must be >= 1");

    let n_items = samples.len();
    let n_missing = samples
        .iter()
        .filter(|s| s.predicted.is_none() && s.probs.is_none())
        .count();
    let predicted: Vec<&Sample> = samples.iter().filter(|s| s.predicted.is_some()).collect();
    let n_predicted = predicted.len();

    let accuracy = if n_predicted == 0 {
        None
    } else {
        let hits = predicted
            .iter()
            .filter(|s| s.predicted.as_deref() == Some(s.gold.as_str()))
            .count();
        Some(hits as f64 / n_predicted as f64)
    };

    // Per-class recall over the classes that actually occur as gold.
    let mut tallies: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for s in &predicted {
        let entry = tallies.entry(s.gold.as_str()).or_insert((0, 0));
        entry.0 += 1;
        if s.predicted.as_deref() == Some(s.gold.as_str()) {
            entry.1 += 1;
        }
    }
    let per_class_recall: BTreeMap<String, f64> = tallies
        .iter()
        .map(|(class, (n, ok))| ((*class).to_string(), *ok as f64 / *n as f64))
        .collect();
    let classes_in_gold: Vec<String> = tallies.keys().map(|k| (*k).to_string()).collect();
    let balanced_accuracy = if per_class_recall.is_empty() {
        None
    } else {
        Some(per_class_recall.values().sum::<f64>() / per_class_recall.len() as f64)
    };

    let scored: Vec<(&Sample, &[f64])> = samples
        .iter()
        .filter_map(|s| {
            s.probs
                .as_deref()
                .filter(|p| p.len() == s.slots.len())
                .map(|p| (s, p))
        })
        .collect();
    let n_scored = scored.len();

    let mut nll_sum = 0.0;
    let mut brier_multiclass_sum = 0.0;
    let mut brier_binary_sum = 0.0;
    let mut n_binary = 0usize;
    let mut bin_accs = vec![
        BinAcc {
            confidence_sum: 0.0,
            n: 0,
            correct: 0,
        };
        bins
    ];

    for (sample, probs) in scored.iter().copied() {
        // `gold_index` is `Some` for every sample that came through
        // `load_predictions`; a `None` here would mean a slot set that does not
        // contain its own gold, which the loader rejects.
        let Some(gold_index) = sample.gold_index() else {
            continue;
        };

        nll_sum += -probs[gold_index].max(NLL_PROBABILITY_FLOOR).ln();

        let mut squared = 0.0;
        for (k, p) in probs.iter().enumerate() {
            let target = if k == gold_index { 1.0 } else { 0.0 };
            squared += (p - target) * (p - target);
        }
        brier_multiclass_sum += squared;

        if let Some(pos) = sample
            .positive
            .as_ref()
            .and_then(|key| sample.slots.iter().position(|slot| slot == key))
        {
            let target = if pos == gold_index { 1.0 } else { 0.0 };
            brier_binary_sum += (probs[pos] - target) * (probs[pos] - target);
            n_binary += 1;
        }

        let top = argmax_first_max(probs);
        let confidence = probs[top];
        let bin = bin_of(confidence, bins);
        bin_accs[bin].confidence_sum += confidence;
        bin_accs[bin].n += 1;
        if top == gold_index {
            bin_accs[bin].correct += 1;
        }
    }

    let mut reliability = Vec::new();
    let mut ece = 0.0;
    for (index, acc) in bin_accs.iter().enumerate() {
        if acc.n == 0 {
            continue;
        }
        let mean_confidence = acc.confidence_sum / acc.n as f64;
        let bin_accuracy = acc.correct as f64 / acc.n as f64;
        ece += (acc.n as f64 / n_scored as f64) * (bin_accuracy - mean_confidence).abs();
        reliability.push(ReliabilityBin {
            lower: index as f64 / bins as f64,
            upper: (index + 1) as f64 / bins as f64,
            n: acc.n,
            mean_confidence,
            accuracy: bin_accuracy,
        });
    }

    let divide = |sum: f64| -> Option<f64> {
        if n_scored == 0 {
            None
        } else {
            Some(sum / n_scored as f64)
        }
    };

    StratumMetrics {
        n_items,
        n_missing,
        n_predicted,
        n_scored,
        accuracy,
        balanced_accuracy,
        classes_in_gold,
        per_class_recall,
        nll: divide(nll_sum),
        brier_multiclass: divide(brier_multiclass_sum),
        n_binary,
        brier_binary: if n_binary == 0 {
            None
        } else {
            Some(brier_binary_sum / n_binary as f64)
        },
        // NOTE: `ece` above is already the item-weighted sum
        // `Σ_bins (n_bin / n_scored) · |acc_bin − conf_bin|` — i.e. it is already
        // normalised. It must NOT go through `divide()` as well: doing that
        // silently divides ECE by the item count a second time (a 4-item set
        // reported 0.125 instead of 0.5, a 2-item set 0.5 instead of 1.0), which
        // is exactly the kind of quietly-wrong calibration number this crate
        // exists to avoid.
        ece: if n_scored == 0 { None } else { Some(ece) },
        reliability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(gold: &str, probs: Option<[f64; 2]>, positive: Option<&str>) -> Sample {
        let slots = vec!["A".to_string(), "B".to_string()];
        let vector = probs.map(|p| vec![p[0], p[1]]);
        let predicted = vector.as_ref().map(|v| slots[argmax_first_max(v)].clone());
        Sample {
            id: format!("s-{gold}-{}", vector.is_some()),
            source: "unit".into(),
            category: "unit".into(),
            gold: gold.to_string(),
            slots,
            predicted,
            probs: vector,
            positive: positive.map(str::to_owned),
        }
    }

    /// The four-row example that `tests/ece_hand_computed.rs` pins end to end,
    /// with probabilities chosen to be exact in binary floating point.
    fn known_four() -> Vec<Sample> {
        vec![
            sample("A", Some([0.5, 0.5]), Some("A")),
            sample("A", Some([0.25, 0.75]), Some("A")),
            sample("B", Some([0.75, 0.25]), Some("A")),
            sample("A", Some([1.0, 0.0]), Some("A")),
        ]
    }

    #[test]
    fn bin_assignment_clamps_the_top_edge() {
        assert_eq!(bin_of(0.0, 10), 0);
        assert_eq!(bin_of(0.05, 10), 0);
        assert_eq!(bin_of(0.75, 10), 7);
        assert_eq!(bin_of(0.999, 10), 9);
        assert_eq!(bin_of(1.0, 10), 9);
        assert_eq!(bin_of(1.0, 7), 6);
        assert_eq!(bin_of(-0.25, 10), 0);
    }

    #[test]
    fn argmax_breaks_ties_towards_the_earlier_slot() {
        assert_eq!(argmax_first_max(&[0.5, 0.5]), 0);
        assert_eq!(argmax_first_max(&[0.2, 0.8]), 1);
        assert_eq!(argmax_first_max(&[0.3]), 0);
    }

    #[test]
    fn empty_stratum_has_no_numbers() {
        let m = summarize(&[], 10);
        assert_eq!(m.n_items, 0);
        assert_eq!(m.n_scored, 0);
        assert!(m.accuracy.is_none());
        assert!(m.balanced_accuracy.is_none());
        assert!(m.ece.is_none());
        assert!(m.reliability.is_empty());
    }

    #[test]
    fn the_known_four_row_example_reproduces_hand_computed_values() {
        let m = summarize(&known_four(), 10);
        assert_eq!(m.n_items, 4);
        assert_eq!(m.n_predicted, 4);
        assert_eq!(m.n_scored, 4);
        assert_eq!(m.n_missing, 0);
        assert_eq!(m.accuracy, Some(0.5));
        // Gold `A` appears 3x (rows 1, 2, 4) with 2 hits; gold `B` once with 0.
        assert_eq!(m.classes_in_gold, vec!["A".to_string(), "B".to_string()]);
        assert_eq!(m.per_class_recall["A"], 2.0 / 3.0);
        assert_eq!(m.per_class_recall["B"], 0.0);
        assert_eq!(m.balanced_accuracy, Some(1.0 / 3.0));
        // binary Brier (positive = A): 0.25 + 0.5625 + 0.5625 + 0.0 = 1.375 / 4
        assert_eq!(m.brier_binary, Some(0.34375));
        assert_eq!(m.n_binary, 4);
        // and multiclass is exactly twice that for two slots
        assert_eq!(m.brier_multiclass, Some(0.6875));
        // ECE = 1/4*|1-0.5| + 2/4*|0-0.75| + 1/4*|1-1| = 0.5, all exact in binary
        assert_eq!(m.ece, Some(0.5));
        assert_eq!(m.reliability.len(), 3);
        assert_eq!(
            m.reliability[0],
            ReliabilityBin {
                lower: 0.5,
                upper: 0.6,
                n: 1,
                mean_confidence: 0.5,
                accuracy: 1.0
            }
        );
        assert_eq!(
            m.reliability[1],
            ReliabilityBin {
                lower: 0.7,
                upper: 0.8,
                n: 2,
                mean_confidence: 0.75,
                accuracy: 0.0
            }
        );
        assert_eq!(
            m.reliability[2],
            ReliabilityBin {
                lower: 0.9,
                upper: 1.0,
                n: 1,
                mean_confidence: 1.0,
                accuracy: 1.0
            }
        );
        // NLL = (-ln0.5 - ln0.25 - ln0.25 - ln1.0) / 4
        let expected = (0.5f64.ln().abs() + 0.25f64.ln().abs() * 2.0) / 4.0;
        assert!((m.nll.unwrap() - expected).abs() < 1e-15);
    }

    #[test]
    fn multiclass_brier_is_exactly_twice_binary_for_two_slots() {
        // Holds for *any* two-slot row, which is why reporting both is not
        // redundant-but-confusing: it is a cheap consistency check.
        for (p0, gold) in [(0.1, "A"), (0.37, "B"), (0.5, "A"), (0.93, "B")] {
            let rows = vec![sample(gold, Some([p0, 1.0 - p0]), Some("A"))];
            let m = summarize(&rows, 10);
            let mc = m.brier_multiclass.unwrap();
            let bi = m.brier_binary.unwrap();
            assert!((mc - 2.0 * bi).abs() < 1e-15, "p0={p0} mc={mc} bi={bi}");
        }
    }

    #[test]
    fn a_perfect_forecast_has_zero_nll_and_zero_ece() {
        let rows = vec![
            sample("A", Some([1.0, 0.0]), Some("A")),
            sample("B", Some([0.0, 1.0]), Some("A")),
        ];
        let m = summarize(&rows, 10);
        assert_eq!(m.accuracy, Some(1.0));
        assert_eq!(m.balanced_accuracy, Some(1.0));
        assert_eq!(m.ece, Some(0.0));
        assert_eq!(m.nll.unwrap(), 0.0);
        assert_eq!(m.brier_multiclass, Some(0.0));
    }

    #[test]
    fn a_maximally_wrong_confident_forecast_saturates_ece_at_one() {
        let rows = vec![
            sample("A", Some([0.0, 1.0]), Some("A")),
            sample("B", Some([1.0, 0.0]), Some("A")),
        ];
        let m = summarize(&rows, 10);
        assert_eq!(m.accuracy, Some(0.0));
        assert_eq!(m.ece, Some(1.0));
        assert_eq!(m.brier_binary, Some(1.0));
    }

    #[test]
    fn the_nll_floor_bounds_an_impossible_gold() {
        // p_gold = 0 must not become +inf.
        let rows = vec![sample("A", Some([0.0, 1.0]), Some("A"))];
        let m = summarize(&rows, 10);
        let expected = -NLL_PROBABILITY_FLOOR.ln();
        assert!((m.nll.unwrap() - expected).abs() < 1e-9);
        assert!(m.nll.unwrap().is_finite());
    }

    #[test]
    fn rows_without_probabilities_are_excluded_from_nll_brier_and_ece() {
        let mut rows = known_four();
        let mut hard = sample("A", None, Some("A"));
        hard.predicted = Some("A".to_string());
        rows.push(hard);
        let m = summarize(&rows, 10);
        // Accuracy sees all five rows...
        assert_eq!(m.n_predicted, 5);
        assert_eq!(m.accuracy, Some(3.0 / 5.0));
        // ...but the probability metrics still see only the four scored ones.
        assert_eq!(m.n_scored, 4);
        assert_eq!(m.ece, Some(0.5));
        // positive = A: 0.25 + 0.5625 + 0.5625 + 0.0 = 1.375 / 4
        assert_eq!(m.brier_binary, Some(0.34375));
    }

    #[test]
    fn missing_rows_are_counted_not_silently_dropped() {
        let rows = vec![
            sample("A", Some([0.5, 0.5]), Some("A")),
            Sample {
                id: "no-answer".into(),
                source: "unit".into(),
                category: "unit".into(),
                gold: "A".into(),
                slots: vec!["A".into(), "B".into()],
                predicted: None,
                probs: None,
                positive: None,
            },
        ];
        let m = summarize(&rows, 10);
        assert_eq!(m.n_items, 2);
        assert_eq!(m.n_missing, 1);
        assert_eq!(m.n_predicted, 1);
        assert_eq!(m.n_scored, 1);
        assert_eq!(m.accuracy, Some(1.0));
    }

    #[test]
    fn a_bin_count_of_one_is_plain_calibration_gap() {
        let m = summarize(&known_four(), 1);
        // One bin, so everything is clamped into it: mean confidence
        // (0.5+0.75+0.75+1.0)/4 = 0.75, accuracy 0.5, ECE = |0.5 - 0.75|.
        assert_eq!(m.reliability.len(), 1);
        assert_eq!(m.reliability[0].lower, 0.0);
        assert_eq!(m.reliability[0].upper, 1.0);
        assert_eq!(m.reliability[0].n, 4);
        assert!((m.reliability[0].mean_confidence - 0.75).abs() < 1e-15);
        assert_eq!(m.reliability[0].accuracy, 0.5);
        assert!((m.ece.unwrap() - 0.25).abs() < 1e-15);
    }

    #[test]
    fn summarize_is_a_pure_function_of_its_input() {
        let rows = known_four();
        assert_eq!(summarize(&rows, 10), summarize(&rows, 10));
        let finer = summarize(&rows, 25);
        assert_ne!(finer, summarize(&rows, 10));
    }
}
