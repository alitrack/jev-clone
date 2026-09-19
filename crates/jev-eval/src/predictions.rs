//! The prediction file: one JSON object per line, `{id, label?, probabilities?}`.
//!
//! This is deliberately **narrower than the server contract**: it carries exactly
//! what the metrics consume, so the metrics never have to guess. `probabilities`
//! is keyed by the item's slot keys (see [`crate::items::Item::slot_labels`]), and
//! the key set must match the item exactly — a partial vector is an error, not a
//! reason to renormalise or to score a missing option as zero (`specs/M1.md` §5②).
//!
//! A row with neither `label` nor `probabilities` is legal and means "no answer";
//! it is counted in `n_missing` rather than dropped, because coverage is part of
//! the result.

use crate::error::EvalError;
use crate::items::ItemSet;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Largest tolerated deviation of `Σp` from 1. Matches the server's contract
/// tolerance (`specs/M0.md` §3).
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-6;

/// One row of the prediction file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Prediction {
    pub id: String,
    /// Hard label, for rows that have no probability vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Slot key -> probability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<BTreeMap<String, f64>>,
}

impl Prediction {
    /// A row with nothing to score.
    pub fn missing(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: None,
            probabilities: None,
        }
    }

    fn validate(&self, slots: &[String]) -> Result<(), String> {
        if let Some(label) = &self.label {
            if !slots.contains(label) {
                return Err(format!(
                    "label {label:?} is not one of this item's slots {slots:?}"
                ));
            }
        }
        let Some(probabilities) = &self.probabilities else {
            return Ok(());
        };
        if probabilities.is_empty() {
            return Err("`probabilities` is present but empty".to_string());
        }
        if probabilities.len() != slots.len() {
            return Err(format!(
                "`probabilities` has {} keys but the item declares {} slots {slots:?}",
                probabilities.len(),
                slots.len()
            ));
        }
        for slot in slots {
            if !probabilities.contains_key(slot) {
                return Err(format!("`probabilities` is missing slot {slot:?}"));
            }
        }
        let mut sum = 0.0;
        for (slot, value) in probabilities {
            if !value.is_finite() {
                return Err(format!("probability for slot {slot:?} is not finite"));
            }
            if *value < 0.0 {
                return Err(format!("probability for slot {slot:?} is negative"));
            }
            sum += value;
        }
        if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(format!(
                "probabilities sum to {sum} which is more than {PROBABILITY_SUM_TOLERANCE} away from 1"
            ));
        }
        // If both are present they must agree, otherwise the file has two
        // contradictory answers in it and we would silently pick one.
        if let Some(label) = &self.label {
            let ordered: Vec<(&String, &f64)> = {
                let mut pairs: Vec<(&String, &f64)> = probabilities.iter().collect();
                pairs.sort_by(|a, b| {
                    let ia = slots.iter().position(|s| s == a.0).unwrap_or(usize::MAX);
                    let ib = slots.iter().position(|s| s == b.0).unwrap_or(usize::MAX);
                    ia.cmp(&ib)
                });
                pairs
            };
            let mut best = &ordered[0];
            for pair in &ordered {
                if pair.1 > best.1 {
                    best = pair;
                }
            }
            if best.0 != label {
                return Err(format!(
                    "label {label:?} disagrees with the argmax of `probabilities` ({:?})",
                    best.0
                ));
            }
        }
        Ok(())
    }
}

/// Read + validate a prediction file against an item set.
///
/// Every row is checked against the item it names: an unknown id, a label or slot
/// key the item does not declare, a partial probability vector, a negative or
/// non-finite probability, or `Σp` outside `1 ± 1e-6` are all errors. Rows may be
/// absent entirely; the missing items simply show up as `n_missing` in the report.
pub fn load_predictions(path: &Path, set: &ItemSet) -> Result<Vec<Prediction>, EvalError> {
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|source| EvalError::Io {
        path: display.clone(),
        source,
    })?;
    let text = String::from_utf8(bytes).map_err(|_| EvalError::NotUtf8 {
        path: display.clone(),
    })?;

    let known: BTreeMap<&str, &crate::items::Item> = set
        .items
        .iter()
        .map(|item| (item.id.as_str(), item))
        .collect();

    let mut predictions = Vec::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for (index, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line_no = index + 1;
        let prediction: Prediction =
            serde_json::from_str(line).map_err(|err| EvalError::Json {
                path: display.clone(),
                line: line_no,
                message: err.to_string(),
            })?;
        let item = known.get(prediction.id.as_str()).ok_or_else(|| EvalError::Prediction {
            path: display.clone(),
            line: line_no,
            id: prediction.id.clone(),
            message: "no such item in the frozen item set".to_string(),
        })?;
        prediction
            .validate(&item.slot_labels())
            .map_err(|message| EvalError::Prediction {
                path: display.clone(),
                line: line_no,
                id: prediction.id.clone(),
                message,
            })?;
        if let Some(previous) = seen.insert(prediction.id.clone(), line_no) {
            return Err(EvalError::Prediction {
                path: display.clone(),
                line: line_no,
                id: prediction.id,
                message: format!("duplicate prediction (first seen on line {previous})"),
            });
        }
        predictions.push(prediction);
    }
    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots() -> Vec<String> {
        vec!["no".to_string(), "yes".to_string()]
    }

    fn row(label: Option<&str>, probs: Option<[f64; 2]>) -> Prediction {
        Prediction {
            id: "t-1".into(),
            label: label.map(str::to_owned),
            probabilities: probs.map(|p| {
                let s = slots();
                let mut map = BTreeMap::new();
                map.insert(s[0].clone(), p[0]);
                map.insert(s[1].clone(), p[1]);
                map
            }),
        }
    }

    #[test]
    fn a_well_formed_row_passes() {
        assert!(row(None, Some([0.3, 0.7])).validate(&slots()).is_ok());
        assert!(row(Some("yes"), None).validate(&slots()).is_ok());
        assert!(row(Some("yes"), Some([0.3, 0.7])).validate(&slots()).is_ok());
    }

    #[test]
    fn an_empty_row_is_legal_and_means_no_answer() {
        assert!(row(None, None).validate(&slots()).is_ok());
    }

    #[test]
    fn a_partial_vector_is_rejected_rather_than_renormalised() {
        let mut partial = row(None, Some([0.3, 0.7]));
        partial.probabilities.as_mut().unwrap().remove("no");
        let err = partial.validate(&slots()).unwrap_err();
        assert!(err.contains("1 keys but the item declares 2"), "{err}");
    }

    #[test]
    fn a_missing_slot_is_named_in_the_error() {
        let mut map = BTreeMap::new();
        map.insert("no".to_string(), 0.5);
        map.insert("other".to_string(), 0.5);
        let bad = Prediction {
            id: "t-1".into(),
            label: None,
            probabilities: Some(map),
        };
        let err = bad.validate(&slots()).unwrap_err();
        assert!(err.contains("missing slot \"yes\""), "{err}");
    }

    #[test]
    fn an_unnormalised_vector_is_rejected() {
        let err = row(None, Some([0.3, 0.8])).validate(&slots()).unwrap_err();
        assert!(err.contains("sum to 1.1"), "{err}");
    }

    #[test]
    fn negative_and_non_finite_probabilities_are_rejected() {
        assert!(row(None, Some([-0.1, 1.1]))
            .validate(&slots())
            .unwrap_err()
            .contains("negative"));
        assert!(row(None, Some([f64::NAN, 1.0]))
            .validate(&slots())
            .unwrap_err()
            .contains("not finite"));
    }

    #[test]
    fn a_label_that_contradicts_the_argmax_is_rejected() {
        let err = row(Some("no"), Some([0.2, 0.8]))
            .validate(&slots())
            .unwrap_err();
        assert!(err.contains("disagrees with the argmax"), "{err}");
    }

    #[test]
    fn a_label_outside_the_slots_is_rejected() {
        let err = row(Some("maybe"), None).validate(&slots()).unwrap_err();
        assert!(err.contains("not one of this item's slots"), "{err}");
    }
}
