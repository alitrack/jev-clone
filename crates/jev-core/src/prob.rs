//! Probability and confidence math.
//!
//! Two facts drive this module:
//!
//! * The distribution we return is **conditional on the options the caller
//!   supplied**: we softmax over the declared answer slots only, then renormalize.
//!   It is therefore an *option score*, not a calibrated decision confidence
//!   (the reference implementations label the same quantity the same way).
//! * The upstream `confidence` formula is not published — the vendor only says it
//!   is "derived from probabilities" and that callers are not locked into their
//!   definition. So we ship several well-defined candidates and pick the default by
//!   expected calibration error on the frozen evaluation set (see docs/design.md
//!   §2.4), rather than copying an approximation from a third party.

/// Candidate definitions of `confidence`. All map a probability vector to `[0, 1]`
/// where higher means more peaked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfidenceMode {
    /// `1 - H(p) / log(n)`: 1.0 when one option has all the mass, 0.0 when uniform.
    /// This is the shape "spread out vs concentrated" as a single number.
    #[default]
    NormalizedEntropy,
    /// `p(top1) - p(top2)`: how far clear the winner is. Not order-invariant in the
    /// same way as entropy, but easier to threshold.
    Margin,
    /// `p(top1)`.
    Top1,
}

/// Numerically stable softmax (subtract the max). Returns `1/n` for an empty or
/// all-`-inf` input instead of producing NaN.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let n = logits.len();
    if n == 0 {
        return Vec::new();
    }
    let finite: Vec<f64> = logits.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        // All inputs are -inf (or NaN): no logits carry any evidence, so fall
        // back to the flat prior rather than a NaN distribution.
        return vec![1.0 / n as f64; n];
    }
    let m = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|x| x - m).map(|y| y.exp()).sum();
    if sum == 0.0 || !sum.is_finite() {
        return vec![1.0 / n as f64; n];
    }
    logits.iter().map(|x| (x - m).exp() / sum).collect()
}

/// Probability-weighted position across ordered levels: `Σ i * p_i`.
/// Used by Score answers, which may land between levels.
pub fn score_expectation(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| (i as f64) * p)
        .sum()
}

/// Compute `confidence` for one answer's probability vector.
pub fn confidence_from_probabilities(probabilities: &[f64], mode: ConfidenceMode) -> f64 {
    let n = probabilities.len();
    if n == 0 {
        return 0.0;
    }

    match mode {
        ConfidenceMode::NormalizedEntropy => {
            if n == 1 {
                return 1.0;
            }
            let h: f64 = probabilities
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();
            clamp01(1.0 - h / (n as f64).ln())
        }
        ConfidenceMode::Margin => {
            let mut sorted: Vec<f64> = probabilities.to_vec();
            sorted.sort_by(|a, b| b.total_cmp(a));
            let top1 = sorted[0];
            let top2 = sorted.get(1).copied().unwrap_or(0.0);
            clamp01(top1 - top2)
        }
        ConfidenceMode::Top1 => probabilities.iter().cloned().fold(0.0, f64::max),
    }
}

/// Clamp a float into `[0, 1]` to absorb float error at the edges.
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}
