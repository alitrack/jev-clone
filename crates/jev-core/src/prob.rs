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

/// Numerically stable softmax. Returns `1/n` for an empty or all-`-inf` input.
pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let _ = logits;
    todo!("worker A: implement per the module doc + specs/M0.md §C")
}

/// Probability-weighted position across ordered levels: `Σ i * p_i`.
/// Used by Score answers, which may land between levels.
pub fn score_expectation(probabilities: &[f64]) -> f64 {
    let _ = probabilities;
    todo!("worker A")
}

/// Compute `confidence` for one answer's probability vector.
pub fn confidence_from_probabilities(probabilities: &[f64], mode: ConfidenceMode) -> f64 {
    let _ = (probabilities, mode);
    todo!("worker A")
}
