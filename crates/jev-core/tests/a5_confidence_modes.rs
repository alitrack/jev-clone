//! A4-5: `confidence_from_probabilities` across the three modes.

use jev_core::{confidence_from_probabilities, ConfidenceMode};

#[test]
fn one_hot_gives_full_confidence_in_all_modes() {
    // one-hot at index 1, three options.
    let p = [0.0, 1.0, 0.0];
    assert!((confidence_from_probabilities(&p, ConfidenceMode::NormalizedEntropy) - 1.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&p, ConfidenceMode::Margin) - 1.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&p, ConfidenceMode::Top1) - 1.0).abs() <= 1e-12);

    // single-option vector: n == 1 special case.
    assert!((confidence_from_probabilities(&[1.0], ConfidenceMode::NormalizedEntropy) - 1.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&[1.0], ConfidenceMode::Margin) - 1.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&[1.0], ConfidenceMode::Top1) - 1.0).abs() <= 1e-12);
}

#[test]
fn uniform_distribution_gives_zero_confidence() {
    let p4 = [0.25, 0.25, 0.25, 0.25];
    assert!((confidence_from_probabilities(&p4, ConfidenceMode::NormalizedEntropy) - 0.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&p4, ConfidenceMode::Margin) - 0.0).abs() <= 1e-12);
    // Top1 of a uniform is 1/n, which is NOT zero — only the spread-based
    // modes collapse to 0. Document that explicitly.
    assert!((confidence_from_probabilities(&p4, ConfidenceMode::Top1) - 0.25).abs() <= 1e-12);

    let p2 = [0.5, 0.5];
    assert!((confidence_from_probabilities(&p2, ConfidenceMode::NormalizedEntropy) - 0.0).abs() <= 1e-12);
    assert!((confidence_from_probabilities(&p2, ConfidenceMode::Margin) - 0.0).abs() <= 1e-12);
}

#[test]
fn margin_is_monotonic_in_peakedness() {
    // As the distribution becomes more peaked, Margin must not decrease.
    let flat = [0.34, 0.33, 0.33];
    let mid = [0.50, 0.25, 0.25];
    let peaked = [0.80, 0.10, 0.10];
    let m_flat = confidence_from_probabilities(&flat, ConfidenceMode::Margin);
    let m_mid = confidence_from_probabilities(&mid, ConfidenceMode::Margin);
    let m_peaked = confidence_from_probabilities(&peaked, ConfidenceMode::Margin);
    assert!(m_flat <= m_mid && m_mid <= m_peaked, "Margin must be monotone: {m_flat} {m_mid} {m_peaked}");

    // NormalizedEntropy and Top1 are monotone in the same direction.
    for mode in [ConfidenceMode::NormalizedEntropy, ConfidenceMode::Top1] {
        let c_flat = confidence_from_probabilities(&flat, mode);
        let c_mid = confidence_from_probabilities(&mid, mode);
        let c_peaked = confidence_from_probabilities(&peaked, mode);
        assert!(c_flat <= c_mid && c_mid <= c_peaked, "{mode:?} must be monotone: {c_flat} {c_mid} {c_peaked}");
    }
}

#[test]
fn confidence_is_always_in_unit_interval() {
    let cases: &[&[f64]] = &[
        &[0.0, 0.0, 1.0],
        &[0.7, 0.1, 0.2],
        &[1.0, 0.0, 0.0],
        &[0.1, 0.1, 0.1, 0.1, 0.6],
        &[0.0, 1.0],
    ];
    for p in cases {
        for mode in [ConfidenceMode::NormalizedEntropy, ConfidenceMode::Margin, ConfidenceMode::Top1] {
            let c = confidence_from_probabilities(p, mode);
            assert!(c.is_finite() && (0.0..=1.0).contains(&c), "confidence for {p:?} in {mode:?} = {c}");
        }
    }
}

#[test]
fn entropy_matches_closed_form() {
    // p = [0.5, 0.5] over 2 options: H = ln 2, normalized = 0.
    // p = [0.9, 0.1]: H = -(0.9 ln0.9 + 0.1 ln0.1).
    let p = [0.9_f64, 0.1_f64];
    let h = -p[0] * p[0].ln() - p[1] * p[1].ln();
    let expected = 1.0 - h / 2_f64.ln();
    let got = confidence_from_probabilities(&p, ConfidenceMode::NormalizedEntropy);
    assert!((got - expected).abs() <= 1e-12, "entropy closed form: got {got}, want {expected}");
}
