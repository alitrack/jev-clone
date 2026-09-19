//! A4-4: `softmax` — normalization, extremes without NaN, uniform input.

use jev_core::softmax;

#[test]
fn softmax_sums_to_one() {
    for logits in [
        vec![0.0, 1.0, 2.0],
        vec![-12.5, 0.1, 3.7, -8.2],
        vec![1000.0, 1001.0, 1002.0], // large values: must not overflow
        vec![0.0, 0.0, 0.0],
        vec![5.0],
    ] {
        let p = softmax(&logits);
        assert_eq!(p.len(), logits.len());
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() <= 1e-9,
            "softmax must normalize to 1 for {logits:?}, got sum={sum}"
        );
        for &x in &p {
            assert!(x.is_finite() && x >= 0.0, "probabilities must be finite and non-negative");
        }
    }
}

#[test]
fn softmax_is_stable_for_extremes_and_all_minus_inf() {
    // Huge spread: the max-subtraction must keep this finite and sensible.
    let p = softmax(&[1e308, -1e308]);
    assert!(p.iter().all(|x| x.is_finite()), "extreme logits must not produce NaN/inf");
    assert!(p[0] > 0.999, "the dominant logit should carry ~all the mass");

    // All -inf: the spec mandates a uniform fallback, never NaN.
    let p = softmax(&[f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY]);
    assert!(p.iter().all(|x| x.is_finite()), "all -inf input must not yield NaN");
    for &x in &p {
        assert!((x - 1.0 / 3.0).abs() <= 1e-12, "all -inf must fall back to 1/n, got {x}");
    }

    // Empty input: uniform over zero elements is just empty (no panic, no NaN).
    let p = softmax(&[]);
    assert!(p.is_empty());
}

#[test]
fn softmax_uniform_input_gives_one_over_n() {
    let p = softmax(&[0.0, 0.0, 0.0, 0.0, 0.0]);
    for &x in &p {
        assert!((x - 0.2).abs() <= 1e-12, "equal logits must give 1/n, got {x}");
    }

    // Any constant offset must give the same uniform distribution.
    let q = softmax(&[7.5, 7.5, 7.5]);
    for &x in &q {
        assert!((x - 1.0 / 3.0).abs() <= 1e-12);
    }
}

#[test]
fn softmax_is_order_consistent() {
    // Higher logit -> higher probability, strictly.
    let p = softmax(&[1.0, 2.0, 3.0]);
    assert!(p[0] < p[1] && p[1] < p[2]);
}
