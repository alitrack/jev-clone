//! A4-6: `score_expectation` — Σ i * p_i, including the one-hot identity.

use jev_core::score_expectation;

#[test]
fn one_hot_at_level_k_equals_k() {
    // The spec's headline case: a one-hot at the k-th level must give exactly k.
    for k in 0..5 {
        let mut p = vec![0.0_f64; 5];
        p[k] = 1.0;
        let got = score_expectation(&p);
        assert!((got - k as f64).abs() <= 1e-12, "one-hot at {k} must equal {k}, got {got}");
    }
}

#[test]
fn weighted_average_of_levels() {
    // [0.2, 0.3, 0.5] -> 0*0.2 + 1*0.3 + 2*0.5 = 1.3
    assert!((score_expectation(&[0.2, 0.3, 0.5]) - 1.3).abs() <= 1e-12);

    // uniform over 3 levels -> (0+1+2)/3 = 1 (the middle level)
    let u = score_expectation(&[1.0 / 3.0; 3]);
    assert!((u - 1.0).abs() <= 1e-12);
}

#[test]
fn degenerate_vectors() {
    // empty -> 0 (no levels to weight)
    assert_eq!(score_expectation(&[]), 0.0);
    // single level always 0
    assert_eq!(score_expectation(&[1.0]), 0.0);
    // all mass on the last of four levels -> 3
    assert!((score_expectation(&[0.0, 0.0, 0.0, 1.0]) - 3.0).abs() <= 1e-12);
}
