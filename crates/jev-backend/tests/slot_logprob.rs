//! `Readout::slot_logprob` — the slot-matching rule (specs/M0.md §4 B2/B5.1).
//!
//! The rule is: every key whose `trim()` equals the slot letter denotes the *same*
//! slot (on the live endpoint `"A"` and `" A"` are two distinct tokens that both
//! mean "the model answered A"), so their log-probabilities are **added in log
//! space** — never compared. Getting this wrong (taking the max) silently
//! under-reports the probability of the leading-space variant, which is common on
//! real endpoints.

use jev_backend::Readout;
use std::collections::BTreeMap;

fn readout(pairs: &[(&str, f64)]) -> Readout {
    Readout {
        top: pairs
            .iter()
            .map(|(t, l)| ((*t).to_string(), *l))
            .collect::<BTreeMap<_, _>>(),
        prompt_tokens: 0,
        completion_tokens: 1,
    }
}

/// `log(e^a + e^b)`, computed naively: the expected value, independent of the
/// implementation's own log-sum-exp helper.
fn log_add(a: f64, b: f64) -> f64 {
    (a.exp() + b.exp()).ln()
}

#[test]
fn slot_present_without_leading_space_matches() {
    let r = readout(&[("A", -0.25), ("B", -2.0)]);
    let got = r.slot_logprob("A").expect("A is in the top-k");
    assert!((got - -0.25).abs() < 1e-12, "got {got}");
}

#[test]
fn slot_present_with_leading_space_matches() {
    // The tokenizer surfaces the answer as " A" (leading space) — still slot A.
    let r = readout(&[(" A", -1.75), (" B", -2.0)]);
    let got = r.slot_logprob("A").expect("` A` is the same slot as `A`");
    assert!((got - -1.75).abs() < 1e-12, "got {got}");
}

#[test]
fn both_surface_forms_are_logsumexp_added_not_maximized() {
    let a = -0.5_f64;
    let b = -2.0_f64;
    let r = readout(&[("A", a), (" A", b), ("B", -4.0)]);

    let got = r.slot_logprob("A").expect("A is in the top-k");
    let expected = log_add(a, b);

    assert!(
        (got - expected).abs() < 1e-12,
        "slot_logprob must be log-sum-exp({a}, {b}) = {expected}, got {got}"
    );
    // The point of the test: *not* the max. `max(a, b)` is -0.5, so a max-based
    // implementation would land exactly on `a`.
    assert!(
        got > a + 1e-9,
        "got {got}; a max-based implementation would return {a} (the larger logprob)"
    );
    assert!(
        (got - a).abs() > 1e-6,
        "got {got} — indistinguishable from max(a, b) = {a}"
    );
}

#[test]
fn all_three_surface_forms_are_added() {
    let xs = [-1.0_f64, -1.25, -2.5];
    let r = readout(&[("A", xs[0]), (" A", xs[1]), ("  A  ", xs[2]), ("Z", -9.0)]);
    let got = r.slot_logprob("A").expect("A is in the top-k");
    // log-sum of three, assembled from the two-term helper.
    let expected = log_add(log_add(xs[0], xs[1]), xs[2]);
    assert!((got - expected).abs() < 1e-12, "got {got}, expected {expected}");
}

#[test]
fn slot_absent_returns_none() {
    let r = readout(&[("B", -0.1), (" C", -1.0), ("yes", -3.0)]);
    assert_eq!(r.slot_logprob("A"), None);
}

#[test]
fn matching_is_case_sensitive_and_exact() {
    // `"a"` / `" a"` are *different tokens* from `"A"` on the live endpoint (they
    // are exactly the failure mode letter slots exist to avoid): they must never
    // be counted as slot A.
    let r = readout(&[("a", -0.1), (" a", -0.2), ("A.", -0.3)]);
    assert_eq!(r.slot_logprob("A"), None);
}

#[test]
fn empty_readout_returns_none() {
    let r = readout(&[]);
    assert_eq!(r.slot_logprob("A"), None);
}
