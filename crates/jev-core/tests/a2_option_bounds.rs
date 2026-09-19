//! A4-2: option-count bounds for choice and score (and the M0 letter-slot cap).

use jev_core::{render_question, ChoiceQuestion, CoreError, Question, ScoreQuestion};
use serde_json::{json, Value};

fn state() -> Value {
    Value::String("state".to_string())
}

fn choice_with(n: usize, desc: Option<Value>) -> Question {
    let criteria = (0..n)
        .map(|i| (format!("opt-{i:02}"), desc.clone().unwrap_or(Value::Null)))
        .collect();
    Question::Choice(ChoiceQuestion {
        instructions: Value::String("pick one".to_string()),
        criteria,
    })
}

fn score_with(n: usize) -> Question {
    Question::Score(ScoreQuestion {
        instructions: Value::String("rate".to_string()),
        criteria: (0..n).map(|i| Value::String(format!("level {i}"))).collect(),
    })
}

#[test]
fn choice_with_one_option_is_rejected() {
    let err = render_question(&state(), &choice_with(1, None), "q").unwrap_err();
    assert_eq!(err, CoreError::TooFewChoiceOptions { id: "q".into(), got: 1 });
}

#[test]
fn choice_with_zero_options_is_rejected() {
    let err = render_question(&state(), &choice_with(0, None), "q").unwrap_err();
    assert_eq!(err, CoreError::TooFewChoiceOptions { id: "q".into(), got: 0 });
}

#[test]
fn choice_with_256_options_is_rejected_by_the_contract_limit() {
    let err = render_question(&state(), &choice_with(256, None), "q").unwrap_err();
    assert_eq!(
        err,
        CoreError::TooManyChoiceOptions { id: "q".into(), got: 256, max: 255 }
    );
}

#[test]
fn choice_26_options_ok_but_27_is_refused_explicitly() {
    // 26 is the M0 letter-slot maximum and must render.
    let r = render_question(&state(), &choice_with(26, None), "q").expect("26 options render");
    assert!(r.prompt.ends_with("- Z: opt-25\n\nAnswer:\n"));
    assert_eq!(r.slots.len(), 26);
    assert_eq!(r.slots[25], ("Z".to_string(), "opt-25".to_string()));

    // 27 > 26 letters: refused explicitly, never silently truncated.
    // M1 gave this its own variant instead of borrowing `PromptTooLong` (which
    // would have claimed a token-budget overflow that never happened).
    let err = render_question(&state(), &choice_with(27, None), "q").unwrap_err();
    assert_eq!(
        err,
        CoreError::TooManyLetterSlots { id: "q".into(), got: 27, max: 26 }
    );
}

#[test]
fn choice_with_null_descriptions_renders_bare_label_lines() {
    let q = choice_with(2, None);
    let r = render_question(&state(), &q, "q").expect("renders");
    assert!(r.prompt.contains("- A: opt-00\n- B: opt-01\n"), "null criteria render bare");
}

#[test]
fn score_with_one_level_is_rejected() {
    let err = render_question(&state(), &score_with(1), "q").unwrap_err();
    assert_eq!(err, CoreError::TooFewScoreLevels { id: "q".into(), got: 1 });
}

#[test]
fn score_with_zero_levels_is_rejected() {
    let err = render_question(&state(), &score_with(0), "q").unwrap_err();
    assert_eq!(err, CoreError::TooFewScoreLevels { id: "q".into(), got: 0 });
}

#[test]
fn score_with_27_levels_is_refused_explicitly() {
    let err = render_question(&state(), &score_with(27), "q").unwrap_err();
    assert_eq!(
        err,
        CoreError::TooManyLetterSlots { id: "q".into(), got: 27, max: 26 }
    );
}

#[test]
fn two_option_choice_renders() {
    let r = render_question(&state(), &choice_with(2, Some(json!("a description"))), "q").expect("renders");
    assert!(r.prompt.contains("- A: opt-00 — a description\n"));
    assert!(r.prompt.contains("- B: opt-01 — a description\n"));
}
