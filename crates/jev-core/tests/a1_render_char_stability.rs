//! A4-1: character-for-character prompt assertions for all three question types.

use jev_core::{
    render_question, ChoiceQuestion, Entry, NoulCriteria, NoulQuestion, Question, ScoreQuestion,
};
use serde_json::{json, Value};

fn state() -> Entry {
    Value::String(
        "Order A-104: the customer was charged twice for a monthly subscription."
            .to_string(),
    )
}

#[test]
fn choice_prompt_is_char_stable() {
    let q = Question::Choice(ChoiceQuestion {
        instructions: Value::String("Does the customer request a refund?".to_string()),
        criteria: [
            ("no", Value::String("does not ask for money back".to_string())),
            ("yes", Value::Null),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect(),
    });
    let r = render_question(&state(), &q, "q").expect("renders");
    assert_eq!(
        r.prompt,
        "State:\n\
         Order A-104: the customer was charged twice for a monthly subscription.\n\
         \n\
         Question:\n\
         Does the customer request a refund?\n\
         \n\
         Choose exactly one option.\n\
         Options:\n\
         - A: no — does not ask for money back\n\
         - B: yes\n\
         \n\
         Answer:\n",
        "choice prompt must be exactly the M0 shape (eval hashes it)"
    );
    assert_eq!(
        r.slots,
        vec![("A".to_string(), "no".to_string()), ("B".to_string(), "yes".to_string())]
    );
    assert_eq!(r.labels, vec!["no".to_string(), "yes".to_string()]);
}

#[test]
fn score_prompt_is_char_stable() {
    let q = Question::Score(ScoreQuestion {
        instructions: Value::String("How severe is this issue?".to_string()),
        criteria: vec![
            Value::String("low: cosmetic confusion".to_string()),
            Value::String("medium: one duplicate charge".to_string()),
            Value::String("high: repeated billing failure".to_string()),
        ],
    });
    let r = render_question(&state(), &q, "q").expect("renders");
    assert_eq!(
        r.prompt,
        "State:\n\
         Order A-104: the customer was charged twice for a monthly subscription.\n\
         \n\
         Question:\n\
         How severe is this issue?\n\
         \n\
         Choose exactly one level.\n\
         Levels:\n\
         - A: 0: low: cosmetic confusion\n\
         - B: 1: medium: one duplicate charge\n\
         - C: 2: high: repeated billing failure\n\
         \n\
         Answer:\n",
        "score prompt must be exactly the M0 shape (eval hashes it)"
    );
    assert_eq!(
        r.slots,
        vec![
            ("A".to_string(), "0".to_string()),
            ("B".to_string(), "1".to_string()),
            ("C".to_string(), "2".to_string()),
        ]
    );
}

#[test]
fn noul_prompt_with_criteria_is_char_stable() {
    let q = Question::Noul(NoulQuestion {
        instructions: Value::String("Did support reply the same day?".to_string()),
        criteria: Some(NoulCriteria {
            r#true: Some(Value::String("replied the same day".to_string())),
            r#false: Some(Value::String("slower than that".to_string())),
        }),
    });
    let r = render_question(&state(), &q, "q").expect("renders");
    assert_eq!(
        r.prompt,
        "State:\n\
         Order A-104: the customer was charged twice for a monthly subscription.\n\
         \n\
         Question:\n\
         Did support reply the same day?\n\
         \n\
         Choose exactly one option.\n\
         Options:\n\
         - A: yes — replied the same day\n\
         - B: no — slower than that\n\
         \n\
         Answer:\n",
        "noul prompt must be exactly the M0 shape (eval hashes it)"
    );
    assert_eq!(
        r.slots,
        vec![("A".to_string(), "true".to_string()), ("B".to_string(), "false".to_string())]
    );
}

#[test]
fn noul_prompt_without_criteria_is_char_stable() {
    let q = Question::Noul(NoulQuestion {
        instructions: Value::String("Was the order refunded?".to_string()),
        criteria: None,
    });
    let r = render_question(&state(), &q, "q").expect("renders");
    assert_eq!(
        r.prompt,
        "State:\n\
         Order A-104: the customer was charged twice for a monthly subscription.\n\
         \n\
         Question:\n\
         Was the order refunded?\n\
         \n\
         Choose exactly one option.\n\
         Options:\n\
         - A: yes\n\
         - B: no\n\
         \n\
         Answer:\n"
    );
}

#[test]
fn object_state_and_null_descriptions_render_as_pretty_json() {
    let s = json!({ "order": "A-104", "items": [1, 2] });
    let q = Question::Choice(ChoiceQuestion {
        instructions: Value::String("Which item?".to_string()),
        criteria: [
            ("alpha", Value::Object(json!({"note": "first item"}).as_object().unwrap().clone())),
            ("beta", Value::Null),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect(),
    });
    let r = render_question(&s, &q, "q").expect("renders");
    // `concat!` instead of a `"\`-continued literal: continuation strips the
    // next line's leading whitespace, which would eat the pretty-JSON indent.
    assert_eq!(
        r.prompt,
        concat!(
            "State:\n",
            "{\n",
            "  \"items\": [\n",
            "    1,\n",
            "    2\n",
            "  ],\n",
            "  \"order\": \"A-104\"\n",
            "}\n",
            "\nQuestion:\n",
            "Which item?\n",
            "\nChoose exactly one option.\nOptions:\n",
            "- A: alpha — {\n",
            "  \"note\": \"first item\"\n",
            "}\n",
            "- B: beta\n",
            "\nAnswer:\n",
        ),
        "object state and object descriptions render via to_string_pretty; null is skipped"
    );
    let null_state = Value::Null;
    let r = render_question(&null_state, &q, "q").expect("renders");
    assert!(r.prompt.starts_with("State:\nnull\n\nQuestion:\n"), "null renders as literal `null`");
}

#[test]
fn question_id_is_never_in_the_prompt() {
    let q = Question::Noul(NoulQuestion {
        instructions: Value::String("Was it done?".to_string()),
        criteria: None,
    });
    let r = render_question(&state(), &q, "id-that-must-not-appear").expect("renders");
    assert!(
        !r.prompt.contains("id-that-must-not-appear"),
        "invariant §0.3: question ids never enter the prompt"
    );
}
