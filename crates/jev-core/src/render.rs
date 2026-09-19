//! Prompt rendering for the three primitives.
//!
//! **Why letter slots.** The model's distribution is read at the single token
//! position that follows `Answer:`. We therefore need each option to be named by
//! exactly one token. Natural-language labels are not safe: on the live
//! qwen3.8-27b endpoint, `"yes"`, `"Yes"` and `" yes"` are three *different*
//! tokens, so `- yes` / `- no` would split the probability mass across variants
//! and silently mis-rank options. Uppercase letters `A`, `B`, `C`, … are single
//! tokens with no casing or whitespace variants, so they are used as slots, while
//! the human-readable label stays in the prompt body.
//!
//! Rendering shape (must stay stable; the eval harness hashes prompts):
//!
//! ```text
//! State:
//! <state rendered: string as-is, object/array as pretty JSON>
//!
//! Question:
//! <instructions rendered the same way>
//!
//! Choose exactly one option.
//! Options:
//! - A: <label 1> — <description, if any>
//! - B: <label 2>
//!
//! Answer:
//! ```
//!
//! Score uses `- A: 0: <level>` ordering `0..n-1` and asks for the level letter.
//! Noul renders the optional yes/no clarifications and asks for `A` (yes) or
//! `B` (no).
//!
//! **M0 letter-slot limit.** M0 uses one uppercase letter per slot, so at most 26
//! options can be named. `choice` legally allows up to 255, so a `choice` with
//! more than 26 options is **refused explicitly** (no silent truncation, no
//! silently wrong readout) via [`CoreError::TooManyLetterSlots`] — the dedicated
//! "too many options" error of `CoreError` is reserved for the public contract
//! limit (255). Multi-letter slots (e.g. `AA`, `AB`, …) are the planned M1
//! extension and must be added as a new, separately-eval'd rendering variant.

use crate::contract::{Entry, Question};
use crate::error::CoreError;
use serde_json::Value;

/// Maximum number of options nameable with one uppercase letter in M0.
pub const MAX_LETTER_SLOTS: usize = 26;

/// A rendered question: the prompt text plus the ordered slot -> option-label map.
///
/// `slots[i]` is the caller-facing value that a probability maps to:
/// for `choice` it is the criteria key, for `score` the level index as a string,
/// for `noul` it is `"true"` then `"false"`.
///
/// `labels[i]` is the *human-readable* side of the same option: the criteria key
/// for `choice`, the level **description** for `score` (which is what `legend`
/// reports), and `"yes"`/`"no"` for `noul`.

#[derive(Debug, Clone, PartialEq)]
pub struct RenderedQuestion {
    pub prompt: String,
    /// slot letter ("A".."Z") -> value reported back to the caller.
    pub slots: Vec<(String, String)>,
    /// human-readable label of each option, in the same order (for logs/eval only).
    pub labels: Vec<String>,
}

/// What a per-type renderer produces: the question section of the prompt, the
/// ordered `slot letter -> caller-facing value` map, and the human-readable labels.
type QuestionBody = (String, Vec<(String, String)>, Vec<String>);

/// Render one question against a shared, already-rendered state.
pub fn render_question(
    state: &Entry,
    question: &Question,
    question_id: &str,
) -> Result<RenderedQuestion, CoreError> {
    let (question_lines, slots, labels) = match question {
        Question::Choice(c) => render_choice(c, question_id),
        Question::Score(s) => render_score(s, question_id),
        Question::Noul(n) => render_noul(n),
    }?;

    let mut prompt = String::new();
    prompt.push_str("State:\n");
    prompt.push_str(&render_entry(state));
    prompt.push_str("\n\nQuestion:\n");
    prompt.push_str(&render_entry(instructions_of(question)));
    prompt.push_str("\n\n");
    prompt.push_str(&question_lines);
    prompt.push_str("\n\nAnswer:\n");

    Ok(RenderedQuestion { prompt, slots, labels })
}

/// Render an `Entry` for inclusion in the prompt: strings as-is, JSON pretty-printed.
pub fn render_entry(entry: &Entry) -> String {
    match entry {
        // The string carries its own text; re-encoding would add quotes and
        // escape characters that change the prompt the model sees.
        Value::String(s) => s.clone(),
        // `null`, numbers, objects and arrays are all canonicalized as pretty
        // JSON so the rendered text is deterministic for a given input.
        other => serde_json::to_string_pretty(other)
                    .expect("a Value is always serializable"),
    }
}

/// The `instructions` of whichever question type `q` is.
///
/// A private helper rather than an inherent method on `Question`: `contract.rs` is
/// frozen for M0, and adding public API to the contract types is out of scope.
fn instructions_of(q: &Question) -> &Entry {
    match q {
        Question::Choice(c) => &c.instructions,
        Question::Score(s) => &s.instructions,
        Question::Noul(n) => &n.instructions,
    }
}

fn render_choice(
    c: &crate::contract::ChoiceQuestion,
    id: &str,
) -> Result<QuestionBody, CoreError> {
    let n = c.criteria.len();
    if n < 2 {
        return Err(CoreError::TooFewChoiceOptions { id: id.to_string(), got: n });
    }
    if n > crate::contract::MAX_CHOICE_OPTIONS {
        return Err(CoreError::TooManyChoiceOptions {
            id: id.to_string(),
            got: n,
            max: crate::contract::MAX_CHOICE_OPTIONS,
        });
    }
    if n > MAX_LETTER_SLOTS {
        // Explicit refusal: one letter names at most 26 slots, and we never
        // silently drop or merge options. See the module doc.
        return Err(CoreError::TooManyLetterSlots { id: id.to_string(), got: n, max: MAX_LETTER_SLOTS });
    }

    // `criteria` is a BTreeMap, so iteration is in dictionary order of the keys:
    // the slot assignment (A, B, C, …) is stable for a given question object.
    let mut body = String::from("Choose exactly one option.\nOptions:");
    let mut slots = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for (i, (label, desc)) in c.criteria.iter().enumerate() {
        let letter = letter_at(i);
        let mut line = format!("- {letter}: {label}");
        if !desc.is_null() {
            line.push_str(" — ");
            line.push_str(&render_entry(desc));
        }
        body.push('\n');
        body.push_str(&line);
        slots.push((letter.to_string(), label.clone()));
        labels.push(label.clone());
    }
    Ok((body, slots, labels))
}

fn render_score(
    s: &crate::contract::ScoreQuestion,
    id: &str,
) -> Result<QuestionBody, CoreError> {
    let n = s.criteria.len();
    if n < 2 {
        return Err(CoreError::TooFewScoreLevels { id: id.to_string(), got: n });
    }
    if n > MAX_LETTER_SLOTS {
        return Err(CoreError::TooManyLetterSlots { id: id.to_string(), got: n, max: MAX_LETTER_SLOTS });
    }

    let mut body = String::from("Choose exactly one level.\nLevels:");
    let mut slots = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for (i, level) in s.criteria.iter().enumerate() {
        let letter = letter_at(i);
        let description = render_entry(level);
        body.push('\n');
        body.push_str(&format!("- {letter}: {i}: {description}"));
        slots.push((letter.to_string(), i.to_string()));
        // `labels` is the human-readable side for logs/legend: for `score` that is
        // the level *description*, while the caller-facing value (the index string
        // `"0"`) lives in `slots`. Storing the index string here (the first cut)
        // forced every consumer to re-read the caller's raw `criteria`.
        labels.push(description);
    }
    Ok((body, slots, labels))
}

fn render_noul(
    n: &crate::contract::NoulQuestion,
) -> Result<QuestionBody, CoreError> {
    // The letters MUST appear in the prompt. The readout is the distribution at
    // the single token following `Answer:`, so the model has to be *asked* for a
    // letter. M0's first cut rendered only `- yes:` / `- no:` — `A`/`B` were
    // mentioned nowhere, the live model answered "Yes", and the readout failed
    // with `slot "A" missing from the backend's top-10 logprobs`. Only a run
    // against the real endpoint could catch that: every stub in the test suite is
    // queried with a prompt the test itself wrote. Same header as `choice`, so all
    // three primitives present the model with one shape.
    let mut body = String::from("Choose exactly one option.\nOptions:");
    let yes = n.criteria.as_ref().and_then(|c| c.r#true.as_ref());
    let no = n.criteria.as_ref().and_then(|c| c.r#false.as_ref());

    let mut line = String::from("- A: yes");
    if let Some(t) = yes {
        line.push_str(" — ");
        line.push_str(&render_entry(t));
    }
    body.push('\n');
    body.push_str(&line);

    let mut line = String::from("- B: no");
    if let Some(f) = no {
        line.push_str(" — ");
        line.push_str(&render_entry(f));
    }
    body.push('\n');
    body.push_str(&line);

    let slots = vec![
        ("A".to_string(), "true".to_string()),
        ("B".to_string(), "false".to_string()),
    ];
    let labels = vec!["yes".to_string(), "no".to_string()];
    Ok((body, slots, labels))
}

/// The slot letter for slot index `i` (0 -> 'A'). Only valid for `i < 26`,
/// which every caller has checked against [`MAX_LETTER_SLOTS`] beforehand.
fn letter_at(i: usize) -> char {
    (b'A' + i as u8) as char
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice_q() -> Question {
        Question::Choice(crate::contract::ChoiceQuestion {
            instructions: Value::String("Does the customer request a refund?".into()),
            criteria: [
                ("no", json!("does not ask for money back")),
                ("yes", json!(null)),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
        })
    }

    #[test]
    fn choice_prompt_is_exactly_the_spec_shape() {
        let state = Value::String("Customer paid twice.".into());
        let r = render_question(&state, &choice_q(), "q").unwrap();
        assert_eq!(
            r.prompt,
            "State:\n\
             Customer paid twice.\n\
             \n\
             Question:\n\
             Does the customer request a refund?\n\
             \n\
             Choose exactly one option.\n\
             Options:\n\
             - A: no — does not ask for money back\n\
             - B: yes\n\
             \n\
             Answer:\n"
        );
        assert_eq!(r.slots, vec![("A".to_string(), "no".to_string()), ("B".to_string(), "yes".to_string())]);
        assert_eq!(r.labels, vec!["no".to_string(), "yes".to_string()]);
    }

    #[test]
    fn score_and_noul_prompts_are_exactly_the_spec_shape() {
        let state = Value::Null;
        let score = Question::Score(crate::contract::ScoreQuestion {
            instructions: Value::String("rate".into()),
            criteria: vec![Value::String("low".into()), Value::String("high".into())],
        });
        let r = render_question(&state, &score, "q").unwrap();
        assert_eq!(
            r.prompt,
            "State:\n\
             null\n\
             \n\
             Question:\n\
             rate\n\
             \n\
             Choose exactly one level.\n\
             Levels:\n\
             - A: 0: low\n\
             - B: 1: high\n\
             \n\
             Answer:\n"
        );
        assert_eq!(
            r.slots,
            vec![("A".to_string(), "0".to_string()), ("B".to_string(), "1".to_string())]
        );

        let noul = Question::Noul(crate::contract::NoulQuestion {
            instructions: Value::String("Did support reply the same day?".into()),
            criteria: Some(crate::contract::NoulCriteria {
                r#true: Some(Value::String("replied the same day".into())),
                r#false: Some(Value::String("slower than that".into())),
            }),
        });
        let r = render_question(&state, &noul, "q").unwrap();
        assert_eq!(
            r.prompt,
            "State:\n\
             null\n\
             \n\
             Question:\n\
             Did support reply the same day?\n\
             \n\
             Choose exactly one option.\n\
             Options:\n\
             - A: yes — replied the same day\n\
             - B: no — slower than that\n\
             \n\
             Answer:\n"
        );
        assert_eq!(
            r.slots,
            vec![("A".to_string(), "true".to_string()), ("B".to_string(), "false".to_string())]
        );
    }
}
