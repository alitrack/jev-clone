//! `import` — the minimal "we can already produce an answer, now score it" path.
//!
//! The serving stack already emits a `SystemOneResponse`-shaped body
//! (`{model, answers: {<id>: Answer}, usage}`). This module turns that into the
//! evaluation's prediction JSONL, without calling any model: no weights, no
//! network, no endpoint. That is deliberate — the orchestrator owns model calls;
//! this crate owns turning a result into a number.
//!
//! Conversion, per question kind (`specs/M0.md` §4 B4 is the contract side):
//!
//! | item kind | prediction `probabilities` | why |
//! |---|---|---|
//! | `choice` | `Answer::Choice.probabilities` verbatim (label -> p) | same keys by contract |
//! | `score` | `Answer::Score.probabilities` verbatim (index string -> p) | same keys by contract |
//! | `noul` | `{"true": noul, "false": 1 - noul}` | the contract ships a single `noul` float |
//!
//! The emitted rows carry `probabilities` and **no `label`**: the predicted slot is
//! the argmax of the vector under one frozen tie-break rule, and writing a second,
//! redundant `label` would create two sources of truth for the same number. Where
//! the server's own `choice` disagrees with that argmax it is reported as a
//! warning (see [`ImportOutcome::contract_disagreements`]) rather than silently
//! reconciled.

use crate::error::EvalError;
use crate::items::Item;
use crate::metrics::argmax_first_max;
use crate::predictions::Prediction;
use jev_core::Answer;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// The subset of the server response that `import` consumes. `usage` is ignored
/// on purpose; it is not a metric.
#[derive(Debug, Deserialize)]
struct AnswersFile {
    #[serde(default)]
    model: Option<String>,
    answers: BTreeMap<String, Answer>,
}

/// What an import produced, including the things a human should look at.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportOutcome {
    pub predictions: Vec<Prediction>,
    pub model: Option<String>,
    /// Items the answers file said nothing about. They become no-answer rows, which
    /// the report counts as `n_missing` — never silently dropped.
    pub missing_answers: Vec<String>,
    /// Items where the server's own argmax disagree with ours. Empty is the
    /// expected outcome.
    pub contract_disagreements: Vec<String>,
}

/// Convert one `Answer` into a prediction row for `item`.
pub fn prediction_from_answer(item: &Item, answer: &Answer) -> Result<Prediction, String> {
    match (&item.question, answer) {
        (jev_core::Question::Choice(_), Answer::Choice(choice)) => Ok(Prediction {
            id: item.id.clone(),
            label: None,
            probabilities: Some(choice.probabilities.clone()),
        }),
        (jev_core::Question::Score(_), Answer::Score(score)) => Ok(Prediction {
            id: item.id.clone(),
            label: None,
            probabilities: Some(score.probabilities.clone()),
        }),
        (jev_core::Question::Noul(_), Answer::Noul(noul)) => {
            let mut probabilities = BTreeMap::new();
            probabilities.insert("true".to_string(), noul.noul);
            probabilities.insert("false".to_string(), 1.0 - noul.noul);
            Ok(Prediction {
                id: item.id.clone(),
                label: None,
                probabilities: Some(probabilities),
            })
        }
        (question, answer) => Err(format!(
            "answer kind {answer:?} does not match question kind {}",
            match question {
                jev_core::Question::Choice(_) => "choice",
                jev_core::Question::Score(_) => "score",
                jev_core::Question::Noul(_) => "noul",
            }
        )),
    }
}

/// Read an answers file and convert every item the frozen set declares.
pub fn import_answers(answers_path: &Path, items: &[Item]) -> Result<ImportOutcome, EvalError> {
    let display = answers_path.display().to_string();
    let bytes = std::fs::read(answers_path).map_err(|source| EvalError::Io {
        path: display.clone(),
        source,
    })?;
    let parsed: AnswersFile = serde_json::from_slice(&bytes).map_err(|err| EvalError::Answers {
        path: display.clone(),
        message: err.to_string(),
    })?;

    let mut predictions = Vec::with_capacity(items.len());
    let mut missing_answers = Vec::new();
    let mut contract_disagreements = Vec::new();

    for item in items {
        let Some(answer) = parsed.answers.get(&item.id) else {
            missing_answers.push(item.id.clone());
            predictions.push(Prediction::missing(item.id.clone()));
            continue;
        };
        let prediction = prediction_from_answer(item, answer).map_err(|message| EvalError::Answers {
            path: display.clone(),
            message: format!("item {:?}: {message}", item.id),
        })?;

        // Cross-check the server's own decision against the frozen argmax rule.
        //
        // Only `choice` is checked. The contract defines `ScoreAnswer.score` as the
        // expectation `Σ i·pᵢ`, not as a level index, so comparing it with an
        // argmax would manufacture disagreements that are not defects (0.4/0.0/0.6
        // has argmax 2 and expectation 1.2). `noul` ships no discrete label at all.
        if let (Answer::Choice(choice), Some(probabilities)) =
            (answer, prediction.probabilities.as_ref())
        {
            let slots = item.slot_labels();
            let vector: Vec<f64> = slots
                .iter()
                .map(|slot| probabilities.get(slot).copied().unwrap_or(f64::NAN))
                .collect();
            if slots.len() == probabilities.len() && vector.iter().all(|value| value.is_finite()) {
                let ours = &slots[argmax_first_max(&vector)];
                if choice.choice != *ours {
                    contract_disagreements.push(format!(
                        "{}: server says {:?}, argmax of its own distribution is {ours:?}",
                        item.id, choice.choice
                    ));
                }
            }
        }

        predictions.push(prediction);
    }

    Ok(ImportOutcome {
        predictions,
        model: parsed.model,
        missing_answers,
        contract_disagreements,
    })
}

/// Serialize predictions as JSONL (one object per line, trailing newline).
pub fn predictions_to_jsonl(predictions: &[Prediction]) -> Result<String, EvalError> {
    let mut out = String::new();
    for prediction in predictions {
        let line = serde_json::to_string(prediction)
            .map_err(|err| EvalError::message(format!("cannot serialize prediction: {err}")))?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use jev_core::{ChoiceAnswer, NoulAnswer, ScoreAnswer};
    use serde_json::json;

    fn item(id: &str, question: serde_json::Value) -> Item {
        Item {
            id: id.into(),
            category: "evidence_judgment".into(),
            source: "unit".into(),
            state: json!("state"),
            question: serde_json::from_value(question).unwrap(),
            gold: json!("yes"),
            positive: None,
            provenance: None,
        }
    }

    #[test]
    fn a_choice_answer_becomes_its_probability_map() {
        let it = item(
            "a",
            json!({"type":"choice","instructions":"pick","criteria":{"no":null,"yes":null}}),
        );
        let mut probabilities = BTreeMap::new();
        probabilities.insert("no".to_string(), 0.25);
        probabilities.insert("yes".to_string(), 0.75);
        let answer = Answer::Choice(ChoiceAnswer {
            choice: "yes".into(),
            probabilities,
            confidence: 0.75,
        });
        let prediction = prediction_from_answer(&it, &answer).unwrap();
        assert_eq!(prediction.label, None);
        let emitted = prediction.probabilities.unwrap();
        assert_eq!(emitted["yes"], 0.75);
        assert_eq!(emitted["no"], 0.25);
    }

    #[test]
    fn a_score_answer_keeps_the_level_index_keys() {
        let it = item(
            "a",
            json!({"type":"score","instructions":"rate","criteria":["bad","ok","good"]}),
        );
        let mut probabilities = BTreeMap::new();
        probabilities.insert("0".to_string(), 0.1);
        probabilities.insert("1".to_string(), 0.2);
        probabilities.insert("2".to_string(), 0.7);
        let answer = Answer::Score(ScoreAnswer {
            score: 2.0,
            legend: BTreeMap::new(),
            probabilities,
            confidence: 0.7,
        });
        let prediction = prediction_from_answer(&it, &answer).unwrap();
        assert_eq!(prediction.probabilities.unwrap()["2"], 0.7);
    }

    #[test]
    fn a_noul_answer_splits_into_true_and_false() {
        let it = item("a", json!({"type":"noul","instructions":"hold?"}));
        let answer = Answer::Noul(NoulAnswer { noul: 0.8 });
        let prediction = prediction_from_answer(&it, &answer).unwrap();
        let probabilities = prediction.probabilities.unwrap();
        assert!((probabilities["true"] - 0.8).abs() < 1e-15);
        assert!((probabilities["false"] - 0.2).abs() < 1e-15);
    }

    #[test]
    fn a_mismatched_answer_kind_is_an_error_not_a_guess() {
        let it = item(
            "a",
            json!({"type":"choice","instructions":"pick","criteria":{"no":null,"yes":null}}),
        );
        let answer = Answer::Noul(NoulAnswer { noul: 0.5 });
        let err = prediction_from_answer(&it, &answer).unwrap_err();
        assert!(err.contains("does not match question kind choice"), "{err}");
    }

    #[test]
    fn jsonl_output_is_one_object_per_line_and_round_trips() {
        let predictions = vec![
            Prediction::missing("a"),
            Prediction {
                id: "b".into(),
                label: Some("yes".into()),
                probabilities: None,
            },
        ];
        let text = predictions_to_jsonl(&predictions).unwrap();
        assert_eq!(text.lines().count(), 2);
        let parsed: Vec<Prediction> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(parsed, predictions);
    }

    #[test]
    fn import_reads_a_real_answers_file_and_flags_missing_items() {
        let dir = std::env::temp_dir().join(format!("jev-eval-import-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("answers.json");
        std::fs::write(
            &path,
            r#"{"model":"mock","answers":{"a":{"type":"choice","choice":"yes",
               "probabilities":{"no":0.2,"yes":0.8},"confidence":0.8}},"usage":{"input_tokens":1,"output_tokens":1}}"#,
        )
        .unwrap();

        let items = vec![
            item(
                "a",
                json!({"type":"choice","instructions":"pick","criteria":{"no":null,"yes":null}}),
            ),
            item(
                "b",
                json!({"type":"choice","instructions":"pick","criteria":{"no":null,"yes":null}}),
            ),
        ];
        let outcome = import_answers(&path, &items).unwrap();
        assert_eq!(outcome.model.as_deref(), Some("mock"));
        assert_eq!(outcome.predictions.len(), 2);
        assert_eq!(outcome.missing_answers, vec!["b".to_string()]);
        assert!(outcome.contract_disagreements.is_empty());
        assert!(outcome.predictions[1].probabilities.is_none());
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn import_flags_a_server_decision_that_contradicts_its_own_distribution() {
        let dir = std::env::temp_dir().join(format!("jev-eval-import-disagree-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("answers.json");
        std::fs::write(
            &path,
            r#"{"answers":{"a":{"type":"choice","choice":"no",
               "probabilities":{"no":0.2,"yes":0.8},"confidence":0.8}}}"#,
        )
        .unwrap();
        let items = vec![item(
            "a",
            json!({"type":"choice","instructions":"pick","criteria":{"no":null,"yes":null}}),
        )];
        let outcome = import_answers(&path, &items).unwrap();
        assert_eq!(outcome.contract_disagreements.len(), 1);
        assert!(outcome.contract_disagreements[0].contains("argmax"));
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir_all(&dir).ok();
    }
}
