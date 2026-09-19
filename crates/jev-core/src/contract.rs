//! Wire contract — field-for-field compatible with the public TypeSafe System One
//! HTTP API (docs.typesafe.ai/api). Field names are load-bearing: do not rename.
//!
//! Request:  { state, model, questions: { <id>: Question } }
//! Response: { model, answers: { <id>: Answer }, usage: { input_tokens, output_tokens } }
//!
//! Notes taken from the public contract:
//! * `state` is `string | object | array`.
//! * `instructions` is required on all three question types and may itself be
//!   a string, object or array.
//! * `choice.criteria` is a map label -> description (description may be null),
//!   at least 2 entries, at most [`MAX_CHOICE_OPTIONS`].
//! * `score.criteria` is an ordered array of level descriptions, at least 2.
//! * `noul.criteria` is optional and may only carry the keys `true` / `false`.
//! * Question IDs are never sent to the model.
//! * `noul` answers carry no `confidence`.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Maximum number of `choice` options, per the public contract.
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// A free-form text/JSON blob (`string | object | array | null`).
pub type Entry = Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    Choice(ChoiceQuestion),
    Score(ScoreQuestion),
    Noul(NoulQuestion),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceQuestion {
    pub instructions: Entry,
    /// option label -> rubric description (may be null).
    pub criteria: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreQuestion {
    pub instructions: Entry,
    /// ordered level descriptions, at least two.
    pub criteria: Vec<Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoulQuestion {
    pub instructions: Entry,
    /// optional `{ "true": ..., "false": ... }` clarification of what yes/no mean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct NoulCriteria {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#true: Option<Entry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#false: Option<Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneRequest {
    pub state: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
    Noul(NoulAnswer),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: BTreeMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScoreAnswer {
    pub score: f64,
    pub legend: BTreeMap<String, String>,
    pub probabilities: BTreeMap<String, f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoulAnswer {
    pub noul: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    pub usage: Usage,
}
