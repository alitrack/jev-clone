//! Typed errors. Every one of these must surface as a client error (HTTP 422 for
//! contract violations, 502 for backend/readout failures) — never as a silently
//! wrong probability.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum CoreError {
    #[error("request must contain at least one question")]
    NoQuestions,

    #[error("question `{id}`: choice criteria must have at least 2 options, got {got}")]
    TooFewChoiceOptions { id: String, got: usize },

    #[error("question `{id}`: choice criteria exceeds {max} options (got {got})")]
    TooManyChoiceOptions { id: String, got: usize, max: usize },

    #[error("question `{id}`: score criteria must have at least 2 levels, got {got}")]
    TooFewScoreLevels { id: String, got: usize },

    #[error("question `{id}`: noul criteria may only contain `true` and/or `false`")]
    BadNoulCriteriaKeys { id: String },

    /// More options than one-letter answer slots can name.
    ///
    /// Deliberately distinct from both neighbours: `TooManyChoiceOptions` is the
    /// public contract's own 255 limit, and `PromptTooLong` means a real token
    /// budget overflow. Here the prompt is perfectly fine — we simply cannot ask
    /// the model for one token per option once the alphabet runs out.
    #[error("question `{id}`: {got} options exceed the {max} single-letter answer slots M1 can name")]
    TooManyLetterSlots { id: String, got: usize, max: usize },

    /// The prompt plus one answer slot exceeded the caller's token budget.
    #[error("question `{id}`: prompt is {tokens} tokens, over the {limit} token budget (no truncation)")]
    PromptTooLong { id: String, tokens: usize, limit: usize },

    /// A declared answer slot is not a single, round-tripping token.
    #[error("answer slot {slot:?} is not one exact round-trip token (encoded as {encoded:?})")]
    SlotNotSingleToken { slot: String, encoded: Vec<u32> },

    /// Appending the slot token changed how the prompt itself tokenizes.
    #[error("answer boundary is unstable: tokenizing `prompt + {slot:?}` differs from `tokenize(prompt) + slot`")]
    SlotBoundaryChanged { slot: String },

    #[error("answer slot tokens collide: {slots:?}")]
    SlotCollision { slots: Vec<String> },

    /// The backend did not return a logprob for a declared slot inside the top-k it
    /// was asked for. We refuse to guess: this is a readout failure, not a low score.
    #[error("slot {slot:?} missing from the backend's top-{top_k} logprobs for question `{id}`")]
    SlotMissingFromTopK { id: String, slot: String, top_k: usize },

    #[error("backend returned no logprobs for question `{id}` (endpoint may not support them)")]
    NoLogprobs { id: String },

    #[error("probabilities do not sum to 1 (got {sum})")]
    Unnormalized { sum: f64 },
}
