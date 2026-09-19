//! jev-core — contract types, prompt rendering, answer-slot verification, and
//! probability/confidence math for a Jev-compatible System One decision server.
//!
//! Design rules enforced by this crate:
//! 1. Answers are **constrained to the options supplied by the caller**. The only
//!    guarantee we make is "the answer cannot fall outside your option set" — never
//!    "the answer is correct" (see `docs/design.md` §0).
//! 2. Question IDs are never sent to the model (they only key the response).
//! 3. Questions in one request are independent; one answer never becomes another
//!    question's context.
//! 4. Slot verification failures are **errors**, never silent mis-scoring.

pub mod contract;
pub mod error;
pub mod prob;
pub mod render;
pub mod slots;

pub use contract::{
    Answer, ChoiceAnswer, ChoiceQuestion, Entry, NoulAnswer, NoulCriteria, NoulQuestion, Question,
    ScoreAnswer, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage, MAX_CHOICE_OPTIONS,
};
pub use error::CoreError;
pub use prob::{confidence_from_probabilities, score_expectation, softmax, ConfidenceMode};
pub use render::{render_entry, render_question, RenderedQuestion};
pub use slots::{verify_slots, SlotSet, SlotVerifier};
