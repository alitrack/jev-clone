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

use crate::contract::{Entry, Question};
use crate::error::CoreError;

/// A rendered question: the prompt text plus the ordered slot -> option-label map.
///
/// `slots[i]` is the caller-facing value that a probability maps to:
/// for `choice` it is the criteria key, for `score` the level index as a string,
/// for `noul` it is `"true"` then `"false"`.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderedQuestion {
    pub prompt: String,
    /// slot letter ("A".."Z") -> value reported back to the caller.
    pub slots: Vec<(String, String)>,
    /// human-readable label of each option, in the same order (for logs/eval only).
    pub labels: Vec<String>,
}

/// Render one question against a shared, already-rendered state.
pub fn render_question(
    state: &Entry,
    question: &Question,
    question_id: &str,
) -> Result<RenderedQuestion, CoreError> {
    let _ = (state, question, question_id);
    todo!("worker A: implement per the module doc + specs/M0.md §A")
}

/// Render an `Entry` for inclusion in the prompt: strings as-is, JSON pretty-printed.
pub fn render_entry(entry: &Entry) -> String {
    let _ = entry;
    todo!("worker A")
}
