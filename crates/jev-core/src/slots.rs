//! Answer-slot verification.
//!
//! Three checks, all mandatory, all *fatal* (see `CoreError`):
//!
//! 1. **Single token + round trip.** Encoding a slot letter with the tokenizer must
//!    yield exactly one token id, and decoding it must give the letter back.
//! 2. **Boundary stability.** `tokenize(prompt + letter)` must equal
//!    `tokenize(prompt) + [slot_token]`. If appending the answer letter re-tokenizes
//!    the tail of the prompt, the readout position is not where we think it is.
//! 3. **No collisions.** All slot tokens in one question must be distinct.
//!
//! This mirrors the reference implementation's `_slot_ids` / `encode_prompt`
//! assertions and is the single most important correctness device in the project:
//! a server that skips it will happily return confident nonsense.

use crate::error::CoreError;

/// A verified set of single-token answer slots for one question.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotSet {
    /// slot letter, e.g. "A"
    pub letters: Vec<String>,
    /// token id of each slot letter
    pub token_ids: Vec<u32>,
    /// the literal text each token decodes to (needed by token-string backends)
    pub token_texts: Vec<String>,
}

/// Tokenizer surface the verifier needs. Implemented for the real tokenizer in
/// `jev-backend`; a fake implementation is used in tests.
pub trait SlotVerifier {
    /// Encode without adding special tokens, returning token ids.
    fn encode(&self, text: &str) -> Vec<u32>;
    /// Decode a single token id back to text.
    fn decode_token(&self, id: u32) -> String;
}

/// Build and verify the slot set for `count` options against `prompt`.
pub fn verify_slots(
    tokenizer: &dyn SlotVerifier,
    prompt: &str,
    count: usize,
) -> Result<SlotSet, CoreError> {
    let _ = (tokenizer, prompt, count);
    todo!("worker A: implement per the module doc + specs/M0.md §B")
}
