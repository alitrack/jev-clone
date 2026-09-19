//! Answer-slot verification.
//!
//! Three checks, all mandatory, all *fatal* (see `CoreError`):
//!
//! 1. **Single token.** Encoding a slot letter on its own must yield exactly one
//!    token id. (No decode round trip is asserted in M0: the backend's
//!    `decode_token` is explicitly best-effort, so a hard assertion here could
//!    fail on a healthy endpoint. Worth adding once a real detokenize path exists.)
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
    let letters: Vec<String> = (0..count)
        .map(|i| (b'A' + i as u8) as char)
        .map(|c| c.to_string())
        .collect();
    let prompt_ids = tokenizer.encode(prompt);

    let mut slot_ids = Vec::with_capacity(count);
    let mut token_texts = Vec::with_capacity(count);
    for letter in &letters {
        // Check 1: the letter is exactly one token on its own.
        let letter_ids = tokenizer.encode(letter);
        if letter_ids.len() != 1 {
            return Err(CoreError::SlotNotSingleToken { slot: letter.clone(), encoded: letter_ids });
        }
        // Check 2: appending the letter must not re-tokenize the prompt.
        let combined = tokenizer.encode(&format!("{prompt}{letter}"));
        let expected: Vec<u32> = prompt_ids.iter().copied().chain(std::iter::once(letter_ids[0])).collect();
        if combined != expected {
            return Err(CoreError::SlotBoundaryChanged { slot: letter.clone() });
        }
        slot_ids.push(letter_ids[0]);
        token_texts.push(letter.clone());
    }

    // Check 3: no two slots may share a token id.
    let mut collisions = Vec::new();
    for i in 0..slot_ids.len() {
        for j in (i + 1)..slot_ids.len() {
            if slot_ids[i] == slot_ids[j] {
                if !collisions.iter().any(|s| s == &letters[i]) {
                    collisions.push(letters[i].clone());
                }
                if !collisions.iter().any(|s| s == &letters[j]) {
                    collisions.push(letters[j].clone());
                }
            }
        }
    }
    if !collisions.is_empty() {
        return Err(CoreError::SlotCollision { slots: collisions });
    }

    Ok(SlotSet { letters, token_ids: slot_ids, token_texts })
}
