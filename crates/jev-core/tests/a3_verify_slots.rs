//! A4-3: `verify_slots` — one success case (real endpoint ids) plus one case
//! for each of the three failure branches.

use jev_core::{verify_slots, CoreError, SlotVerifier};

/// Deterministic fake tokenizer keyed on the real qwen3.8-27b `/tokenize`
/// observations: `"Answer:\n"` -> [15666, 25, 198], `"A"` -> [32], `"B"` -> [33],
/// `"C"` -> [34]. All other inputs encode to a single bogus id (77) unless a
/// `mode` below overrides them.
#[derive(Default)]
struct FakeTokenizer {
    mode: Mode,
}

#[derive(Clone, Copy, Default, PartialEq)]
enum Mode {
    /// Faithful to the observed endpoint.
    #[default]
    Real,
    /// One letter (C) encodes as two tokens.
    MultiToken,
    /// Appending a letter re-tokenizes the prompt tail.
    UnstableBoundary,
    /// Two letters (B and C) share one token id.
    Colliding,
}

const PROMPT: &str = "Answer:\n";
const PROMPT_IDS: &[u32] = &[15666, 25, 198];

impl SlotVerifier for FakeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self.mode {
            Mode::MultiToken => {
                if text == "C" {
                    return vec![100, 101];
                }
            }
            Mode::UnstableBoundary => {
                if let Some(letter) = text.strip_prefix(PROMPT) {
                    if letter.len() == 1 {
                        // The prompt tail re-tokenizes when the letter is appended.
                        return vec![15666, 25, 999];
                    }
                }
            }
            // Colliding falls through to the shared paths below, where B and C
            // both encode to id 33.
            Mode::Colliding | Mode::Real => {}
        }

        if text == PROMPT {
            return PROMPT_IDS.to_vec();
        }
        // Standalone letters: the Colliding mode maps both B and C to id 33.
        if text == "A" {
            return vec![32];
        }
        if text == "B" || (self.mode == Mode::Colliding && text == "C") {
            return vec![33];
        }
        if text == "C" {
            return vec![34];
        }
        // The combined path (prompt + one letter) must stay consistent with the
        // standalone encoding, or check 2 (boundary) would fire before check 3
        // (collision) in the Colliding mode.
        if let Some(letter) = text.strip_prefix(PROMPT) {
            if letter.len() == 1 {
                let id = match letter {
                    "A" => 32,
                    "B" => 33,
                    "C" => {
                        if self.mode == Mode::Colliding {
                            33
                        } else {
                            34
                        }
                    }
                    _ => 77,
                };
                return [PROMPT_IDS, std::slice::from_ref(&id)].concat();
            }
        }
        vec![77]
    }

    fn decode_token(&self, id: u32) -> String {
        match id {
            32 => "A".to_string(),
            33 => "B".to_string(),
            34 => "C".to_string(),
            _ => "<unk>".to_string(),
        }
    }
}

#[test]
fn success_with_real_endpoint_ids() {
    let tz = FakeTokenizer::default();
    let slots = verify_slots(&tz, PROMPT, 2).expect("slots verify");
    assert_eq!(slots.letters, vec!["A".to_string(), "B".to_string()]);
    assert_eq!(slots.token_ids, vec![32, 33]);
    assert_eq!(slots.token_texts, vec!["A".to_string(), "B".to_string()]);

    // A third slot still verifies; order follows the option order.
    let slots3 = verify_slots(&tz, PROMPT, 3).expect("slots verify");
    assert_eq!(slots3.token_ids, vec![32, 33, 34]);
    assert_eq!(slots3.letters, vec!["A".to_string(), "B".to_string(), "C".to_string()]);
}

#[test]
fn zero_slots_is_trivially_valid() {
    let tz = FakeTokenizer::default();
    let slots = verify_slots(&tz, PROMPT, 0).expect("empty set");
    assert!(slots.letters.is_empty() && slots.token_ids.is_empty() && slots.token_texts.is_empty());
}

#[test]
fn multi_token_slot_is_rejected() {
    let tz = FakeTokenizer { mode: Mode::MultiToken };
    let err = verify_slots(&tz, PROMPT, 3).unwrap_err();
    assert_eq!(err, CoreError::SlotNotSingleToken { slot: "C".into(), encoded: vec![100, 101] });
}

#[test]
fn boundary_change_is_rejected() {
    let tz = FakeTokenizer { mode: Mode::UnstableBoundary };
    let err = verify_slots(&tz, PROMPT, 2).unwrap_err();
    assert_eq!(err, CoreError::SlotBoundaryChanged { slot: "A".into() });
}

#[test]
fn colliding_slot_ids_are_rejected() {
    let tz = FakeTokenizer { mode: Mode::Colliding };
    let err = verify_slots(&tz, PROMPT, 3).unwrap_err();
    assert_eq!(err, CoreError::SlotCollision { slots: vec!["B".into(), "C".into()] });
}
