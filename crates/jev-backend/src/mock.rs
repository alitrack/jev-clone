//! Deterministic in-process backend for tests (no network, no GPU).
//!
//! Feeds a queue of scripted readouts, one per question, in order.
//!
//! Batched reads use the trait's default implementation (a sequential loop over
//! [`token_logprobs`](DecisionBackend::token_logprobs)): asking for N prompts
//! consumes the next N scripted readouts in call order, which is exactly the
//! per-prompt semantics the batched readout promises. An explicit override would
//! only duplicate that logic; `tests/openai_backend.rs` pins the behaviour of the
//! real batched parser (ordering by `index`, all-or-nothing failures).

use crate::Readout;
use crate::{BackendError, DecisionBackend};
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct MockBackend {
    model: String,
    readouts: Vec<Readout>,
    cursor: AtomicUsize,
}

impl MockBackend {
    pub fn new(model: impl Into<String>, readouts: Vec<Readout>) -> Self {
        Self { model: model.into(), readouts, cursor: AtomicUsize::new(0) }
    }

    /// Convenience: build a readout from `(token, logprob)` pairs.
    pub fn readout(pairs: &[(&str, f64)]) -> Readout {
        Readout {
            top: pairs.iter().map(|(t, l)| ((*t).to_string(), *l)).collect::<BTreeMap<_, _>>(),
            prompt_tokens: 0,
            completion_tokens: 1,
        }
    }
}

#[async_trait]
impl DecisionBackend for MockBackend {
    async fn token_logprobs(&self, _prompt: &str, _top_k: usize) -> Result<Readout, BackendError> {
        let i = self.cursor.fetch_add(1, Ordering::SeqCst);
        self.readouts
            .get(i)
            .cloned()
            .ok_or_else(|| BackendError::Http(format!("mock exhausted at call {i}")))
    }

    fn model_name(&self) -> String {
        self.model.clone()
    }
}
