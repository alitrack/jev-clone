//! Errors. Every variant names the file (and usually the line and item id) that
//! caused it, because the whole point of this crate is that a failure is
//! traceable back to the evidence.

/// Anything that can go wrong while loading evidence or computing metrics.
///
/// Note the deliberate asymmetry: a *malformed item set* or a *prediction whose
/// probabilities do not cover the item's slots* is an error, never a silent
/// zero. Scoring an option the model never saw a probability for would invent a
/// number (`specs/M0.md` §0.5, `specs/M1.md` §5②).
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} is not valid UTF-8")]
    NotUtf8 { path: String },

    #[error("{path}:{line}: invalid JSON: {message}")]
    Json {
        path: String,
        line: usize,
        message: String,
    },

    #[error("{path}:{line}: item {id:?}: {message}")]
    Item {
        path: String,
        line: usize,
        id: String,
        message: String,
    },

    #[error("{path}:{line}: prediction {id:?}: {message}")]
    Prediction {
        path: String,
        line: usize,
        id: String,
        message: String,
    },

    #[error("{path}: duplicate item id {id:?} (first seen on line {first_line})")]
    DuplicateItemId {
        path: String,
        id: String,
        first_line: usize,
    },

    #[error("{path}: no items (a frozen item set must be non-empty)")]
    EmptyItemSet { path: String },

    #[error("{path}: malformed JSON: {message}")]
    FileJson { path: String, message: String },

    #[error("{path}: missing required categories {missing:?} (all of {required:?} must appear at least once)")]
    MissingCategories {
        path: String,
        missing: Vec<String>,
        required: Vec<String>,
    },

    #[error("item-set hash mismatch for {path}: manifest records {expected}, file hashes to {actual} — refusing to compute (the item set is frozen; re-freeze it deliberately if this is intended)")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },

    #[error("manifest {path}: no entry for item set {items:?}")]
    ManifestEntryMissing { path: String, items: String },

    #[error("item {id:?}: cannot render it for prompt hashing: {source}")]
    Render {
        id: String,
        #[source]
        source: jev_core::CoreError,
    },

    #[error("{path}: malformed report: {message}")]
    Report { path: String, message: String },

    #[error("{path}: malformed answers file: {message}")]
    Answers { path: String, message: String },

    #[error("{0}")]
    Message(String),
}

impl EvalError {
    /// Convenience for ad-hoc, already-formatted failures.
    pub fn message(msg: impl Into<String>) -> Self {
        Self::Message(msg.into())
    }
}
