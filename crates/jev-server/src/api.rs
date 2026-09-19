//! jev-server — HTTP surface.
//!
//! M0 scope: a correct, single-question-at-a-time request path. Prefix sharing
//! across questions of one request is M1 and must not change this contract.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use jev_backend::{DecisionBackend, HttpTokenizer, Prefetch};
use jev_core::{
    confidence_from_probabilities, render_entry, render_question, score_expectation, softmax,
    verify_slots, Answer, ChoiceAnswer, CoreError, NoulAnswer, ScoreAnswer, SlotVerifier,
    SystemOneRequest, SystemOneResponse, Usage,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn DecisionBackend>,
    pub tokenizer: Arc<HttpTokenizer>,
    /// token budget for one request (public contract: 64k, state + longest question <= 32k)
    pub max_prompt_tokens: usize,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/systemone", post(systemone))
        .route("/healthz", get(healthz))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(serde_json::json!({"status": "ok", "service": "jev-clone"}))
}

/// Errors that must be visible to the caller. Contract violations are 422, readout
/// failures are 502 — never a silently wrong probability.
#[derive(Debug)]
pub enum ApiError {
    Contract(CoreError),
    Backend(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (code, message) = match self {
            ApiError::Contract(e) => (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()),
            ApiError::Backend(e) => (StatusCode::BAD_GATEWAY, e),
        };
        (code, Json(serde_json::json!({"error": {"message": message}}))).into_response()
    }
}

/// Round to 6 decimal places, keeping the wire values clean
/// (0.3 stays 0.3 instead of 0.30000000000000004).
fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

async fn systemone(
    State(state): State<AppState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, ApiError> {
    // Step 1: at least one question, otherwise 422.
    if req.questions.is_empty() {
        return Err(ApiError::Contract(CoreError::NoQuestions));
    }

    let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;

    // Step 2: per question — render -> prefetch -> verify slots -> readout ->
    // slot logprobs -> softmax. Questions are independent: each one gets its own
    // prompt, its own readout; nothing leaks between them.
    for (id, question) in &req.questions {
        let rendered =
            render_question(&req.state, question, id).map_err(ApiError::Contract)?;
        let n_options = rendered.slots.len();

        // The verifier (and the prompt-length budget) needs encodings of the
        // prompt, of every standalone slot letter, and of prompt+letter —
        // fetch them all up front so the synchronous verify_slots never hits a
        // missing cache entry.
        let mut texts: Vec<String> = vec![rendered.prompt.clone()];
        for (letter, _) in &rendered.slots {
            texts.push(letter.clone());
            texts.push(format!("{}{}", rendered.prompt, letter));
        }
        state
            .tokenizer
            .prefetch(&texts)
            .await
            .map_err(|e| ApiError::Backend(format!("tokenization failed: {e}")))?;

        // Budget is a hard limit: over it is an error, never a silent truncation.
        let prompt_tokens = state.tokenizer.encode(&rendered.prompt).len();
        if prompt_tokens > state.max_prompt_tokens {
            return Err(ApiError::Contract(CoreError::PromptTooLong {
                id: id.clone(),
                tokens: prompt_tokens,
                limit: state.max_prompt_tokens,
            }));
        }

        let _slot_set =
            verify_slots(state.tokenizer.as_ref(), &rendered.prompt, n_options)
                .map_err(ApiError::Contract)?;

        // Read the distribution at the answer position. top_k covers every
        // declared slot plus margin, floored at 10 (the probe-verified minimum).
        let top_k = n_options.saturating_add(5).max(10);
        let readout = state
            .backend
            .token_logprobs(&rendered.prompt, top_k)
            .await
            .map_err(|e| ApiError::Backend(e.to_string()))?;

        let mut slot_logprobs: Vec<f64> = Vec::with_capacity(n_options);
        for (letter, _) in &rendered.slots {
            match readout.slot_logprob(letter) {
                Some(lp) => slot_logprobs.push(lp),
                // A declared slot missing from the top-k is a readout failure:
                // we refuse to score it as zero.
                None => {
                    return Err(ApiError::Contract(CoreError::SlotMissingFromTopK {
                        id: id.clone(),
                        slot: letter.clone(),
                        top_k,
                    }))
                }
            }
        }

        let probabilities = softmax(&slot_logprobs);
        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(ApiError::Contract(CoreError::Unnormalized { sum }));
        }

        input_tokens = input_tokens.saturating_add(readout.prompt_tokens);
        // One question costs exactly one generated token: the backend is required
        // to set `completion_tokens = 1` and to generate the single token purely to
        // read the distribution at that position (its text is discarded). Counted
        // per question rather than summed from the backend report so the public
        // contract (`output_tokens == 题数 × 1`) cannot drift if a backend
        // under-reports. The readout's token fields are still used for the input
        // side, where the backend is the only source of truth.
        output_tokens = output_tokens.saturating_add(1);

        // Step 3: assemble the typed answer.
        let answer = match question {
            jev_core::Question::Choice(_c) => {
                // Plain argmax over the slot probabilities. A strict `>` keeps the
                // *first* maximum on ties (the numpy.argmax convention), so an
                // all-equal distribution resolves to the first declared option
                // instead of the last.
                let argmax = probabilities
                    .iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |best, (i, &p)| {
                        if p > best.1 {
                            (i, p)
                        } else {
                            best
                        }
                    })
                    .0;
                let mut probs = BTreeMap::new();
                for ((_, value), p) in rendered.slots.iter().zip(&probabilities) {
                    probs.insert(value.clone(), round6(*p));
                }
                Answer::Choice(ChoiceAnswer {
                    choice: rendered.slots[argmax].1.clone(),
                    probabilities: probs,
                    confidence: round6(confidence_from_probabilities(&probabilities, Default::default())),
                })
            }
            jev_core::Question::Score(s) => {
                let mut probs = BTreeMap::new();
                let mut legend = BTreeMap::new();
                for (i, ((_, value), p)) in rendered.slots.iter().zip(&probabilities).enumerate() {
                    probs.insert(value.clone(), round6(*p));
                    // `legend` maps the level index (the caller-facing value of a
                    // score slot, e.g. "0") to the *human-readable level
                    // description*, rendered exactly as it appears in the prompt.
                    // `RenderedQuestion::labels` carries only the index string for
                    // score, so the description has to come from the caller's own
                    // criteria; the fallback keeps the map complete even if a
                    // renderer ever produced more slots than criteria.
                    let description = s
                        .criteria
                        .get(i)
                        .map(render_entry)
                        .unwrap_or_else(|| value.clone());
                    legend.insert(value.clone(), description);
                }
                Answer::Score(ScoreAnswer {
                    score: round6(score_expectation(&probabilities)),
                    legend,
                    probabilities: probs,
                    confidence: round6(confidence_from_probabilities(&probabilities, Default::default())),
                })
            }
            jev_core::Question::Noul(_) => {
                // render (A1) fixes the noul slot order: A -> "true" (yes),
                // B -> "false". Take the mass on the "true" slot explicitly.
                let noul = probabilities
                    .iter()
                    .enumerate()
                    .find(|(i, _)| rendered.slots[*i].1 == "true")
                    .map(|(i, _)| probabilities[i])
                    .unwrap_or(probabilities.first().copied().unwrap_or(0.0));
                Answer::Noul(NoulAnswer { noul: round6(noul) })
            }
        };
        answers.insert(id.clone(), answer);
    }

    // Steps 4+5: usage totals and the backend's model id.
    Ok(Json(SystemOneResponse {
        model: state.backend.model_name(),
        answers,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    }))
}
