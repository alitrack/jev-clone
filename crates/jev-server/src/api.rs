//! jev-server — HTTP surface.
//!
//! M0 scope: a correct, single-question-at-a-time request path. Prefix sharing
//! across questions of one request is M1 and must not change this contract.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use jev_backend::DecisionBackend;
use jev_core::{CoreError, SystemOneRequest, SystemOneResponse};
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<dyn DecisionBackend>,
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

async fn systemone(
    State(_state): State<AppState>,
    Json(_req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, ApiError> {
    todo!(
        "worker B: for each question -> render_question -> verify_slots -> \
         backend.token_logprobs -> softmax over slot logprobs -> build Answer; \
         accumulate usage. Return ApiError::Contract for CoreError."
    )
}
