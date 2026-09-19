//! jev-server — HTTP surface.
//!
//! Request path: render + slot-verify every question, then **one** batched
//! readout for the whole request (all prompts share the `state` prefix, so a
//! prefix-caching server prefills it once — specs/M1.md §3), then assemble the
//! typed answers. A single-question request walks the same code path.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use jev_backend::{DecisionBackend, HttpTokenizer, Prefetch};
use jev_core::{
    confidence_from_probabilities, render_question, score_expectation, softmax, verify_slots,
    Answer, ChoiceAnswer, CoreError, NoulAnswer, ScoreAnswer, SlotVerifier,
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

/// Repair the sum of an already-rounded distribution, in place.
///
/// Rounding each slot independently is what a reader expects on the wire, but it
/// also breaks the contract: `Σp` can drift to `1 ± n·5e-7`, and any caller that
/// checks the sum then sees a violation the server itself created. Measured on
/// the live endpoint: 2 of 45 answers came back exactly 1e-6 off, and `jev-eval`
/// (correctly) refused to score the file over it. So round first, then hand the
/// residual to **one** slot.
///
/// That slot is the first maximum in `order` (the item's declared slot order) —
/// i.e. exactly the slot the answer's argmax already names. Picking "the largest"
/// naively would hand the residual to the *last* of several tied maxima, which
/// would give the serialised distribution an argmax that disagrees with the
/// `choice`/`score` field right next to it. The residual is at most `n·5e-7`, so
/// the slot moves by less than the rounding it just absorbed, and may carry one
/// extra decimal.
fn repair_probability_sum(order: &[String], probs: &mut BTreeMap<String, f64>) {
    let residual = 1.0 - probs.values().sum::<f64>();
    if residual == 0.0 {
        return;
    }
    let target = order
        .iter()
        .filter_map(|k| probs.get(k).map(|v| (k, *v)))
        .fold(None, |best: Option<(&String, f64)>, (k, v)| match best {
            Some((_, best_v)) if best_v >= v => best,
            _ => Some((k, v)),
        })
        .map(|(k, _)| k.clone());
    if let Some(slot) = target.and_then(|k| probs.get_mut(&k)) {
        *slot += residual;
    }
}

async fn systemone(
    State(state): State<AppState>,
    Json(req): Json<SystemOneRequest>,
) -> Result<Json<SystemOneResponse>, ApiError> {
    // Step 1: at least one question, otherwise 422.
    if req.questions.is_empty() {
        return Err(ApiError::Contract(CoreError::NoQuestions));
    }

    // One rendered, slot-verified question, ready to be read.
    struct Prepared<'a> {
        id: String,
        question: &'a jev_core::Question,
        rendered: jev_core::RenderedQuestion,
        n_options: usize,
    }

    // Step 2 (no backend calls yet): per question — render -> prefetch the
    // encodings verification needs -> token-budget check -> verify slots. Every
    // contract violation surfaces here, *before* any readout is paid for, and in
    // the questions' own (BTreeMap) order, so which error wins is deterministic.
    let mut prepared: Vec<Prepared> = Vec::with_capacity(req.questions.len());
    let mut max_slots = 0usize;
    for (id, question) in &req.questions {
        let rendered = render_question(&req.state, question, id).map_err(ApiError::Contract)?;
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

        verify_slots(state.tokenizer.as_ref(), &rendered.prompt, n_options)
            .map_err(ApiError::Contract)?;

        max_slots = max_slots.max(n_options);
        prepared.push(Prepared {
            id: id.clone(),
            question,
            rendered,
            n_options,
        });
    }

    // Step 3: **one** batched readout for every question of the request.
    //
    // Every prompt is `state` prefix + question suffix + `Answer:`, so the whole
    // request shares one prefix and a server with prefix caching (SGLang radix
    // cache) prefills it once instead of once per question. A single-question
    // request walks this exact same code path — there is no "fast path for one".
    //
    // top_k covers the largest declared option set plus margin, floored at 100.
    // The floor is a measured number, not a guess: on the live endpoint
    // (21 questions x 3 repeats, `jev-bench`) a top_k of 20 omitted a declared
    // slot in 118/126 readouts — ~6% of real requests would have failed with
    // `SlotMissingFromTopK`. At top_k = 100 the same workload covered every slot
    // (126/126) while the per-decision cost moved ~1% (median 226 ms -> 228 ms on
    // a 284-token prompt). Deeper is essentially free; a missed slot is not.
    let top_k = max_slots.saturating_add(5).max(100);
    let prompts: Vec<String> = prepared.iter().map(|p| p.rendered.prompt.clone()).collect();
    let readouts = state
        .backend
        .token_logprobs_batch(&prompts, top_k)
        .await
        .map_err(|e| ApiError::Backend(e.to_string()))?;
    // The trait promises one readout per prompt; a backend that breaks that
    // promise must not be papered over by zipping whatever lines up.
    if readouts.len() != prompts.len() {
        return Err(ApiError::Backend(format!(
            "backend returned {} readouts for {} prompts",
            readouts.len(),
            prompts.len()
        )));
    }

    // Step 4: assemble the typed answers (probability math, legend and usage all
    // use the same core functions as M0). Questions are independent: each answer
    // is derived from its own readout only, and nothing of one question enters
    // another's prompt.
    let mut answers: BTreeMap<String, Answer> = BTreeMap::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;

    for (p, readout) in prepared.iter().zip(&readouts) {
        let mut slot_logprobs: Vec<f64> = Vec::with_capacity(p.n_options);
        for (letter, _) in &p.rendered.slots {
            match readout.slot_logprob(letter) {
                Some(lp) => slot_logprobs.push(lp),
                // A declared slot missing from the top-k is a readout failure:
                // we refuse to score it as zero (that would invent probability
                // mass) or to renormalize over the rest (that would invent a
                // distribution). See specs/M1.md §5②.
                None => {
                    return Err(ApiError::Contract(CoreError::SlotMissingFromTopK {
                        id: p.id.clone(),
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

        let answer = match p.question {
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
                let order: Vec<String> = p
                    .rendered
                    .slots
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect();
                for ((_, value), prob) in p.rendered.slots.iter().zip(&probabilities) {
                    probs.insert(value.clone(), round6(*prob));
                }
                repair_probability_sum(&order, &mut probs);
                Answer::Choice(ChoiceAnswer {
                    choice: p.rendered.slots[argmax].1.clone(),
                    probabilities: probs,
                    confidence: round6(confidence_from_probabilities(&probabilities, Default::default())),
                })
            }
            jev_core::Question::Score(_) => {
                let mut probs = BTreeMap::new();
                let mut legend = BTreeMap::new();
                let order: Vec<String> = p
                    .rendered
                    .slots
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect();
                for (i, ((_, value), prob)) in p.rendered.slots.iter().zip(&probabilities).enumerate() {
                    probs.insert(value.clone(), round6(*prob));
                    // `legend` maps the level index (the caller-facing value of a
                    // score slot, e.g. "0") to the *human-readable level
                    // description*, rendered exactly as it appears in the prompt.
                    // M1 moved that text into `RenderedQuestion::labels` (the
                    // renderer owns the rendering, so nobody has to re-render the
                    // caller's raw criteria here to get the same string).
                    let description = p
                        .rendered
                        .labels
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| value.clone());
                    legend.insert(value.clone(), description);
                }
                repair_probability_sum(&order, &mut probs);
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
                    .find(|(i, _)| p.rendered.slots[*i].1 == "true")
                    .map(|(i, _)| probabilities[i])
                    .unwrap_or(probabilities.first().copied().unwrap_or(0.0));
                Answer::Noul(NoulAnswer { noul: round6(noul) })
            }
        };
        answers.insert(p.id.clone(), answer);
    }

    // Steps 5+6: usage totals and the backend's model id.
    Ok(Json(SystemOneResponse {
        model: state.backend.model_name(),
        answers,
        usage: Usage {
            input_tokens,
            output_tokens,
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probs(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn order(keys: &[&str]) -> Vec<String> {
        keys.iter().map(|s| s.to_string()).collect()
    }

    fn sum(p: &BTreeMap<String, f64>) -> f64 {
        p.values().sum()
    }

    #[test]
    fn rounding_three_equal_slots_needs_the_residual() {
        // 1/3 → round6 → 0.333333 each → Σ = 0.999999, exactly the drift that
        // made `jev-eval` refuse a real answers file.
        let mut p = probs(&[
            ("a", round6(1.0 / 3.0)),
            ("b", round6(1.0 / 3.0)),
            ("c", round6(1.0 / 3.0)),
        ]);
        assert!((sum(&p) - 0.999999).abs() < 1e-12, "precondition: Σ drifts low");
        repair_probability_sum(&order(&["a", "b", "c"]), &mut p);
        assert!((sum(&p) - 1.0).abs() < 1e-15, "Σ must be 1 after repair, got {}", sum(&p));
        assert_eq!(p["a"], 0.333334, "the residual lands on the first slot");
        assert_eq!((p["b"], p["c"]), (0.333333, 0.333333));
    }

    #[test]
    fn residual_goes_to_the_slot_the_answer_names_not_the_last_tie() {
        // All three tie, so `choice` names the FIRST declared slot. If the repair
        // picked "the largest" naively it would take the last tie (max_by's
        // convention), and the serialised distribution would then have an argmax
        // disagreeing with the `choice` field sitting next to it.
        let mut p = probs(&[("x", 0.5), ("y", 0.5), ("z", 0.0)]);
        repair_probability_sum(&order(&["x", "y", "z"]), &mut p);
        assert_eq!(p["x"], 0.5, "first tie keeps the residual");
        assert!(p["y"] <= 0.5);
    }

    #[test]
    fn repair_handles_drift_in_both_directions_and_leaves_exact_vectors_alone() {
        // Drift high (0.45142 + 0.511526 + 0.037055 = 1.000001, from a real answer).
        // The largest slot is "1", so it gives the residual back.
        let mut high = probs(&[("0", 0.45142), ("1", 0.511526), ("2", 0.037055)]);
        repair_probability_sum(&order(&["0", "1", "2"]), &mut high);
        assert!((sum(&high) - 1.0).abs() < 1e-15, "Σ = {}", sum(&high));
        assert!(high["1"] < 0.511526, "the largest slot absorbs a negative residual");

        // Already exact: nothing moves.
        let mut exact = probs(&[("yes", 0.25), ("no", 0.75)]);
        repair_probability_sum(&order(&["yes", "no"]), &mut exact);
        assert_eq!(exact, probs(&[("yes", 0.25), ("no", 0.75)]));

        // Degenerate maps must not panic.
        let mut empty = BTreeMap::new();
        repair_probability_sum(&order(&[]), &mut empty);
        assert!(empty.is_empty());
    }

    #[test]
    fn repaired_vector_never_goes_negative() {
        // Worst case the endpoint can produce: slots that all round down.
        let mut p = probs(&[("A", 0.4999995), ("B", 0.4999995), ("C", 0.000001)]);
        repair_probability_sum(&order(&["A", "B", "C"]), &mut p);
        assert!(p.values().all(|v| *v >= 0.0), "{p:?}");
        assert!((sum(&p) - 1.0).abs() < 1e-15, "Σ = {}", sum(&p));
    }
}
