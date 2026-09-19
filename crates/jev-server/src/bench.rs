//! `jev-bench` — the prefix-reuse measurement, as a pure-Rust binary body.
//!
//! Two paths over the same N rendered questions and the same `state`:
//!
//! * **fresh** — one independent `/completions` request per question (M0's
//!   shape): every question pays its own prefill.
//! * **shared-prefix** — one batched request (`prompt` as an array, specs/M1.md
//!   §3): every prompt shares the `state` prefix, so a prefix-caching server
//!   (SGLang radix cache) prefills it once.
//!
//! Both paths are timed with [`std::time::Instant`] around the *whole* call, and
//! every raw sample is written to disk. Nothing here is post-processed to look
//! better: the raw arrays are the evidence, the percentiles are derived from them
//! and can be recomputed by anyone holding the file.
//!
//! ## Why the win has to be measured in wall clock
//!
//! The endpoint reports `usage.prompt_tokens` as `N x prefix` — the radix cache
//! does **not** discount the count (probed, specs/M1.md §1). Token accounting is
//! therefore identical on both paths; only wall time separates them. Reporting a
//! token-based speedup would be reporting a number that does not exist.
//!
//! ## What the number does and does not isolate
//!
//! The baseline is per-question requests sent **sequentially to the same
//! server**. A radix cache persists across requests, so from the second request
//! on, part of the baseline's prefix may already be cached. The measured speedup
//! is therefore "batched request vs sequential requests against this server",
//! which is the deployment question we actually care about — not a
//! cache-disabled upper bound. Batching and prefix reuse are measured together
//! and the report says so.
//!
//! The module lives in the library (not in `src/bin/`) so its parts that can be
//! tested without a GPU — argument parsing, question generation, percentiles, UTC
//! date formatting, report assembly — have real tests. `src/bin/jev-bench.rs` is
//! the thin CLI shell.

use jev_backend::{BackendError, DecisionBackend, OpenAiCompatBackend, OpenAiCompatConfig};
use jev_core::{render_question, ChoiceQuestion, NoulCriteria, NoulQuestion, Question, ScoreQuestion};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Default endpoint (the verified reference server, specs/M1.md §1).
pub const DEFAULT_BASE_URL: &str = "http://10.10.10.115:8014/v1";
/// Default model id as the reference server knows it.
pub const DEFAULT_MODEL: &str = "qwen3.8-27b";
/// Default question count (the acceptance gate is stated for 21).
pub const DEFAULT_QUESTIONS: usize = 21;
/// Default repeat count (the acceptance gate is stated for >= 3 runs).
pub const DEFAULT_REPEAT: usize = 3;
/// Default output directory for the result JSON, relative to the repository root.
pub const DEFAULT_OUT_DIR: &str = "bench/results";

pub const USAGE: &str = "\
jev-bench — measure per-question readouts vs one batched readout over a shared state

usage:
  jev-bench --state-file <path> [--base-url <url>] [--model <id>] \\
            [--questions <n>] [--repeat <r>] [--api-key <key>] \\
            [--out-dir <dir>] [--timeout-secs <s>]

flags:
  --state-file <path>   state text shared by every question (required)
  --base-url <url>      OpenAI-compatible base       (default: reference endpoint)
  --model <id>          model id the server knows    (default: qwen3.8-27b)
  --questions <n>       number of questions          (default: 21)
  --repeat <r>          repeats per path             (default: 3)
  --api-key <key>       bearer token                 (default: EMPTY)
  --out-dir <dir>       result directory             (default: bench/results)
  --timeout-secs <s>    per-request timeout          (default: 120)
  --top-k <n>           override top_k (default: the server rule, max(n_slots+5, 100))
  -h, --help            print this help
";

/// Command-line settings. Every value is already validated by [`parse_args`].
#[derive(Debug, Clone, PartialEq)]
pub struct BenchArgs {
    pub base_url: String,
    pub model: String,
    pub state_file: String,
    pub questions: usize,
    pub repeat: usize,
    pub api_key: String,
    pub out_dir: String,
    pub timeout_secs: u64,
    /// `Some(n)` forces `top_k = n` instead of the server rule
    /// (`max(n_slots + 5, 100)`). Used to measure how the declared-slot coverage
    /// of a readout responds to a deeper top-k; a forced value is recorded in the
    /// report's notes so the two can never be confused.
    pub top_k: Option<usize>,
}

/// Parse `argv` (without the program name). Unknown flags and malformed values
/// are errors, never silently ignored: a mistyped `--questions` must not fall
/// back to 21 and quietly produce a report for the wrong workload.
pub fn parse_args(argv: &[String]) -> Result<BenchArgs, String> {
    let mut args = BenchArgs {
        base_url: DEFAULT_BASE_URL.to_string(),
        model: DEFAULT_MODEL.to_string(),
        state_file: String::new(),
        questions: DEFAULT_QUESTIONS,
        repeat: DEFAULT_REPEAT,
        api_key: "EMPTY".to_string(),
        out_dir: DEFAULT_OUT_DIR.to_string(),
        timeout_secs: 120,
        top_k: None,
    };

    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        // Every supported flag takes a value; read it once, uniformly.
        let value = || -> Result<String, String> {
            argv.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("flag {flag} needs a value"))
        };
        match flag {
            "--base-url" => args.base_url = value()?,
            "--model" => args.model = value()?,
            "--state-file" => args.state_file = value()?,
            "--api-key" => args.api_key = value()?,
            "--out-dir" => args.out_dir = value()?,
            "--questions" => args.questions = parse_usize(flag, &value()?)?,
            "--top-k" => args.top_k = Some(parse_usize(flag, &value()?)?),
            "--repeat" => args.repeat = parse_usize(flag, &value()?)?,
            "--timeout-secs" => args.timeout_secs = parse_usize(flag, &value()?)? as u64,
            other => return Err(format!("unknown flag {other:?} (see --help)")),
        }
        i += 2;
    }

    if args.state_file.is_empty() {
        return Err("--state-file is required".to_string());
    }
    if args.questions == 0 {
        return Err("--questions must be >= 1".to_string());
    }
    if args.repeat == 0 {
        return Err("--repeat must be >= 1".to_string());
    }
    Ok(args)
}

fn parse_usize(flag: &str, raw: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|e| format!("flag {flag} needs a non-negative integer, got {raw:?}: {e}"))
}

// ---------------------------------------------------------------------------
// Workload generation
// ---------------------------------------------------------------------------

/// N questions of one `state`, mixing the three primitives deterministically
/// (`i % 3`), each with a distinct instruction text so the readouts cannot all
/// collapse to the same distribution.
pub fn generate_questions(n: usize) -> Vec<(String, Question)> {
    (0..n).map(|i| (format!("q{i:02}"), question_at(i))).collect()
}

fn question_at(i: usize) -> Question {
    match i % 3 {
        0 => Question::Choice(ChoiceQuestion {
            instructions: Value::String(format!(
                "Item {i}: does the state say the duplicate payment for invoice 88213 was refunded?"
            )),
            criteria: [
                ("no", Value::String("the state says it was not refunded".into())),
                ("unclear", Value::Null),
                ("yes", Value::String("the state says it was refunded".into())),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        }),
        1 => Question::Score(ScoreQuestion {
            instructions: Value::String(format!(
                "Item {i}: how well does the state support a refund for invoice 88213?"
            )),
            criteria: vec![
                Value::String("no support in the state".into()),
                Value::String("weak support in the state".into()),
                Value::String("clear support in the state".into()),
            ],
        }),
        _ => Question::Noul(NoulQuestion {
            instructions: Value::String(format!(
                "Item {i}: does the state record a finance approval for the duplicate payment?"
            )),
            criteria: Some(NoulCriteria {
                r#true: Some(Value::String("the state records an approval".into())),
                r#false: Some(Value::String("the state records no approval".into())),
            }),
        }),
    }
}

/// `max(n_slots) + 5`, floored at 100 — the same rule the server uses
/// (specs/M1.md §2). Kept here as a function so the bench and the server cannot
/// drift apart silently. The floor is 100 because a shallower readout dropped a
/// declared slot in ~6% of live readouts; see the comment in `api.rs`.
pub fn top_k_for(max_slots: usize) -> usize {
    max_slots.saturating_add(5).max(100)
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Percentile with linear interpolation between closest ranks (the numpy
/// default). `p` is in `[0, 100]`. Empty input yields 0.0 (and the caller
/// reports the sample count alongside, so nothing is hidden by the sentinel).
pub fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("latencies are never NaN"));
    let last = sorted.len() - 1;
    let rank = (p.clamp(0.0, 100.0) / 100.0) * last as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = rank - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// One path's timings, all derived from `raw_ms` (nothing is recomputed from a
/// different source, so the file can be re-checked by hand).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Timing {
    /// raw wall-clock samples in milliseconds, in the order they were taken.
    pub raw_ms: Vec<f64>,
    pub samples: usize,
    pub total_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    /// Amortised wall time per decision (total / decision count).
    pub per_decision_ms: f64,
    pub decisions_per_second: f64,
}

fn timing(samples_ms: Vec<f64>, decisions: usize) -> Timing {
    let total_ms: f64 = samples_ms.iter().sum();
    let per_decision_ms = if decisions == 0 { 0.0 } else { total_ms / decisions as f64 };
    let decisions_per_second = if total_ms > 0.0 {
        decisions as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    Timing {
        samples: samples_ms.len(),
        p50_ms: percentile(&samples_ms, 50.0),
        p95_ms: percentile(&samples_ms, 95.0),
        raw_ms: samples_ms,
        total_ms,
        per_decision_ms,
        decisions_per_second,
    }
}

fn ms_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------------------
// Dates / hostname (no chrono: `time`/`chrono` are not in this machine's offline
// cargo cache, and the two values we need are a dozen lines of arithmetic)
// ---------------------------------------------------------------------------

/// Civil date from days since the Unix epoch (Howard Hinnant's `civil_from_days`).
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `YYYY-MM-DD` for a `SystemTime`, in UTC.
pub fn utc_date(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// `YYYY-MM-DDTHH:MM:SSZ` for a `SystemTime`, in UTC.
pub fn utc_timestamp(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    let (y, mo, d) = civil_from_days(secs.div_euclid(86_400));
    let tod = secs.rem_euclid(86_400);
    let (h, mi, s) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Host name for the result filename: `/proc/sys/kernel/hostname`, falling back
/// to `$HOSTNAME`, reduced to `[A-Za-z0-9._-]` so it is always a safe filename.
pub fn hostname() -> String {
    let raw = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok().filter(|s| !s.is_empty()))
        .unwrap_or_else(|| "unknown-host".to_string());
    sanitize_host(&raw)
}

fn sanitize_host(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' })
        .collect();
    if cleaned.is_empty() {
        "unknown-host".to_string()
    } else {
        cleaned
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct TokenAccounting {
    pub fresh_prompt_tokens: u64,
    pub shared_prompt_tokens: u64,
    pub fresh_completion_tokens: u64,
    pub shared_completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SlotCoverage {
    /// readouts whose top-k contained every declared slot of their question.
    pub complete: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BenchReport {
    pub timestamp_utc: String,
    pub host: String,
    pub endpoint: String,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sglang_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info_error: Option<String>,
    pub state_file: String,
    pub state_chars: usize,
    pub questions: usize,
    pub repeat: usize,
    pub top_k: usize,
    pub question_types: BTreeMap<String, usize>,
    /// Baseline: one independent request per question, sequentially.
    pub fresh: Timing,
    /// Optimised: one batched request per repeat, `prompt` as an array.
    pub shared_prefix: Timing,
    /// `fresh.total_ms / shared_prefix.total_ms` (see `speedup_basis`).
    pub speedup: f64,
    pub speedup_basis: String,
    pub token_accounting: TokenAccounting,
    pub slot_coverage: SlotCoverage,
    pub notes: Vec<String>,
    /// Where this very report was written (filled in after serialisation).
    pub result_file: String,
}

fn question_types(questions: &[(String, Question)]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, q) in questions {
        let key = match q {
            Question::Choice(_) => "choice",
            Question::Score(_) => "score",
            Question::Noul(_) => "noul",
        };
        *counts.entry(key.to_string()).or_insert(0) += 1;
    }
    counts
}

/// Best-effort `/get_server_info` probe: the spec asks for the SGLang version
/// *if available*, and a server that does not expose it must not fail the run.
async fn probe_server_info(base_url: &str, api_key: &str) -> (Option<String>, Option<String>) {
    let base = base_url.trim_end_matches('/').to_string();
    let root = base.strip_suffix("/v1").unwrap_or(&base).to_string();
    let client = match reqwest::Client::builder().timeout(Duration::from_secs(5)).build() {
        Ok(c) => c,
        Err(e) => return (None, Some(format!("client build failed: {e}"))),
    };
    for url in [format!("{root}/get_server_info"), format!("{base}/get_server_info")] {
        match client.get(&url).bearer_auth(api_key).send().await {
            Ok(resp) if resp.status().is_success() => {
                let text = resp.text().await.unwrap_or_default();
                if let Ok(v) = serde_json::from_str::<Value>(&text) {
                    let version = v
                        .get("version")
                        .or_else(|| v.get("sglang_version"))
                        .and_then(|x| x.as_str())
                        .map(str::to_string);
                    if version.is_some() {
                        return (version, None);
                    }
                    return (None, Some(format!("{url}: no `version` field in server info")));
                }
                return (None, Some(format!("{url}: server info is not JSON")));
            }
            Ok(resp) => {
                return (None, Some(format!("{url}: server info returned {}", resp.status())))
            }
            // Nothing answered here; try the next spelling.
            Err(_) => continue,
        }
    }
    (None, Some("no /get_server_info endpoint answered".to_string()))
}

/// Run both paths and assemble the report. Writes the result JSON and returns
/// the report (with `result_file` filled in).
pub async fn run(args: &BenchArgs) -> anyhow::Result<BenchReport> {
    let state_text = std::fs::read_to_string(&args.state_file)
        .map_err(|e| anyhow::anyhow!("cannot read --state-file {}: {e}", args.state_file))?;
    let state = Value::String(state_text.clone());

    let questions = generate_questions(args.questions);

    // Render once and reuse the exact same prompts on both paths: any difference
    // between the two measurements must come from the request shape, never from
    // a different prompt.
    let mut rendered = Vec::with_capacity(questions.len());
    let mut max_slots = 0usize;
    for (id, q) in &questions {
        let r = render_question(&state, q, id)?;
        max_slots = max_slots.max(r.slots.len());
        rendered.push(r);
    }
    let top_k = args.top_k.unwrap_or_else(|| top_k_for(max_slots));
    let prompts: Vec<String> = rendered.iter().map(|r| r.prompt.clone()).collect();
    let declared_slots: Vec<usize> = rendered.iter().map(|r| r.slots.len()).collect();
    let decisions_per_repeat = questions.len();

    let backend = OpenAiCompatBackend::new(OpenAiCompatConfig {
        base_url: args.base_url.clone(),
        model: args.model.clone(),
        api_key: args.api_key.clone(),
        timeout_secs: args.timeout_secs,
    })?;

    let (sglang_version, server_info_error) = probe_server_info(&args.base_url, &args.api_key).await;

    let mut coverage_complete = 0usize;
    let mut coverage_total = 0usize;
    let mut count_coverage = |readout: &jev_backend::Readout, n_slots: usize| {
        coverage_total += 1;
        let all_present = (0..n_slots).all(|i| {
            let letter = ((b'A' + i as u8) as char).to_string();
            readout.slot_logprob(&letter).is_some()
        });
        if all_present {
            coverage_complete += 1;
        }
    };

    // ---- interleaved A/B: per repeat, one fresh block then one batched block ---
    //
    // Deliberately interleaved rather than "all fresh, then all batched". The
    // reference endpoint is shared with other jobs, and a contention spike that
    // lands inside a single arm's window gets reported as a property of that path:
    // four consecutive runs of the same workload produced ratios of 1.50x, 2.23x,
    // 0.88x and 1.06x while the arms were timed as separate blocks. Interleaving
    // spreads spikes across both arms, and the raw arrays plus the per-repeat
    // ratios stay in the output so nobody has to take the summary on faith.
    eprintln!(
        "jev-bench: interleaved A/B — {} repeats, each = {} fresh requests then 1 batched request of {} prompts",
        args.repeat,
        questions.len(),
        prompts.len()
    );
    let mut fresh_ms = Vec::with_capacity(questions.len() * args.repeat);
    let mut shared_ms = Vec::with_capacity(args.repeat);
    let mut fresh_prompt_tokens = 0u64;
    let mut fresh_completion_tokens = 0u64;
    let mut shared_prompt_tokens = 0u64;
    let mut shared_completion_tokens = 0u64;
    let mut per_repeat_ratio: Vec<f64> = Vec::with_capacity(args.repeat);
    for repeat in 0..args.repeat {
        // Arm 1 — one independent request per question, sent sequentially.
        let fresh_start = Instant::now();
        for (i, prompt) in prompts.iter().enumerate() {
            let started = Instant::now();
            let readout = backend
                .token_logprobs(prompt, top_k)
                .await
                .map_err(|e: BackendError| {
                    anyhow::anyhow!("fresh request {i} (repeat {}) failed: {e}", repeat + 1)
                })?;
            fresh_ms.push(ms_since(started));
            fresh_prompt_tokens += readout.prompt_tokens;
            fresh_completion_tokens += readout.completion_tokens;
            count_coverage(&readout, declared_slots[i]);
        }
        let fresh_block_s = fresh_start.elapsed().as_secs_f64();

        // Arm 2 — the same workload as one request, `prompt` as an array.
        let shared_start = Instant::now();
        let readouts = backend
            .token_logprobs_batch(&prompts, top_k)
            .await
            .map_err(|e: BackendError| {
                anyhow::anyhow!("batched request (repeat {}) failed: {e}", repeat + 1)
            })?;
        shared_ms.push(ms_since(shared_start));
        if readouts.len() != prompts.len() {
            anyhow::bail!(
                "batched request returned {} readouts for {} prompts",
                readouts.len(),
                prompts.len()
            );
        }
        for (i, readout) in readouts.iter().enumerate() {
            shared_prompt_tokens += readout.prompt_tokens;
            shared_completion_tokens += readout.completion_tokens;
            count_coverage(readout, declared_slots[i]);
        }
        let shared_block_s = shared_start.elapsed().as_secs_f64();
        let ratio = if shared_block_s > 0.0 {
            fresh_block_s / shared_block_s
        } else {
            0.0
        };
        per_repeat_ratio.push(ratio);
        eprintln!(
            "  repeat {}/{}: fresh {:.2}s  shared {:.2}s  ratio {:.2}x",
            repeat + 1,
            args.repeat,
            fresh_block_s,
            shared_block_s,
            ratio
        );
    }

    let decisions = decisions_per_repeat * args.repeat;
    let fresh = timing(fresh_ms, decisions);
    let shared_prefix = timing(shared_ms, decisions);
    let speedup = if shared_prefix.total_ms > 0.0 {
        fresh.total_ms / shared_prefix.total_ms
    } else {
        0.0
    };

    let now = SystemTime::now();
    let host = hostname();
    let mut notes = vec![
        "Wall clock is the only valid measure of the prefix-reuse win: the endpoint reports \
         usage.prompt_tokens as N x prefix (no cache discount), so token counts are ~equal on \
         both paths (specs/M1.md §1)."
            .to_string(),
        "Baseline = one independent /completions request per question, sent sequentially to the \
         same server; optimised = one request with `prompt` as an array. The server's radix cache \
         also persists across separate requests, so the ratio measures batched-vs-sequential \
         against this server, not a cache-disabled upper bound."
            .to_string(),
        "No warmup request was issued on either path (nothing was removed to inflate the ratio); \
         the first sample of each path is visible in raw_ms."
            .to_string(),
        format!(
            "speedup basis: total wall clock of the {} fresh requests divided by the total wall \
             clock of the {} batched requests, for the same {} decisions.",
            questions.len() * args.repeat,
            args.repeat,
            decisions
        ),
        format!(
            "raw latencies: fresh = {} per-question samples, shared-prefix = {} per-batch samples \
             (divide by {} for the per-decision cost).",
            fresh.samples,
            shared_prefix.samples,
            questions.len()
        ),
        "The M3 Ultra / local-backend table is not part of this run (specs/M1.md §4).".to_string(),
    ];
    if let Some(forced) = args.top_k {
        // Never let a hand-picked top_k masquerade as the server rule.
        notes.push(format!(
            "top_k {forced} was forced with --top-k, not the server rule max(n_slots+5, 100) = {}",
            top_k_for(max_slots)
        ));
    }
    notes.push(format!(
        "Arms were interleaved per repeat (fresh block, then batched block); per-repeat \
         fresh/shared ratios: {}",
        per_repeat_ratio
            .iter()
            .map(|r| format!("{r:.2}x"))
            .collect::<Vec<_>>()
            .join(", ")
    ));

    let date = utc_date(now);
    let result_file = format!("{}/{date}-{host}.json", args.out_dir.trim_end_matches('/'));
    let mut report = BenchReport {
        timestamp_utc: utc_timestamp(now),
        host,
        endpoint: args.base_url.clone(),
        model: args.model.clone(),
        sglang_version,
        server_info_error,
        state_file: args.state_file.clone(),
        state_chars: state_text.chars().count(),
        questions: questions.len(),
        repeat: args.repeat,
        top_k,
        question_types: question_types(&questions),
        fresh,
        shared_prefix,
        speedup,
        speedup_basis: "total wall clock: sum(fresh per-question requests) / sum(batched requests)"
            .to_string(),
        token_accounting: TokenAccounting {
            fresh_prompt_tokens,
            shared_prompt_tokens,
            fresh_completion_tokens,
            shared_completion_tokens,
        },
        slot_coverage: SlotCoverage {
            complete: coverage_complete,
            total: coverage_total,
        },
        notes,
        result_file: result_file.clone(),
    };

    if let Some(dir) = std::path::Path::new(&result_file).parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| anyhow::anyhow!("cannot create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(&result_file, format!("{json}\n"))
        .map_err(|e| anyhow::anyhow!("cannot write {result_file}: {e}"))?;
    // Echo the path in the struct too, so a caller that re-serialises the report
    // keeps the pointer to the original file.
    report.result_file = result_file;
    Ok(report)
}

/// Human-readable table, printed to stdout by the binary.
pub fn render_table(r: &BenchReport) -> String {
    let mut out = String::new();
    out.push_str("jev-bench — per-question readouts vs one batched readout over a shared state\n");
    out.push_str(&format!(
        "endpoint     : {}  (model {}, {})\n",
        r.endpoint,
        r.model,
        match (&r.sglang_version, &r.server_info_error) {
            (Some(v), _) => format!("SGLang {v}"),
            (None, Some(e)) => format!("server info unavailable: {e}"),
            (None, None) => "server version unknown".to_string(),
        }
    ));
    out.push_str(&format!(
        "state        : {} ({} chars; token count is the server's to report)\n",
        r.state_file, r.state_chars
    ));
    let types = r
        .question_types
        .iter()
        .map(|(k, v)| format!("{k} {v}"))
        .collect::<Vec<_>>()
        .join(" / ");
    out.push_str(&format!(
        "workload     : {} questions ({types}), repeat {}, top_k {}\n",
        r.questions, r.repeat, r.top_k
    ));
    out.push_str(&format!(
        "host / time  : {} / {}\n",
        r.host, r.timestamp_utc
    ));
    out.push_str(&format!(
        "slot coverage: {}/{} readouts contained every declared slot\n\n",
        r.slot_coverage.complete, r.slot_coverage.total
    ));

    out.push_str(&format!(
        "{:<16}{:>9}{:>11}{:>10}{:>10}{:>14}{:>14}\n",
        "path", "samples", "total_s", "p50_ms", "p95_ms", "ms/decision", "decisions/s"
    ));
    for (label, t) in [("fresh (per-q)", &r.fresh), ("shared (batch)", &r.shared_prefix)] {
        out.push_str(&format!(
            "{:<16}{:>9}{:>11.3}{:>10.2}{:>10.2}{:>14.2}{:>14.2}\n",
            label,
            t.samples,
            t.total_ms / 1000.0,
            t.p50_ms,
            t.p95_ms,
            t.per_decision_ms,
            t.decisions_per_second
        ));
    }
    // A percentile over a handful of samples is not a percentile: at n = 5 the
    // interpolated p95 sits between the two largest values, i.e. it is effectively
    // the max. The `samples` column shows n, but the two arms differ in n by
    // construction (per-question vs per-round), so say it in words too.
    let min_samples = r.fresh.samples.min(r.shared_prefix.samples);
    if min_samples < 20 {
        out.push_str(&format!(
            "\nnote: p95 is over as few as {min_samples} sample(s) — for an arm with a small \
             sample count, p95 is effectively the maximum, not a stable tail estimate. Compare \
             `total_s` and `ms/decision` before quoting it.\n"
        ));
    }
    out.push_str(&format!(
        "\nspeedup (wall clock, same {} decisions): {:.2}x\n",
        r.questions * r.repeat,
        r.speedup
    ));
    out.push_str(&format!(
        "token accounting: fresh prompt_tokens {}, shared prompt_tokens {} (endpoint counts N x \
         prefix; the cache does not discount it)\n",
        r.token_accounting.fresh_prompt_tokens, r.token_accounting.shared_prompt_tokens
    ));
    out.push_str(&format!("results      : {}\n", r.result_file));
    out.push_str(
        "\nnote: results JSON holds both raw_ms arrays — every number above is recomputable from \
         it.\n",
    );
    out
}

/// Small helper kept for the report's own sanity: a JSON round trip of the
/// report must not change the timing values.
pub fn report_json(r: &BenchReport) -> anyhow::Result<String> {
    Ok(serde_json::to_string_pretty(r)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn argv(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn defaults_fill_every_option_but_the_state_file() {
        let args = parse_args(&argv(&["--state-file", "bench/states/long-state.txt"])).unwrap();
        assert_eq!(args.base_url, DEFAULT_BASE_URL);
        assert_eq!(args.model, DEFAULT_MODEL);
        assert_eq!(args.questions, 21);
        assert_eq!(args.repeat, 3);
        assert_eq!(args.api_key, "EMPTY");
        assert_eq!(args.out_dir, DEFAULT_OUT_DIR);
        assert_eq!(args.state_file, "bench/states/long-state.txt");
    }

    #[test]
    fn every_flag_overrides_its_default() {
        let args = parse_args(&argv(&[
            "--base-url",
            "http://127.0.0.1:9999/v1",
            "--model",
            "m",
            "--state-file",
            "s.txt",
            "--questions",
            "4",
            "--repeat",
            "1",
            "--api-key",
            "k",
            "--out-dir",
            "out",
            "--timeout-secs",
            "7",
        ]))
        .unwrap();
        assert_eq!(args.base_url, "http://127.0.0.1:9999/v1");
        assert_eq!(args.model, "m");
        assert_eq!(args.state_file, "s.txt");
        assert_eq!(args.questions, 4);
        assert_eq!(args.repeat, 1);
        assert_eq!(args.api_key, "k");
        assert_eq!(args.out_dir, "out");
        assert_eq!(args.timeout_secs, 7);
    }

    #[test]
    fn a_missing_state_file_is_an_error() {
        let err = parse_args(&argv(&["--questions", "2"])).unwrap_err();
        assert!(err.contains("--state-file"), "{err}");
    }

    #[test]
    fn unknown_flags_and_missing_values_are_errors() {
        assert!(parse_args(&argv(&["--state-file", "s", "--wat"])).unwrap_err().contains("unknown"));
        assert!(parse_args(&argv(&["--state-file"])).unwrap_err().contains("needs a value"));
        assert!(parse_args(&argv(&["--state-file", "s", "--questions", "x"]))
            .unwrap_err()
            .contains("non-negative integer"));
    }

    #[test]
    fn zero_questions_or_repeats_are_rejected() {
        assert!(parse_args(&argv(&["--state-file", "s", "--questions", "0"]))
            .unwrap_err()
            .contains("--questions"));
        assert!(parse_args(&argv(&["--state-file", "s", "--repeat", "0"]))
            .unwrap_err()
            .contains("--repeat"));
    }

    #[test]
    fn percentile_interpolates_and_handles_edges() {
        assert_eq!(percentile(&[], 50.0), 0.0);
        assert_eq!(percentile(&[7.0], 95.0), 7.0);
        // 1..=4 -> p50 = 2.5, p0 = 1, p100 = 4.
        let s = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&s, 50.0) - 2.5).abs() < 1e-12);
        assert!((percentile(&s, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile(&s, 100.0) - 4.0).abs() < 1e-12);
        // 1..=100 -> p95 = 95.05 (linear interpolation between ranks 94 and 95).
        let s100: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert!((percentile(&s100, 95.0) - 95.05).abs() < 1e-9);
    }

    #[test]
    fn timing_derives_every_field_from_the_raw_samples() {
        let t = timing(vec![100.0, 200.0, 300.0, 400.0], 8);
        assert_eq!(t.samples, 4);
        assert!((t.total_ms - 1000.0).abs() < 1e-9);
        assert!((t.per_decision_ms - 125.0).abs() < 1e-9);
        assert!((t.decisions_per_second - 8.0).abs() < 1e-9, "8 decisions in 1s");
        assert_eq!(t.raw_ms, vec![100.0, 200.0, 300.0, 400.0]);
    }

    #[test]
    fn utc_dates_are_correct_at_known_epochs() {
        assert_eq!(utc_date(UNIX_EPOCH), "1970-01-01");
        assert_eq!(utc_date(UNIX_EPOCH + Duration::from_secs(59 * 86_400)), "1970-03-01");
        // 10_957 days = 30 years including 7 leap days -> 2000-01-01.
        assert_eq!(utc_date(UNIX_EPOCH + Duration::from_secs(10_957 * 86_400)), "2000-01-01");
        assert_eq!(
            utc_timestamp(UNIX_EPOCH + Duration::from_secs(86_400 + 3661)),
            "1970-01-02T01:01:01Z"
        );
    }

    #[test]
    fn civil_from_days_round_trips_against_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        assert_eq!(civil_from_days(19_000), (2022, 1, 8));
    }

    #[test]
    fn hostnames_are_safe_for_filenames() {
        assert_eq!(sanitize_host("DESKTOP-ABC123"), "DESKTOP-ABC123");
        assert_eq!(sanitize_host("host with spaces"), "host-with-spaces");
        assert_eq!(sanitize_host(""), "unknown-host");
        assert!(!hostname().is_empty());
    }

    #[test]
    fn generated_questions_are_deterministic_and_mix_the_three_primitives() {
        let qs = generate_questions(21);
        assert_eq!(qs.len(), 21);
        assert_eq!(qs, generate_questions(21), "generation must be reproducible");
        let counts = question_types(&qs);
        assert_eq!(counts.get("choice"), Some(&7));
        assert_eq!(counts.get("score"), Some(&7));
        assert_eq!(counts.get("noul"), Some(&7));
        assert_eq!(qs[0].0, "q00");
        assert_eq!(qs[20].0, "q20");
    }

    #[test]
    fn generated_questions_render_without_error_and_only_one_letter_each_slot() {
        let state = Value::String("shared state".into());
        for (id, q) in generate_questions(21) {
            let r = render_question(&state, &q, &id).expect("generated questions render");
            assert!(r.slots.len() >= 2 && r.slots.len() <= 26, "{id}: {} slots", r.slots.len());
            assert!(r.prompt.starts_with("State:\nshared state\n"), "{id}: shared prefix shape");
            assert!(r.prompt.ends_with("Answer:\n"), "{id}: prompt ends at the answer slot");
            assert_eq!(r.labels.len(), r.slots.len());
        }
    }

    #[test]
    fn top_k_matches_the_server_rule() {
        // The renderer caps a question at 26 letter slots, so `n_slots + 5 <= 31`
        // and the floor always wins: every reachable input yields exactly 100. The
        // `+ 5` margin is therefore inert — it is kept because it is the correct
        // rule *if* the slot ceiling ever moves, but no reachable call depends on
        // it, so there is nothing here to test above the floor.
        assert_eq!(top_k_for(2), 100);
        assert_eq!(top_k_for(15), 100);
        assert_eq!(top_k_for(16), 100);
        assert_eq!(top_k_for(26), 100, "the widest question the renderer accepts");
        assert_eq!(top_k_for(26 + 5), 100, "even the largest reachable n_slots + 5");
    }

    /// The margin above the floor is dead code for every reachable input — pin that
    /// claim so a future slot-ceiling change surfaces here instead of silently
    /// making `top_k` depend on the question's width.
    #[test]
    fn the_plus_five_margin_never_binds_at_the_current_slot_ceiling() {
        const MAX_LETTER_SLOTS: usize = 26; // render.rs: slots run A..Z
        for n_slots in 2..=MAX_LETTER_SLOTS {
            assert_eq!(
                top_k_for(n_slots),
                100,
                "n_slots = {n_slots} must not depend on the margin"
            );
        }
    }

    #[test]
    fn a_report_round_trips_through_json_without_losing_samples() {
        let report = BenchReport {
            timestamp_utc: utc_timestamp(UNIX_EPOCH),
            host: "test-host".into(),
            endpoint: "http://127.0.0.1:1/v1".into(),
            model: "m".into(),
            sglang_version: Some("0.4.9".into()),
            server_info_error: None,
            state_file: "s.txt".into(),
            state_chars: 12,
            questions: 2,
            repeat: 1,
            top_k: 20,
            question_types: BTreeMap::from([("choice".to_string(), 2)]),
            fresh: timing(vec![1.0, 2.0], 2),
            shared_prefix: timing(vec![1.5], 2),
            speedup: 2.0,
            speedup_basis: "test".into(),
            token_accounting: TokenAccounting {
                fresh_prompt_tokens: 10,
                shared_prompt_tokens: 10,
                fresh_completion_tokens: 2,
                shared_completion_tokens: 2,
            },
            slot_coverage: SlotCoverage { complete: 3, total: 3 },
            notes: vec!["n".into()],
            result_file: "bench/results/x.json".into(),
        };
        let json = report_json(&report).unwrap();
        let back: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(back["fresh"]["raw_ms"], json!([1.0, 2.0]));
        assert_eq!(back["shared_prefix"]["raw_ms"], json!([1.5]));
        assert_eq!(back["speedup"], json!(2.0));
        let table = render_table(&report);
        assert!(table.contains("2.00x"), "{table}");
        assert!(table.contains("bench/results/x.json"), "{table}");
    }
}
