//! Async client for the [TypeSafe AI API](https://docs.typesafe.ai/api).
//!
//! TypeSafe's System One models (such as `jev-latest`) evaluate a `state` —
//! plain text or structured JSON — against a map of typed questions and return
//! one structured answer per question:
//!
//! * [`Question::noul`] — a yes/no question, answered with the probability of
//!   "yes" from 0 to 1.
//! * [`Question::choice`] — picks one option from a set you define and returns
//!   the full probability distribution.
//! * [`Question::score`] — rates the state along an ordered rubric of levels.
//!
//! The client mirrors the official SDKs: it reads configuration from
//! environment variables ([`ENV_API_KEY`], [`ENV_BASE_URL`],
//! [`ENV_DEFAULT_MODEL`]), retries retryable failures with exponential backoff
//! (honoring `Retry-After` / `retry-after-ms` up to a cap), and classifies
//! HTTP errors the same way the JavaScript and Python SDKs do.
//!
//! # Example
//!
//! ```no_run
//! use jev_ncr_demo::typesafe::{Question, SystemOneRequest, TypeSafeClient};
//!
//! # async fn example() -> Result<(), jev_ncr_demo::typesafe::Error> {
//! let client = TypeSafeClient::from_env()?;
//!
//! let result = client
//!     .system_one(SystemOneRequest::new(
//!         "Help! My payouts have been failing for 3 days.",
//!         [
//!             ("is_urgent", Question::noul("Does this convey urgency?")?),
//!             (
//!                 "department",
//!                 Question::choice(
//!                     "Which team should handle this?",
//!                     [
//!                         ("billing", "Payments, invoicing, refunds"),
//!                         ("technical", "Bugs, outages, integrations"),
//!                         ("sales", "Pricing, upgrades, new accounts"),
//!                     ],
//!                 )?,
//!             ),
//!         ],
//!     )?)
//!     .await?;
//!
//! if let Some(urgency) = result.noul("is_urgent") {
//!     println!("urgency: {urgency:.2}");
//! }
//! if let Some(answer) = result.choice("department") {
//!     println!("department: {} (confidence {:.2})", answer.choice, answer.confidence);
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fmt;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use reqwest::{Client, Method, StatusCode, Url};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Version of this client, reported in the `User-Agent` header.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default API root, used when [`ENV_BASE_URL`] is not set.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// Default model, used when [`ENV_DEFAULT_MODEL`] is not set and a request
/// does not override the model.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// Default per-attempt timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// Environment variable holding the API key (read by [`TypeSafeClient::from_env`]).
pub const ENV_API_KEY: &str = "TYPESAFE_API_KEY";

/// Environment variable overriding the API root.
pub const ENV_BASE_URL: &str = "TYPESAFE_BASE_URL";

/// Environment variable overriding the default model.
pub const ENV_DEFAULT_MODEL: &str = "TYPESAFE_DEFAULT_MODEL";

const SYSTEMONE_PATH: &str = "/v1/systemone";
const MODELS_PATH: &str = "/v1/models";
const USER_AGENT_VALUE: &str = concat!("typesafe-rust/", env!("CARGO_PKG_VERSION"));

/// A question, instruction, option, or level description: a string, JSON
/// object, or JSON array. Structured descriptions let you give an option or
/// level fields such as `what`, `not_for`, and `examples`; the field names are
/// yours. See [advanced structure](https://docs.typesafe.ai/primitives/advanced).
///
/// Pass a string, `serde_json::json!(...)`, or a `serde_json::Value`.
pub type Description = Value;

/// The `state` to evaluate: a string, JSON object, JSON array, or `Value::Null`.
/// Use `serde_json::json!` (or `serde_json::to_value`) for structured state.
pub type EntryType = Value;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by this client.
#[derive(Debug)]
pub enum Error {
    /// The client is misconfigured (missing API key, invalid base URL, ...).
    Config(String),
    /// The request is invalid before it is sent (empty questions, bad
    /// criteria, ...).
    InvalidRequest(String),
    /// The API returned a non-2xx response after retries.
    Api(Box<ApiError>),
    /// A payload could not be serialized or a response could not be parsed.
    Json(serde_json::Error),
    /// The request could not be sent (DNS, connect, interrupted body, ...).
    Connection(reqwest::Error),
    /// The request exceeded the per-attempt timeout after retries.
    Timeout {
        timeout: Duration,
        source: reqwest::Error,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(message) => write!(f, "invalid client configuration: {message}"),
            Error::InvalidRequest(message) => write!(f, "invalid request: {message}"),
            Error::Api(error) => write!(f, "{error}"),
            Error::Json(error) => {
                write!(f, "failed to process a TypeSafe API payload: {error}")
            }
            Error::Connection(error) => {
                write!(f, "connection to the TypeSafe API failed: {error}")
            }
            Error::Timeout { timeout, .. } => {
                write!(f, "request to the TypeSafe API timed out after {timeout:?}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Api(error) => Some(error),
            Error::Json(error) => Some(error),
            Error::Connection(error) => Some(error),
            Error::Timeout { source, .. } => Some(source),
            Error::Config(_) | Error::InvalidRequest(_) => None,
        }
    }
}

impl From<ApiError> for Error {
    fn from(error: ApiError) -> Self {
        Error::Api(Box::new(error))
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Error::Json(error)
    }
}

/// An unsuccessful HTTP response from the TypeSafe API.
#[derive(Debug)]
pub struct ApiError {
    /// HTTP status code of the response.
    pub status: StatusCode,
    /// Best-effort human-readable message extracted from the response body.
    pub message: String,
    /// The parsed response body, when the body was valid JSON.
    pub body: Option<Value>,
    /// The response headers.
    pub headers: HeaderMap,
    /// Request ID from `x-typesafe-request-id`, when present.
    pub request_id: Option<String>,
}

impl ApiError {
    /// The kind of error, mirroring the error classes of the official SDKs.
    pub fn kind(&self) -> ApiErrorKind {
        match self.status.as_u16() {
            400 => ApiErrorKind::BadRequest,
            401 => ApiErrorKind::Authentication,
            403 => ApiErrorKind::PermissionDenied,
            404 => ApiErrorKind::NotFound,
            422 => ApiErrorKind::UnprocessableEntity,
            429 => ApiErrorKind::RateLimit,
            529 => ApiErrorKind::Overloaded,
            500..=599 => ApiErrorKind::InternalServer,
            _ => ApiErrorKind::Other,
        }
    }

    /// The server-requested delay before retrying, parsed from the
    /// `retry-after-ms` (milliseconds) or `Retry-After` (integer seconds)
    /// headers. HTTP-date `Retry-After` values are not parsed.
    pub fn retry_after(&self) -> Option<Duration> {
        if let Some(value) = self.headers.get("retry-after-ms") {
            if let Some(ms) = value
                .to_str()
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
            {
                return Some(Duration::from_millis(ms));
            }
        }
        if let Some(value) = self.headers.get("retry-after") {
            if let Some(secs) = value
                .to_str()
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
            {
                if secs.is_finite() && secs >= 0.0 {
                    return Some(Duration::from_secs_f64(secs));
                }
            }
        }
        None
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeSafe API error ({}): {}", self.status, self.message)?;
        if let Some(request_id) = &self.request_id {
            write!(f, " (request id: {request_id})")?;
        }
        Ok(())
    }
}

impl std::error::Error for ApiError {}

/// The kind of an [`ApiError`], mirroring the error classes of the official SDKs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ApiErrorKind {
    /// 400 Bad Request.
    BadRequest,
    /// 401 Unauthorized: missing or invalid API key.
    Authentication,
    /// 403 Forbidden.
    PermissionDenied,
    /// 404 Not Found.
    NotFound,
    /// 422 Unprocessable Entity: the request body failed validation.
    UnprocessableEntity,
    /// 429 Too Many Requests: rate limit exceeded; back off and retry.
    RateLimit,
    /// 529 Overloaded: TypeSafe is temporarily overloaded; retry after a
    /// short delay.
    Overloaded,
    /// Any other 5xx server error.
    InternalServer,
    /// Any other status code.
    Other,
}

/// Classify a `reqwest` transport error as a [`Error::Timeout`] or
/// [`Error::Connection`].
fn transport_error(error: reqwest::Error, timeout: Duration) -> Error {
    if error.is_timeout() {
        Error::Timeout {
            timeout,
            source: error,
        }
    } else {
        Error::Connection(error)
    }
}

/// Build a human-readable message from an error response body.
fn error_message(body: &Option<Value>, raw: &str) -> String {
    if let Some(message) = body.as_ref().and_then(extract_message) {
        return message;
    }
    let raw = raw.trim();
    if raw.is_empty() {
        "the API returned an error with no details".to_owned()
    } else {
        let mut message: String = raw.chars().take(500).collect();
        if raw.chars().count() > 500 {
            message.push('…');
        }
        message
    }
}

/// Find a message inside an error body: the first of the keys `message`,
/// `error`, `detail`, or `msg` (recursively, as in `{"detail": [{"msg": ...}]}`).
fn extract_message(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => ["message", "error", "detail", "msg"]
            .iter()
            .find_map(|key| map.get(*key))
            .and_then(extract_message),
        Value::Array(items) => items.iter().find_map(extract_message),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

/// Retry configuration, mirroring `RetryPolicy` in the official SDKs.
///
/// A request is retried when the API returns one of `retry_statuses` (by
/// default 408, 429, and 500–599) or the transport fails (connection error or
/// per-attempt timeout). Between attempts the client waits with exponential
/// backoff: the first retry waits `backoff_initial`, doubling on every
/// subsequent retry up to `backoff_max`, with a random fraction of up to
/// `backoff_jitter` subtracted to avoid thundering herds. When the server asks
/// for a specific delay with `Retry-After` / `retry-after-ms`, that delay is
/// used instead, capped at `max_retry_after`.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum retries after the initial attempt; `0` disables retries.
    pub max_retries: u32,
    /// Delay before the first retry; doubled on every subsequent retry.
    pub backoff_initial: Duration,
    /// Upper bound on the backoff delay.
    pub backoff_max: Duration,
    /// Fraction (0.0–1.0) of each backoff delay that is randomly subtracted.
    pub backoff_jitter: f64,
    /// Whether to honor the `Retry-After` / `retry-after-ms` headers.
    pub respect_retry_after: bool,
    /// Upper bound on server-requested delays; longer ones fall back to
    /// backoff.
    pub max_retry_after: Duration,
    /// Whether to retry connection failures.
    pub retry_connection_errors: bool,
    /// Whether to retry per-attempt timeouts.
    pub retry_timeout_errors: bool,
    /// HTTP status codes that are retried.
    pub retry_statuses: Vec<u16>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            backoff_initial: Duration::from_millis(500),
            backoff_max: Duration::from_millis(5_000),
            backoff_jitter: 0.25,
            respect_retry_after: true,
            max_retry_after: Duration::from_secs(60),
            retry_connection_errors: true,
            retry_timeout_errors: true,
            retry_statuses: default_retry_statuses(),
        }
    }
}

impl RetryPolicy {
    /// Set [`max_retries`](Self::max_retries); `0` disables retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    fn status_is_retryable(&self, status: u16) -> bool {
        self.retry_statuses.contains(&status)
    }

    fn should_retry(&self, error: &Error) -> bool {
        match error {
            Error::Api(error) => self.status_is_retryable(error.status.as_u16()),
            Error::Connection(_) => self.retry_connection_errors,
            Error::Timeout { .. } => self.retry_timeout_errors,
            Error::Config(_) | Error::InvalidRequest(_) | Error::Json(_) => false,
        }
    }

    /// How long to wait before retry number `retry` (0-based, i.e. after the
    /// `retry + 1`-th failed attempt).
    fn delay_for(&self, retry: u32, error: &Error) -> Duration {
        if self.respect_retry_after {
            if let Error::Api(error) = error {
                if let Some(delay) = error.retry_after() {
                    if delay <= self.max_retry_after {
                        return delay;
                    }
                }
            }
        }
        self.backoff(retry)
    }

    /// Exponential backoff for retry number `retry` (0-based), with jitter.
    fn backoff(&self, retry: u32) -> Duration {
        let mut delay = self.backoff_initial;
        for _ in 0..retry {
            delay = delay.saturating_mul(2);
            if delay >= self.backoff_max {
                delay = self.backoff_max;
                break;
            }
        }
        let jitter = self.backoff_jitter.clamp(0.0, 1.0);
        let factor = 1.0 - jitter * random_01();
        let ms = (delay.as_millis() as f64 * factor).round();
        Duration::from_millis(ms as u64)
    }
}

fn default_retry_statuses() -> Vec<u16> {
    let mut statuses = vec![408, 429];
    statuses.extend(500..=599);
    statuses
}

/// A cheap pseudo-random number in `[0, 1)` for retry jitter.
///
/// Jitter does not need cryptographic quality, so this is a small xorshift
/// generator seeded from the clock, avoiding a dependency on `rand`.
fn random_01() -> f64 {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x2545_F491_4F6C_DD1D)
        | 1;
    let mut x = seed;
    // xorshift64*
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
}

// ---------------------------------------------------------------------------
// Questions
// ---------------------------------------------------------------------------

/// A typed question to ask about the request's
/// [`state`](SystemOneRequest::state).
///
/// Build one with [`Question::noul`], [`Question::choice`],
/// [`Question::choice_with_criteria`], or [`Question::score`]. Each question
/// type has a matching [`Answer`] type.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// A yes/no question; the answer is the probability of "yes".
    Noul {
        instructions: Description,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// Picks one option from a set you define.
    Choice {
        instructions: Description,
        criteria: BTreeMap<String, Option<Description>>,
    },
    /// Rates the state along an ordered rubric of levels.
    Score {
        instructions: Description,
        criteria: Vec<Description>,
    },
}

/// Optional descriptions of what "yes" and "no" mean for a [`Question::noul`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct NoulCriteria {
    /// What a yes (noul value near 1) means.
    #[serde(rename = "true", skip_serializing_if = "Option::is_none")]
    pub true_description: Option<Description>,
    /// What a no (noul value near 0) means.
    #[serde(rename = "false", skip_serializing_if = "Option::is_none")]
    pub false_description: Option<Description>,
}

impl Question {
    /// A yes/no question. The answer is the probability the answer is yes,
    /// from 0 (no) to 1 (yes).
    ///
    /// Optionally describe the two outcomes with
    /// [`criteria_true`](Self::criteria_true) and
    /// [`criteria_false`](Self::criteria_false).
    pub fn noul(instructions: impl Into<Description>) -> Result<Self, Error> {
        let instructions = instructions.into();
        validate_description(&instructions, false, "instructions")?;
        Ok(Question::Noul {
            instructions,
            criteria: None,
        })
    }

    /// Describe what a yes (noul value near 1) means. Only valid on noul
    /// questions.
    pub fn criteria_true(self, description: impl Into<Description>) -> Result<Self, Error> {
        let description = description.into();
        validate_description(&description, false, "criteria description")?;
        match self {
            Question::Noul {
                instructions,
                criteria,
            } => {
                let mut criteria = criteria.unwrap_or_default();
                criteria.true_description = Some(description);
                Ok(Question::Noul {
                    instructions,
                    criteria: Some(criteria),
                })
            }
            _ => Err(Error::InvalidRequest(
                "criteria_true is only valid on noul questions".to_owned(),
            )),
        }
    }

    /// Describe what a no (noul value near 0) means. Only valid on noul
    /// questions.
    pub fn criteria_false(self, description: impl Into<Description>) -> Result<Self, Error> {
        let description = description.into();
        validate_description(&description, false, "criteria description")?;
        match self {
            Question::Noul {
                instructions,
                criteria,
            } => {
                let mut criteria = criteria.unwrap_or_default();
                criteria.false_description = Some(description);
                Ok(Question::Noul {
                    instructions,
                    criteria: Some(criteria),
                })
            }
            _ => Err(Error::InvalidRequest(
                "criteria_false is only valid on noul questions".to_owned(),
            )),
        }
    }

    /// A question that picks one option from a set you define.
    ///
    /// `options` is an iterator of `(name, description)` pairs, where every
    /// option has a description. Use [`Question::choice_with_criteria`] when
    /// some options need no description, and prefer an `other` / `none of the
    /// above` option when the list might not cover every input. A choice
    /// accepts at most 255 options.
    ///
    /// The answer is the highest-probability option, the full probability
    /// distribution, and a confidence value.
    pub fn choice<K, V, I>(instructions: impl Into<Description>, options: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<Description>,
    {
        Self::choice_with_criteria(
            instructions,
            options
                .into_iter()
                .map(|(name, description)| (name, Some(description))),
        )
    }

    /// A question that picks one option from a set, where an option's
    /// description may be `None` (serialized as JSON `null`) when its name is
    /// clear on its own.
    ///
    /// Pass `None::<&str>` (or similar) for undescribed options.
    pub fn choice_with_criteria<K, V, I>(
        instructions: impl Into<Description>,
        criteria: I,
    ) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (K, Option<V>)>,
        K: Into<String>,
        V: Into<Description>,
    {
        let instructions = instructions.into();
        validate_description(&instructions, false, "instructions")?;

        let mut options: BTreeMap<String, Option<Description>> = BTreeMap::new();
        let count = criteria
            .into_iter()
            .try_fold(0usize, |count, (name, description)| {
                let name: String = name.into();
                if name.trim().is_empty() {
                    return Err(Error::InvalidRequest(
                        "option names must not be empty".to_owned(),
                    ));
                }
                let description = description.map(|description| description.into());
                if let Some(description) = &description {
                    validate_description(description, false, "option descriptions")?;
                }
                options.insert(name, description);
                Ok(count + 1)
            })?;

        if options.is_empty() {
            return Err(Error::InvalidRequest(
                "a choice question needs at least one option".to_owned(),
            ));
        }
        if options.len() != count {
            return Err(Error::InvalidRequest(
                "duplicate option name in criteria".to_owned(),
            ));
        }
        if options.len() > 255 {
            return Err(Error::InvalidRequest(
                "a choice question accepts at most 255 options".to_owned(),
            ));
        }
        Ok(Question::Choice {
            instructions,
            criteria: options,
        })
    }

    /// A question that rates the state along an ordered rubric of levels,
    /// from the low end of the scale to the high end.
    ///
    /// `levels` needs between 2 and 10 entries; each level's number is its
    /// position in the array, starting at 0. Describe situations, not degrees:
    /// "Broken feature, but a workaround exists" works; "moderately severe"
    /// does not.
    ///
    /// The answer is a probability-weighted position along the levels (it can
    /// land between two levels), a legend mapping each level back to its
    /// description, per-level probabilities, and a confidence value.
    pub fn score<V, I>(instructions: impl Into<Description>, levels: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = V>,
        V: Into<Description>,
    {
        let instructions = instructions.into();
        validate_description(&instructions, false, "instructions")?;

        let mut criteria = Vec::new();
        for (level, description) in levels.into_iter().enumerate() {
            let description = description.into();
            validate_description(&description, false, &format!("score level {level}"))?;
            criteria.push(description);
        }
        if criteria.len() < 2 || criteria.len() > 10 {
            return Err(Error::InvalidRequest(format!(
                "a score question needs between 2 and 10 levels, got {}",
                criteria.len()
            )));
        }
        Ok(Question::Score {
            instructions,
            criteria,
        })
    }
}

/// Check that a description is a string, object, or array (and not empty when
/// it is a string), matching the API's accepted types.
fn validate_description(
    description: &Description,
    allow_null: bool,
    what: &str,
) -> Result<(), Error> {
    match description {
        Value::Null if allow_null => Ok(()),
        Value::Null => Err(Error::InvalidRequest(format!("{what} must not be null"))),
        Value::String(text) if text.trim().is_empty() => {
            Err(Error::InvalidRequest(format!("{what} must not be empty")))
        }
        Value::String(_) | Value::Object(_) | Value::Array(_) => Ok(()),
        value => Err(Error::InvalidRequest(format!(
            "{what} must be a string, object, or array, got {value}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// A System One request: [`state`](Self::state) plus named
/// [`questions`](Self::questions), sent by [`TypeSafeClient::system_one`].
#[derive(Debug, Clone)]
pub struct SystemOneRequest {
    /// The content to evaluate: a plain string, structured JSON, or
    /// `Value::Null`.
    pub state: EntryType,
    /// Model override for this request; falls back to the client's
    /// `default_model` (by default `jev-latest`).
    pub model: Option<String>,
    /// One or more questions, keyed by the ids their answers are returned
    /// under. The ids are not sent to the model.
    pub questions: BTreeMap<String, Question>,
}

impl SystemOneRequest {
    /// Create a request from a state and an iterator of `(id, question)`
    /// pairs. Ask every question your code might need in one request;
    /// questions are evaluated in parallel and extra questions cost only a
    /// few tokens each.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when `questions` is empty or contains
    /// duplicate or empty ids.
    pub fn new<K, I>(state: impl Into<EntryType>, questions: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = (K, Question)>,
        K: Into<String>,
    {
        let mut map = BTreeMap::new();
        let count = questions
            .into_iter()
            .try_fold(0usize, |count, (id, question)| {
                let id = id.into();
                if id.trim().is_empty() {
                    return Err(Error::InvalidRequest(
                        "question ids must not be empty".to_owned(),
                    ));
                }
                map.insert(id, question);
                Ok(count + 1)
            })?;
        if map.is_empty() {
            return Err(Error::InvalidRequest(
                "a request needs at least one question".to_owned(),
            ));
        }
        if map.len() != count {
            return Err(Error::InvalidRequest("duplicate question id".to_owned()));
        }
        Ok(Self {
            state: state.into(),
            model: None,
            questions: map,
        })
    }

    /// Override the model for this request, e.g. `"jev-latest"` or a
    /// versioned id such as `"jev-1.13.0"`. Pin a version when you have
    /// tuned thresholds against it.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

/// Wire form of a System One request, with the model resolved.
#[derive(Serialize)]
struct SystemOneBody<'a> {
    state: &'a EntryType,
    model: &'a str,
    questions: &'a BTreeMap<String, Question>,
}

// ---------------------------------------------------------------------------
// Results and answers
// ---------------------------------------------------------------------------

/// A System One response: the model that answered, one [`Answer`] per
/// question, and token usage.
#[derive(Debug, Clone, Deserialize)]
pub struct SystemOneResult {
    /// The model that performed the evaluation (a versioned ID when the
    /// request used an alias).
    pub model: String,
    /// Answers keyed by the question ids from the request.
    pub answers: BTreeMap<String, Answer>,
    /// Token usage for the request. Input tokens are charged; output tokens
    /// are free.
    pub usage: Usage,
}

impl SystemOneResult {
    /// The answer for a question id.
    pub fn answer(&self, id: &str) -> Option<&Answer> {
        self.answers.get(id)
    }

    /// The yes-probability of a noul question's answer, if `id` names a noul
    /// question.
    pub fn noul(&self, id: &str) -> Option<f64> {
        self.answers.get(id).and_then(Answer::as_noul)
    }

    /// A choice question's answer, if `id` names a choice question.
    pub fn choice(&self, id: &str) -> Option<&ChoiceAnswer> {
        self.answers.get(id).and_then(Answer::as_choice)
    }

    /// A score question's answer, if `id` names a score question.
    pub fn score(&self, id: &str) -> Option<&ScoreAnswer> {
        self.answers.get(id).and_then(Answer::as_score)
    }
}

/// Token usage for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub struct Usage {
    /// Number of input tokens charged for the request.
    pub input_tokens: u64,
    /// Number of output tokens (free of charge).
    pub output_tokens: u64,
}

/// The typed answer to a [`Question`], carrying the same `type` as its
/// question.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    /// The answer to a [`Question::noul`].
    Noul(NoulAnswer),
    /// The answer to a [`Question::choice`].
    Choice(ChoiceAnswer),
    /// The answer to a [`Question::score`].
    Score(ScoreAnswer),
}

impl Answer {
    /// The yes-probability if this is a noul answer.
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Answer::Noul(answer) => Some(answer.noul),
            _ => None,
        }
    }

    /// The full answer if this is a choice answer.
    pub fn as_choice(&self) -> Option<&ChoiceAnswer> {
        match self {
            Answer::Choice(answer) => Some(answer),
            _ => None,
        }
    }

    /// The full answer if this is a score answer.
    pub fn as_score(&self) -> Option<&ScoreAnswer> {
        match self {
            Answer::Score(answer) => Some(answer),
            _ => None,
        }
    }
}

// Deserialized by hand rather than with `#[serde(tag = "type")]` so that score
// answers can key `probabilities` and `legend` by integer level, the way the
// official Python SDK does. (serde's internally-tagged buffering cannot turn
// string map keys into integers.)
impl<'de> Deserialize<'de> for Answer {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("answer is missing a string `type` field"))?;
        match kind {
            "noul" => serde_json::from_value(value)
                .map(Answer::Noul)
                .map_err(serde::de::Error::custom),
            "choice" => serde_json::from_value(value)
                .map(Answer::Choice)
                .map_err(serde::de::Error::custom),
            "score" => serde_json::from_value(value)
                .map(Answer::Score)
                .map_err(serde::de::Error::custom),
            other => Err(serde::de::Error::custom(format!(
                "unknown answer type `{other}`"
            ))),
        }
    }
}

/// The answer to a [`Question::noul`]: the probability of "yes".
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NoulAnswer {
    /// The yes/no answer on a scale from 0 (no) to 1 (yes).
    pub noul: f64,
}

/// The answer to a [`Question::choice`]: the chosen option plus the full
/// probability distribution and a confidence value.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChoiceAnswer {
    /// The highest-probability option.
    pub choice: String,
    /// Every option mapped to its probability (floats that sum to 1).
    pub probabilities: BTreeMap<String, f64>,
    /// How certain the model is, derived from how `probabilities` is spread;
    /// 0 to 1.
    pub confidence: f64,
}

impl ChoiceAnswer {
    /// The probability of a specific option.
    pub fn probability(&self, option: &str) -> Option<f64> {
        self.probabilities.get(option).copied()
    }
}

/// The answer to a [`Question::score`]: a position along the levels, which
/// can land between two levels.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ScoreAnswer {
    /// The probability-weighted position across the levels, from 0 to the
    /// top level number (`criteria.len() - 1` on the question).
    pub score: f64,
    /// Each level number mapped back to its description.
    pub legend: BTreeMap<u32, Description>,
    /// Each level mapped to its probability (floats that sum to 1).
    pub probabilities: BTreeMap<u32, f64>,
    /// How certain the model is, derived from how `probabilities` is spread;
    /// 0 to 1.
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

/// A model or alias the account can send in the `model` field, from
/// [`TypeSafeClient::list_models`].
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelCard {
    /// The model ID or alias, as accepted by the `model` field.
    pub name: String,
    /// What the model is for.
    pub description: String,
    /// When the model or alias was released.
    pub release_date: String,
}

#[derive(Deserialize)]
struct ModelList {
    models: Vec<ModelCard>,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Client for the TypeSafe AI API.
///
/// Configuration precedence matches the official SDKs: explicit options, then
/// environment variables, then SDK defaults. Empty or whitespace-only
/// environment values are ignored. Build one with
/// [`TypeSafeClient::builder`] or [`TypeSafeClient::from_env`].
///
/// The client is cheap to clone; clones share the underlying connection pool.
#[derive(Debug, Clone)]
pub struct TypeSafeClient {
    http: Client,
    base_url: String,
    auth_header: HeaderValue,
    default_model: String,
    default_headers: HeaderMap,
    timeout: Duration,
    retry: RetryPolicy,
}

impl TypeSafeClient {
    /// A builder for a [`TypeSafeClient`].
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// A client configured from environment variables and SDK defaults:
    /// [`ENV_API_KEY`] is required; [`ENV_BASE_URL`] and
    /// [`ENV_DEFAULT_MODEL`] are optional.
    pub fn from_env() -> Result<Self, Error> {
        Self::builder().build()
    }

    /// The API root, without a trailing slash.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The model used when a request omits one.
    pub fn default_model(&self) -> &str {
        &self.default_model
    }

    /// The timeout per attempt.
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The retry policy in use.
    pub fn retry(&self) -> &RetryPolicy {
        &self.retry
    }

    /// Additional headers sent with every request.
    pub fn default_headers(&self) -> &HeaderMap {
        &self.default_headers
    }

    /// Answer named questions about text or structured state.
    ///
    /// This is the main endpoint of the System One API (`POST
    /// /v1/systemone`). Every question is evaluated in parallel against the
    /// state, and the response carries one typed [`Answer`] per question,
    /// keyed by the ids you chose. The request is retried according to the
    /// client's [`RetryPolicy`] when the API returns a retryable status
    /// (408, 429, 5xx) or the transport fails.
    pub async fn system_one(&self, request: SystemOneRequest) -> Result<SystemOneResult, Error> {
        if request.questions.is_empty() {
            return Err(Error::InvalidRequest(
                "a request needs at least one question".to_owned(),
            ));
        }
        let model = request.model.as_deref().unwrap_or(&self.default_model);
        let body = SystemOneBody {
            state: &request.state,
            model,
            questions: &request.questions,
        };
        let payload = serde_json::to_string(&body)?;
        self.execute(Method::POST, SYSTEMONE_PATH, Some(payload))
            .await
    }

    /// List the models and aliases the account can send in the `model` field
    /// (`GET /v1/models`), with a description and release date for each.
    ///
    /// The list currently contains the aliases; versioned IDs are accepted by
    /// the `model` field whether or not they appear in the list.
    pub async fn list_models(&self) -> Result<Vec<ModelCard>, Error> {
        self.execute::<ModelList>(Method::GET, MODELS_PATH, None)
            .await
            .map(|list| list.models)
    }

    /// Send a request with retries. The payload, if any, is the
    /// already-serialized JSON body.
    async fn execute<T>(
        &self,
        method: Method,
        path: &str,
        payload: Option<String>,
    ) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        let url = format!("{}{path}", self.base_url);
        let mut headers = HeaderMap::new();
        headers.insert(AUTHORIZATION, self.auth_header.clone());
        headers.insert(USER_AGENT, HeaderValue::from_static(USER_AGENT_VALUE));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        if payload.is_some() {
            headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        }
        // User-supplied headers override the built-ins.
        for (name, value) in self.default_headers.iter() {
            headers.insert(name.clone(), value.clone());
        }

        let mut retry = 0;
        loop {
            let error = match self
                .attempt(&url, &method, &headers, payload.as_deref())
                .await
            {
                Ok(value) => return Ok(value),
                Err(error) => error,
            };
            if retry >= self.retry.max_retries || !self.retry.should_retry(&error) {
                return Err(error);
            }
            let delay = self.retry.delay_for(retry, &error);
            tokio::time::sleep(delay).await;
            retry += 1;
        }
    }

    /// One HTTP round trip: send the request, read the whole response, and
    /// decode it into `T`.
    async fn attempt<T>(
        &self,
        url: &str,
        method: &Method,
        headers: &HeaderMap,
        payload: Option<&str>,
    ) -> Result<T, Error>
    where
        T: DeserializeOwned,
    {
        let mut request = self
            .http
            .request(method.clone(), url)
            .timeout(self.timeout)
            .headers(headers.clone());
        if let Some(payload) = payload {
            request = request.body(payload.to_owned());
        }

        let response = request
            .send()
            .await
            .map_err(|error| transport_error(error, self.timeout))?;
        let status = response.status();
        let response_headers = response.headers().clone();
        let bytes = response
            .bytes()
            .await
            .map_err(|error| transport_error(error, self.timeout))?;

        if !status.is_success() {
            let body = serde_json::from_slice::<Value>(&bytes).ok();
            let message = error_message(&body, &String::from_utf8_lossy(&bytes));
            let request_id = response_headers
                .get("x-typesafe-request-id")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned);
            return Err(Error::Api(Box::new(ApiError {
                status,
                message,
                body,
                headers: response_headers,
                request_id,
            })));
        }

        serde_json::from_slice(&bytes).map_err(Error::Json)
    }
}

// ---------------------------------------------------------------------------
// Client builder
// ---------------------------------------------------------------------------

/// Builder for [`TypeSafeClient`]. Start with [`TypeSafeClient::builder`].
#[derive(Debug)]
pub struct ClientBuilder {
    api_key: Option<String>,
    base_url: Option<String>,
    default_model: Option<String>,
    timeout: Option<Duration>,
    default_headers: Option<HeaderMap>,
    retry: Option<RetryPolicy>,
    http: Option<Client>,
}

impl ClientBuilder {
    fn new() -> Self {
        Self {
            api_key: None,
            base_url: None,
            default_model: None,
            timeout: None,
            default_headers: None,
            retry: None,
            http: None,
        }
    }

    /// Set the API key; falls back to [`ENV_API_KEY`].
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the API root; falls back to [`ENV_BASE_URL`], then
    /// [`DEFAULT_BASE_URL`]. Any trailing slash is removed.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model used when a request omits one; falls back to
    /// [`ENV_DEFAULT_MODEL`], then [`DEFAULT_MODEL`].
    pub fn default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Set the timeout per attempt (default: [`DEFAULT_TIMEOUT`]). There is
    /// no total retry budget; each attempt gets the full timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set headers sent with every request. These override the client's
    /// built-in `Authorization` and `User-Agent` headers on collision.
    pub fn default_headers(mut self, headers: HeaderMap) -> Self {
        self.default_headers = Some(headers);
        self
    }

    /// Set the retry policy (default: [`RetryPolicy::default`]).
    pub fn retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = Some(retry);
        self
    }

    /// Use a custom `reqwest::Client` (for transport configuration or tests).
    pub fn http_client(mut self, http: Client) -> Self {
        self.http = Some(http);
        self
    }

    /// Build the client.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when the API key is missing, the base URL is
    /// invalid, or the HTTP client cannot be built.
    pub fn build(self) -> Result<TypeSafeClient, Error> {
        let api_key = match self.api_key {
            Some(key) => key,
            None => env_value(ENV_API_KEY).ok_or_else(|| {
                Error::Config(format!(
                    "missing API key: set the {ENV_API_KEY} environment variable or pass ClientBuilder::api_key"
                ))
            })?,
        };
        let auth_header = HeaderValue::from_str(&format!("Bearer {api_key}")).map_err(|_| {
            Error::Config(
                "the API key contains characters that are not valid in an HTTP header".to_owned(),
            )
        })?;

        let base_url = match self.base_url {
            Some(base_url) => base_url,
            None => env_value(ENV_BASE_URL).unwrap_or_else(|| DEFAULT_BASE_URL.to_owned()),
        };
        let base_url = normalize_base_url(&base_url)?;

        let default_model = match self.default_model {
            Some(model) => model,
            None => env_value(ENV_DEFAULT_MODEL).unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
        };

        let http = match self.http {
            Some(http) => http,
            None => Client::builder().build().map_err(|error| {
                Error::Config(format!("could not build the HTTP client: {error}"))
            })?,
        };

        Ok(TypeSafeClient {
            http,
            base_url,
            auth_header,
            default_model,
            default_headers: self.default_headers.unwrap_or_default(),
            timeout: self.timeout.unwrap_or(DEFAULT_TIMEOUT),
            retry: self.retry.unwrap_or_default(),
        })
    }
}

/// Read an environment variable, ignoring empty or whitespace-only values.
fn env_value(name: &str) -> Option<String> {
    env::var(name).ok().and_then(|value| {
        let trimmed = value.trim().to_owned();
        (!trimmed.is_empty()).then_some(trimmed)
    })
}

/// Validate a base URL and strip its trailing slash.
fn normalize_base_url(base_url: &str) -> Result<String, Error> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let url = Url::parse(trimmed)
        .map_err(|error| Error::Config(format!("invalid base URL {base_url:?}: {error}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::Config(format!(
            "invalid base URL {base_url:?}: the scheme must be http or https"
        )));
    }
    Ok(trimmed.to_owned())
}

// ---------------------------------------------------------------------------
// Tests (offline; the API is mocked with a tiny in-process HTTP server)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::Mutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// All tests that touch environment variables hold this lock, and no
    /// other code reads them while a test runs.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn fast_retry() -> RetryPolicy {
        RetryPolicy {
            max_retries: 2,
            backoff_initial: Duration::from_millis(1),
            backoff_max: Duration::from_millis(2),
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        }
    }

    fn test_client(base_url: String) -> TypeSafeClient {
        TypeSafeClient::builder()
            .api_key("test-key")
            .base_url(base_url)
            .retry(fast_retry())
            .build()
            .unwrap()
    }

    fn request() -> SystemOneRequest {
        SystemOneRequest::new("state under test", [("q", Question::noul("yes?").unwrap())]).unwrap()
    }

    // ---- mock HTTP server --------------------------------------------------

    struct MockResponse {
        status: u16,
        headers: Vec<(&'static str, String)>,
        body: &'static str,
        delay: Duration,
    }

    impl MockResponse {
        fn ok(body: &'static str) -> Self {
            Self {
                status: 200,
                headers: Vec::new(),
                body,
                delay: Duration::ZERO,
            }
        }

        fn error(status: u16, body: &'static str) -> Self {
            Self {
                status,
                headers: Vec::new(),
                body,
                delay: Duration::ZERO,
            }
        }
    }

    struct MockServer {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    impl MockServer {
        fn request_count(&self) -> usize {
            self.requests.lock().unwrap().len()
        }

        fn request(&self, index: usize) -> String {
            self.requests.lock().unwrap()[index].clone()
        }
    }

    /// Serve the given responses in order (repeating the last one for
    /// further requests) and record every request received.
    async fn spawn_mock(responses: Vec<MockResponse>) -> MockServer {
        assert!(!responses.is_empty());
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        tokio::spawn(async move {
            let mut next = 0usize;
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let received = read_request(&mut socket).await;
                seen.lock().unwrap().push(received);
                let response = &responses[next.min(responses.len() - 1)];
                next += 1;
                if response.delay > Duration::ZERO {
                    tokio::time::sleep(response.delay).await;
                }
                write_response(&mut socket, response).await;
            }
        });
        MockServer {
            url: format!("http://{addr}"),
            requests,
        }
    }

    /// Read one full HTTP request (headers plus any Content-Length body).
    async fn read_request(socket: &mut TcpStream) -> String {
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let n = socket.read(&mut chunk).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
            let Some(header_end) = find(&buf, b"\r\n\r\n") else {
                continue;
            };
            let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
            let content_length = headers
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= header_end + 4 + content_length {
                break;
            }
        }
        String::from_utf8_lossy(&buf).into_owned()
    }

    async fn write_response(socket: &mut TcpStream, response: &MockResponse) {
        let mut out = format!(
            "HTTP/1.1 {} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n",
            response.status,
            reason(response.status),
            response.body.len(),
        );
        for (name, value) in &response.headers {
            out.push_str(&format!("{name}: {value}\r\n"));
        }
        out.push_str("\r\n");
        out.push_str(response.body);
        let _ = socket.write_all(out.as_bytes()).await;
        let _ = socket.shutdown().await;
    }

    fn reason(status: u16) -> &'static str {
        match status {
            200 => "OK",
            401 => "Unauthorized",
            422 => "Unprocessable Entity",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            529 => "Overloaded",
            _ => "Unknown",
        }
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    // ---- question construction and serialization ---------------------------

    #[test]
    fn noul_question_serializes() {
        let question = Question::noul("Does this convey urgency?").unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({"type": "noul", "instructions": "Does this convey urgency?"})
        );
    }

    #[test]
    fn noul_criteria_is_omitted_without_descriptions() {
        let question = Question::noul("Is it urgent?").unwrap();
        let value = serde_json::to_value(&question).unwrap();
        assert!(value.get("criteria").is_none());
    }

    #[test]
    fn noul_criteria_descriptions_rename_to_true_and_false() {
        let question = Question::noul("Is it urgent?")
            .unwrap()
            .criteria_true("Explicitly time-sensitive")
            .unwrap()
            .criteria_false("No urgency expressed")
            .unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({
                "type": "noul",
                "instructions": "Is it urgent?",
                "criteria": {
                    "true": "Explicitly time-sensitive",
                    "false": "No urgency expressed"
                }
            })
        );
    }

    #[test]
    fn noul_criteria_can_be_partial() {
        let question = Question::noul("q").unwrap().criteria_true("yes").unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({"type": "noul", "instructions": "q", "criteria": {"true": "yes"}})
        );
    }

    #[test]
    fn choice_question_serializes() {
        let question = Question::choice(
            "Which team should handle this?",
            [
                ("billing", "Payments, invoicing, refunds"),
                ("technical", "Bugs, outages, integrations"),
                ("sales", "Pricing, upgrades, new accounts"),
            ],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments, invoicing, refunds",
                    "technical": "Bugs, outages, integrations",
                    "sales": "Pricing, upgrades, new accounts"
                }
            })
        );
    }

    #[test]
    fn choice_options_can_have_null_descriptions() {
        let question = Question::choice_with_criteria(
            "What is the customer's tone?",
            [("calm", None::<&str>), ("angry", Some("strong language"))],
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({
                "type": "choice",
                "instructions": "What is the customer's tone?",
                "criteria": {"calm": null, "angry": "strong language"}
            })
        );
    }

    #[test]
    fn score_question_serializes() {
        let question =
            Question::score("How frustrated?", ["Calm", "Frustrated", "Very angry"]).unwrap();
        assert_eq!(
            serde_json::to_value(&question).unwrap(),
            json!({
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": ["Calm", "Frustrated", "Very angry"]
            })
        );
    }

    #[test]
    fn structured_descriptions_pass_through() {
        let question = Question::score(
            "How severe is the reported issue?",
            [
                json!({"what": "Cosmetic", "examples": ["typo in a label"]}),
                json!({"what": "Blocking", "examples": ["data loss"]}),
            ],
        )
        .unwrap();
        let value = serde_json::to_value(&question).unwrap();
        assert_eq!(
            value["criteria"],
            json!([
                {"what": "Cosmetic", "examples": ["typo in a label"]},
                {"what": "Blocking", "examples": ["data loss"]}
            ])
        );
    }

    #[test]
    fn rejects_invalid_questions() {
        // Instructions must be a non-empty string, object, or array.
        assert!(Question::noul("   ").is_err());
        assert!(Question::noul(json!(null)).is_err());
        assert!(Question::noul(json!(42)).is_err());
        assert!(Question::noul(json!({"question": "Is it urgent?"})).is_ok());

        // Score: between 2 and 10 levels.
        assert!(Question::score("q", ["only-one"]).is_err());
        assert!(Question::score("q", Vec::<String>::new()).is_err());
        let eleven: Vec<String> = (0..11).map(|i| i.to_string()).collect();
        assert!(Question::score("q", eleven).is_err());

        // Choice: at least one, at most 255, uniquely named options.
        assert!(Question::choice("q", Vec::<(&str, &str)>::new()).is_err());
        assert!(Question::choice("q", [("", "described")]).is_err());
        assert!(
            Question::choice("q", [("a", "1"), ("a", "2")]).is_err(),
            "duplicate option names are rejected"
        );
        let many: Vec<(String, &str)> = (0..256).map(|i| (i.to_string(), "d")).collect();
        assert!(Question::choice("q", many).is_err());

        // Noul criteria setters are only valid on noul questions.
        let choice = Question::choice("q", [("a", "b")]).unwrap();
        assert!(choice.criteria_true("x").is_err());
    }

    #[test]
    fn rejects_invalid_requests() {
        assert!(SystemOneRequest::new("state", Vec::<(&str, Question)>::new()).is_err());
        let question = Question::noul("q").unwrap();
        assert!(
            SystemOneRequest::new("state", [("a", question.clone()), ("a", question.clone())])
                .is_err()
        );
        assert!(SystemOneRequest::new("state", [("", question.clone())]).is_err());

        // State may be null.
        let request = SystemOneRequest::new(Value::Null, [("a", question)]).unwrap();
        assert!(request.state.is_null());
    }

    // ---- response parsing --------------------------------------------------

    const NOUL_RESPONSE: &str = r#"{
        "model": "jev-latest",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.92}},
        "usage": {"input_tokens": 312, "output_tokens": 48}
    }"#;

    const FULL_RESPONSE: &str = r#"{
        "model": "jev-latest",
        "answers": {
            "is_urgent": {"type": "noul", "noul": 0.92},
            "department": {
                "type": "choice",
                "choice": "technical",
                "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
                "confidence": 0.82
            },
            "frustration": {
                "type": "score",
                "score": 1.6,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
                "confidence": 0.78
            }
        },
        "usage": {"input_tokens": 312, "output_tokens": 48}
    }"#;

    #[test]
    fn parses_noul_response() {
        let result: SystemOneResult = serde_json::from_str(NOUL_RESPONSE).unwrap();
        assert_eq!(result.model, "jev-latest");
        assert_eq!(
            result.usage,
            Usage {
                input_tokens: 312,
                output_tokens: 48
            }
        );
        assert_eq!(result.noul("is_urgent"), Some(0.92));
        assert_eq!(result.choice("is_urgent"), None);
        assert!(matches!(result.answers["is_urgent"], Answer::Noul(_)));
    }

    #[test]
    fn parses_choice_response() {
        let result: SystemOneResult = serde_json::from_str(FULL_RESPONSE).unwrap();
        let answer = result.choice("department").unwrap();
        assert_eq!(answer.choice, "technical");
        assert_eq!(answer.confidence, 0.82);
        assert_eq!(answer.probability("billing"), Some(0.08));
        assert_eq!(answer.probability("nonexistent"), None);
        assert!(matches!(result.answers["department"], Answer::Choice(_)));
    }

    #[test]
    fn parses_score_response_keyed_by_level() {
        let result: SystemOneResult = serde_json::from_str(FULL_RESPONSE).unwrap();
        let answer = result.score("frustration").unwrap();
        assert_eq!(answer.score, 1.6);
        assert_eq!(answer.confidence, 0.78);
        assert_eq!(
            answer.legend,
            BTreeMap::from([
                (0, json!("Calm")),
                (1, json!("Frustrated")),
                (2, json!("Very angry"))
            ])
        );
        assert_eq!(
            answer.probabilities,
            BTreeMap::from([(0, 0.05), (1, 0.3), (2, 0.65)])
        );
        assert!(matches!(result.answers["frustration"], Answer::Score(_)));
    }

    #[test]
    fn parses_structured_score_legend() {
        let text = r#"{
            "model": "jev-latest",
            "answers": {"q": {
                "type": "score",
                "score": 1.06,
                "confidence": 0.91,
                "legend": {
                    "0": {"what": "Cosmetic", "examples": ["typo in a label"]},
                    "1": {"what": "Broken", "examples": ["export fails in one browser"]}
                },
                "probabilities": {"0": 0.06, "1": 0.94}
            }},
            "usage": {"input_tokens": 379, "output_tokens": 18}
        }"#;
        let result: SystemOneResult = serde_json::from_str(text).unwrap();
        let answer = result.score("q").unwrap();
        assert_eq!(answer.legend[&1]["what"], "Broken");
        assert_eq!(answer.probabilities[&1], 0.94);
    }

    #[test]
    fn rejects_unknown_answer_type() {
        let text = r#"{
            "model": "jev-latest",
            "answers": {"q": {"type": "weird"}},
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }"#;
        assert!(serde_json::from_str::<SystemOneResult>(text).is_err());
    }

    // ---- retry policy ------------------------------------------------------

    #[test]
    fn backoff_doubles_and_caps() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        assert_eq!(policy.backoff(0), Duration::from_millis(500));
        assert_eq!(policy.backoff(1), Duration::from_millis(1000));
        assert_eq!(policy.backoff(2), Duration::from_millis(2000));
        assert_eq!(policy.backoff(3), Duration::from_millis(4000));
        assert_eq!(policy.backoff(4), Duration::from_millis(5000));
        assert_eq!(policy.backoff(10), Duration::from_millis(5000));
    }

    #[test]
    fn backoff_jitter_stays_in_bounds() {
        let policy = RetryPolicy::default(); // jitter 0.25
        for retry in 0..5u32 {
            let base_ms = 500.0 * (2f64).powi(retry as i32).min(10.0);
            let delay = policy.backoff(retry).as_millis() as f64;
            assert!(
                delay >= base_ms * 0.75 && delay <= base_ms,
                "retry {retry}: delay {delay} outside ({}, {base_ms}]",
                base_ms * 0.75
            );
        }
    }

    fn api_error_with_headers(status: u16, headers: &[(&str, &str)]) -> ApiError {
        let mut map = HeaderMap::new();
        for (name, value) in headers {
            map.insert(
                reqwest::header::HeaderName::from_bytes(name.as_bytes()).unwrap(),
                HeaderValue::from_str(value).unwrap(),
            );
        }
        ApiError {
            status: StatusCode::from_u16(status).unwrap(),
            message: "m".to_owned(),
            body: None,
            headers: map,
            request_id: None,
        }
    }

    #[test]
    fn parses_retry_after_headers() {
        assert_eq!(
            api_error_with_headers(429, &[("retry-after-ms", "120")]).retry_after(),
            Some(Duration::from_millis(120))
        );
        assert_eq!(
            api_error_with_headers(429, &[("Retry-After", "2")]).retry_after(),
            Some(Duration::from_secs(2))
        );
        assert_eq!(
            api_error_with_headers(429, &[("retry-after", "1.5")]).retry_after(),
            Some(Duration::from_millis(1500))
        );
        // HTTP-date Retry-After values are not parsed; fall back to backoff.
        assert_eq!(
            api_error_with_headers(429, &[("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")])
                .retry_after(),
            None
        );
        assert_eq!(api_error_with_headers(429, &[]).retry_after(), None);
    }

    #[test]
    fn retry_delay_prefers_retry_after_header() {
        let policy = RetryPolicy {
            backoff_jitter: 0.0,
            ..RetryPolicy::default()
        };
        let error = Error::Api(Box::new(api_error_with_headers(
            429,
            &[("retry-after-ms", "25")],
        )));
        assert_eq!(policy.delay_for(3, &error), Duration::from_millis(25));

        // Server delays beyond max_retry_after fall back to backoff.
        let long = Error::Api(Box::new(api_error_with_headers(
            429,
            &[("retry-after", "3600")],
        )));
        assert_eq!(policy.delay_for(0, &long), Duration::from_millis(500));

        // Non-API errors fall back to backoff.
        assert_eq!(
            policy.delay_for(0, &Error::InvalidRequest("x".to_owned())),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn status_retryability() {
        let policy = RetryPolicy::default();
        for status in [408u16, 429, 500, 503, 529, 599] {
            assert!(policy.status_is_retryable(status), "{status} should retry");
        }
        for status in [400u16, 401, 403, 404, 422] {
            assert!(
                !policy.status_is_retryable(status),
                "{status} should not retry"
            );
        }
    }

    #[test]
    fn error_kinds() {
        let kind = |status: u16| api_error_with_headers(status, &[]).kind();
        assert_eq!(kind(400), ApiErrorKind::BadRequest);
        assert_eq!(kind(401), ApiErrorKind::Authentication);
        assert_eq!(kind(403), ApiErrorKind::PermissionDenied);
        assert_eq!(kind(404), ApiErrorKind::NotFound);
        assert_eq!(kind(422), ApiErrorKind::UnprocessableEntity);
        assert_eq!(kind(429), ApiErrorKind::RateLimit);
        assert_eq!(kind(500), ApiErrorKind::InternalServer);
        assert_eq!(kind(529), ApiErrorKind::Overloaded);
        assert_eq!(kind(418), ApiErrorKind::Other);
    }

    #[test]
    fn extracts_error_messages() {
        assert_eq!(
            error_message(&Some(json!({"message": "state is required"})), ""),
            "state is required"
        );
        assert_eq!(
            error_message(&Some(json!({"detail": [{"msg": "bad field"}]})), ""),
            "bad field"
        );
        assert_eq!(
            error_message(&Some(json!({"error": {"message": "nested"}})), ""),
            "nested"
        );
        assert_eq!(
            error_message(&None, "  "),
            "the API returned an error with no details"
        );
        let long = "x".repeat(600);
        assert_eq!(error_message(&None, &long).chars().count(), 501); // 500 + ellipsis
    }

    // ---- configuration -----------------------------------------------------

    #[test]
    fn missing_api_key_is_a_config_error() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: every test that touches the environment holds ENV_LOCK.
        unsafe { std::env::remove_var(ENV_API_KEY) };
        match TypeSafeClient::builder().build() {
            Err(Error::Config(message)) => {
                assert!(
                    message.contains(ENV_API_KEY),
                    "message should name the env var: {message}"
                );
            }
            other => panic!("expected Error::Config, got {other:?}"),
        }
    }

    #[test]
    fn env_configuration_is_read() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: every test that touches the environment holds ENV_LOCK.
        unsafe {
            std::env::set_var(ENV_API_KEY, "env-key");
            std::env::set_var(ENV_DEFAULT_MODEL, "jev-preview");
        }
        let client = TypeSafeClient::from_env().unwrap();
        assert_eq!(client.default_model(), "jev-preview");
        unsafe {
            std::env::remove_var(ENV_API_KEY);
            std::env::remove_var(ENV_DEFAULT_MODEL);
        }
        assert!(TypeSafeClient::from_env().is_err());
    }

    #[test]
    fn invalid_base_url_is_rejected() {
        let error = TypeSafeClient::builder()
            .api_key("k")
            .base_url("not a url")
            .build()
            .unwrap_err();
        assert!(matches!(error, Error::Config(_)));

        let error = TypeSafeClient::builder()
            .api_key("k")
            .base_url("ftp://example.com")
            .build()
            .unwrap_err();
        assert!(matches!(error, Error::Config(_)));
    }

    #[test]
    fn base_url_trailing_slash_is_trimmed() {
        let client = TypeSafeClient::builder()
            .api_key("k")
            .base_url("https://proxy.example/api/")
            .build()
            .unwrap();
        assert_eq!(client.base_url(), "https://proxy.example/api");
        assert_eq!(client.default_model(), DEFAULT_MODEL);
        assert_eq!(client.timeout(), DEFAULT_TIMEOUT);
    }

    // ---- end to end against the mock server --------------------------------

    #[tokio::test]
    async fn system_one_sends_and_parses() {
        let server = spawn_mock(vec![MockResponse::ok(FULL_RESPONSE)]).await;
        let client = test_client(server.url.clone());
        let request = SystemOneRequest::new(
            "Help! My payouts have been failing for 3 days.",
            [
                (
                    "is_urgent",
                    Question::noul("Does this convey urgency?").unwrap(),
                ),
                (
                    "department",
                    Question::choice("Which team should handle this?", [("technical", "Bugs")])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();

        let result = client.system_one(request).await.unwrap();

        assert_eq!(result.model, "jev-latest");
        assert_eq!(result.noul("is_urgent"), Some(0.92));
        assert_eq!(result.choice("department").unwrap().choice, "technical");

        assert_eq!(server.request_count(), 1);
        let sent = server.request(0);
        assert!(sent.starts_with("POST /v1/systemone HTTP/1.1"), "{sent}");
        assert!(
            sent.contains("authorization: Bearer test-key\r\n"),
            "{sent}"
        );
        assert!(sent.contains("user-agent: typesafe-rust/"), "{sent}");
        let body: Value = serde_json::from_str(sent.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(
            body["state"],
            "Help! My payouts have been failing for 3 days."
        );
        assert_eq!(body["model"], "jev-latest"); // the client default
        assert_eq!(body["questions"]["is_urgent"]["type"], "noul");
        assert_eq!(
            body["questions"]["department"]["criteria"]["technical"],
            "Bugs"
        );
    }

    #[tokio::test]
    async fn model_override_is_sent() {
        let server = spawn_mock(vec![MockResponse::ok(NOUL_RESPONSE)]).await;
        let client = test_client(server.url.clone());
        client
            .system_one(request().model("jev-1.13.0"))
            .await
            .unwrap();
        let sent = server.request(0);
        let body: Value = serde_json::from_str(sent.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["model"], "jev-1.13.0");
    }

    #[tokio::test]
    async fn null_state_is_sent() {
        let server = spawn_mock(vec![MockResponse::ok(NOUL_RESPONSE)]).await;
        let client = test_client(server.url.clone());
        client
            .system_one(
                SystemOneRequest::new(Value::Null, [("q", Question::noul("q").unwrap())]).unwrap(),
            )
            .await
            .unwrap();
        let sent = server.request(0);
        let body: Value = serde_json::from_str(sent.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["state"], Value::Null);
    }

    #[tokio::test]
    async fn retries_on_500_then_succeeds() {
        let server = spawn_mock(vec![
            MockResponse::error(500, r#"{"error": "overloaded"}"#),
            MockResponse::ok(NOUL_RESPONSE),
        ])
        .await;
        let client = test_client(server.url.clone());
        let result = client.system_one(request()).await.unwrap();
        // The mock returns a fixed body, so the answer id is the fixture's.
        assert_eq!(result.noul("is_urgent"), Some(0.92));
        assert_eq!(server.request_count(), 2);
    }

    #[tokio::test]
    async fn retries_on_529_overloaded() {
        let server = spawn_mock(vec![
            MockResponse::error(529, r#"{"error": "overloaded"}"#),
            MockResponse::ok(NOUL_RESPONSE),
        ])
        .await;
        let client = test_client(server.url.clone());
        assert!(client.system_one(request()).await.is_ok());
        assert_eq!(server.request_count(), 2);
    }

    #[tokio::test]
    async fn honors_retry_after_ms_header() {
        let mut response = MockResponse::error(429, r#"{"error": "rate limited"}"#);
        response.headers.push(("retry-after-ms", "5".to_owned()));
        let server = spawn_mock(vec![response, MockResponse::ok(NOUL_RESPONSE)]).await;
        let client = test_client(server.url.clone());
        assert!(client.system_one(request()).await.is_ok());
        assert_eq!(server.request_count(), 2);
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let server = spawn_mock(vec![MockResponse::error(500, "{}")]).await;
        let client = test_client(server.url.clone()); // max_retries = 2
        let error = client.system_one(request()).await.unwrap_err();
        match error {
            Error::Api(error) => assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR),
            other => panic!("expected Error::Api, got {other:?}"),
        }
        assert_eq!(server.request_count(), 3, "initial attempt + 2 retries");
    }

    #[tokio::test]
    async fn does_not_retry_401() {
        let server = spawn_mock(vec![MockResponse::error(
            401,
            r#"{"message": "invalid api key"}"#,
        )])
        .await;
        let client = test_client(server.url.clone());
        let error = client.system_one(request()).await.unwrap_err();
        let Error::Api(error) = &error else {
            panic!("expected Error::Api, got {error:?}")
        };
        assert_eq!(error.kind(), ApiErrorKind::Authentication);
        assert_eq!(error.message, "invalid api key");
        assert_eq!(server.request_count(), 1);
    }

    #[tokio::test]
    async fn reports_422_body_details() {
        let server = spawn_mock(vec![MockResponse::error(
            422,
            r#"{"detail": [{"msg": "state is required"}]}"#,
        )])
        .await;
        let client = test_client(server.url.clone());
        let error = client.system_one(request()).await.unwrap_err();
        let Error::Api(error) = &error else {
            panic!("expected Error::Api, got {error:?}")
        };
        assert_eq!(error.kind(), ApiErrorKind::UnprocessableEntity);
        assert_eq!(error.message, "state is required");
        assert_eq!(error.request_id, None);
        assert_eq!(server.request_count(), 1, "422 is not retryable");
    }

    #[tokio::test]
    async fn reports_request_id_header() {
        let mut response = MockResponse::error(401, "{}");
        response
            .headers
            .push(("x-typesafe-request-id", "req-123".to_owned()));
        let server = spawn_mock(vec![response]).await;
        let client = test_client(server.url.clone());
        let error = client.system_one(request()).await.unwrap_err();
        let Error::Api(error) = &error else {
            panic!("expected Error::Api, got {error:?}")
        };
        assert_eq!(error.request_id.as_deref(), Some("req-123"));
        assert!(error.to_string().contains("req-123"), "{}", error);
    }

    #[tokio::test]
    async fn connection_errors_are_retried_then_reported() {
        // Pick a port that refuses connections.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);

        let client = test_client(format!("http://{addr}"));
        let error = client.system_one(request()).await.unwrap_err();
        assert!(matches!(error, Error::Connection(_)), "got {error:?}");
    }

    #[tokio::test]
    async fn timeouts_are_retried_then_reported() {
        let server = spawn_mock(vec![MockResponse {
            status: 200,
            headers: Vec::new(),
            body: "{}",
            delay: Duration::from_millis(400),
        }])
        .await;
        let client = TypeSafeClient::builder()
            .api_key("k")
            .base_url(server.url.clone())
            .timeout(Duration::from_millis(50))
            .retry(fast_retry())
            .build()
            .unwrap();
        let error = client.system_one(request()).await.unwrap_err();
        assert!(matches!(error, Error::Timeout { .. }), "got {error:?}");
    }

    #[tokio::test]
    async fn lists_models() {
        let server = spawn_mock(vec![MockResponse::ok(
            r#"{
                "models": [
                    {
                        "name": "jev-latest",
                        "description": "The most recent stable, official release",
                        "release_date": "2026-01-15"
                    }
                ]
            }"#,
        )])
        .await;
        let client = test_client(server.url.clone());
        let models = client.list_models().await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "jev-latest");
        assert_eq!(models[0].release_date, "2026-01-15");
        let sent = server.request(0);
        assert!(sent.starts_with("GET /v1/models HTTP/1.1"), "{sent}");
        assert!(
            sent.contains("authorization: Bearer test-key\r\n"),
            "{sent}"
        );
    }

    #[tokio::test]
    async fn rejects_empty_questions_before_sending() {
        let server = spawn_mock(vec![MockResponse::ok(NOUL_RESPONSE)]).await;
        let client = test_client(server.url.clone());
        let error = client
            .system_one(SystemOneRequest {
                state: Value::String("state".to_owned()),
                model: None,
                questions: BTreeMap::new(),
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidRequest(_)));
        assert_eq!(server.request_count(), 0);
    }
}
