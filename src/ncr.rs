//! NCR defect-code suggestions powered by TypeSafe AI's System One API (the
//! `jev` model).
//!
//! Every defect code in `defect_codes.json` (compiled into the binary) is
//! turned into one [`Question::noul`] — "is the described non-conformance an
//! instance of this code?" — and all of them are sent in a single
//! [`SystemOneRequest`] with the NCR description as the state. The model
//! answers each question with the probability that the code matches the
//! description; that probability becomes the suggestion's
//! [`confidence`](DefectCodeSuggestion::confidence), and codes at or above
//! [`AUTO_SELECT_THRESHOLD`] are auto-selected.
//!
//! One noul per code — rather than a single choice question — keeps the codes
//! independent, so a description can legitimately match several codes.
//!
//! This module is server-side only: the TypeSafe API key is read from the
//! environment and must not ship to the browser.
//!
//! [`Question::noul`]: crate::typesafe::Question::noul
//! [`SystemOneRequest`]: crate::typesafe::SystemOneRequest
//!
//! # Example
//!
//! ```no_run
//! use jev_ncr_demo::ncr;
//!
//! # async fn example() -> Result<(), jev_ncr_demo::typesafe::Error> {
//! let suggestion = ncr::suggest_defect_codes(
//!     "During final inspection, found a scratch on the housing surface.",
//! )
//! .await?;
//!
//! for item in &suggestion.defect_codes {
//!     if item.selected {
//!         println!(
//!             "{} {} ({:.0}%)",
//!             item.defect_code.id,
//!             item.defect_code.title,
//!             item.confidence * 100.0,
//!         );
//!     }
//! }
//! # Ok(())
//! # }
//! ```

use std::collections::BTreeSet;
use std::sync::LazyLock;

use serde::Deserialize;

use crate::typesafe::{Error, Question, SystemOneRequest, SystemOneResult, TypeSafeClient};

/// A defect code from `defect_codes.json`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DefectCode {
    /// Stable identifier, e.g. `D-001`. Also used as the id of the question
    /// sent to the API.
    pub id: String,
    /// Short human-readable name of the defect code.
    pub title: String,
    /// What kinds of non-conformances the defect code covers.
    pub description: String,
}

/// The defect codes suggested for an NCR description.
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// Every known defect code with its confidence, sorted from most
    /// probable to least probable.
    pub defect_codes: Vec<DefectCodeSuggestion>,
}

/// One defect code matched against an NCR description.
#[derive(Debug, Clone, PartialEq)]
pub struct DefectCodeSuggestion {
    /// The defect code that was evaluated.
    pub defect_code: DefectCode,
    /// Whether this defect code was auto-selected based on the threshold.
    pub selected: bool,
    /// Confidence that this defect code matched the given description.
    pub confidence: f64,
}

/// Auto-select defect codes whose confidence is above or at this threshold.
pub const AUTO_SELECT_THRESHOLD: f64 = 0.8;

/// The defect code catalog, compiled into the binary.
const DEFECT_CODES_JSON: &str = include_str!("../defect_codes.json");

/// The defect codes, parsed and validated once on first use.
static DEFECT_CODES: LazyLock<Vec<DefectCode>> = LazyLock::new(|| {
    parse_defect_codes(DEFECT_CODES_JSON)
        .unwrap_or_else(|message| panic!("invalid defect_codes.json: {message}"))
});

/// The known defect codes, parsed once on first use.
fn defect_codes() -> &'static [DefectCode] {
    &DEFECT_CODES
}

/// Parse and validate a defect code catalog.
///
/// Catalogs need at least one entry, and every entry needs a unique,
/// non-empty `id`, `title`, and `description`.
fn parse_defect_codes(json: &str) -> Result<Vec<DefectCode>, String> {
    let codes: Vec<DefectCode> = serde_json::from_str(json).map_err(|error| error.to_string())?;
    if codes.is_empty() {
        return Err("the catalog needs at least one defect code".to_owned());
    }
    let mut ids = BTreeSet::new();
    for code in &codes {
        if code.id.trim().is_empty()
            || code.title.trim().is_empty()
            || code.description.trim().is_empty()
        {
            return Err(format!(
                "defect code {:?} has an empty id, title, or description",
                code.id
            ));
        }
        if !ids.insert(code.id.as_str()) {
            return Err(format!("duplicate defect code id {:?}", code.id));
        }
    }
    Ok(codes)
}

/// The noul question asking whether the described non-conformance is an
/// instance of `code`. The question id is the defect code's id.
fn question_for(code: &DefectCode) -> Result<Question, Error> {
    Ok(Question::noul(format!(
        "Is the described non-conformance an instance of defect code {} — {}?",
        code.id, code.title
    ))?
    .criteria_true(code.description.clone())?
    .criteria_false("Description points to another defect code, is too vague or unclear to assign this defect code.")?)
}

/// Suggest defect codes for an NCR description using the TypeSafe API.
///
/// Every known defect code is matched against the description in a single
/// System One request — one noul question per code — and ranked by the
/// returned confidence. The client is configured from the environment
/// (`TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`), so
/// this must run on the server.
///
/// A blank description yields an empty suggestion without an API call.
///
/// # Errors
///
/// Returns [`Error`](crate::typesafe::Error) when the client is misconfigured
/// (for example a missing API key), the API rejects the request, or the
/// transport fails after retries.
pub async fn suggest_defect_codes(ncr_description: &str) -> Result<Suggestion, Error> {
    let client = TypeSafeClient::from_env()?;
    suggest_with(&client, ncr_description).await
}

/// [`suggest_defect_codes`] against an explicit client, so tests can point at
/// a mock server.
async fn suggest_with(client: &TypeSafeClient, ncr_description: &str) -> Result<Suggestion, Error> {
    if ncr_description.trim().is_empty() {
        return Ok(Suggestion {
            defect_codes: Vec::new(),
        });
    }

    let codes = defect_codes();
    let questions = codes
        .iter()
        .map(|code| question_for(code).map(|question| (code.id.clone(), question)))
        .collect::<Result<Vec<_>, Error>>()?;
    let request = SystemOneRequest::new(ncr_description, questions)?;
    let result = client.system_one(request).await?;
    Ok(parse_result(&result, codes))
}

/// Turn a System One response into a ranked [`Suggestion`]: one entry per
/// defect code, sorted most probable first.
///
/// Codes the model did not answer (which should not happen) count as
/// confidence `0.0`, and codes with equal confidence keep their catalog
/// order.
fn parse_result(result: &SystemOneResult, codes: &[DefectCode]) -> Suggestion {
    let mut defect_codes: Vec<DefectCodeSuggestion> = codes
        .iter()
        .map(|code| {
            let confidence = result.noul(&code.id).unwrap_or(0.0);
            DefectCodeSuggestion {
                defect_code: code.clone(),
                selected: confidence >= AUTO_SELECT_THRESHOLD,
                confidence,
            }
        })
        .collect();
    defect_codes.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    Suggestion { defect_codes }
}

// ---------------------------------------------------------------------------
// Tests (offline; the API is mocked with a tiny in-process HTTP server)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    // ---- defect code catalog -------------------------------------------------

    #[test]
    fn defect_codes_load_from_the_catalog() {
        let codes = defect_codes();
        assert_eq!(codes.len(), 50);
        assert!(codes.iter().all(|code| {
            !code.id.trim().is_empty()
                && !code.title.trim().is_empty()
                && !code.description.trim().is_empty()
        }));
        let unique: BTreeSet<&str> = codes.iter().map(|code| code.id.as_str()).collect();
        assert_eq!(unique.len(), codes.len());
        assert!(codes.iter().any(|code| code.id == "D-002"));
    }

    #[test]
    fn catalog_validation_rejects_bad_files() {
        let valid = r#"[{"id":"D-1","title":"t","description":"d"}]"#;
        assert!(parse_defect_codes(valid).is_ok());
        assert!(parse_defect_codes("[]").is_err(), "empty catalog");
        assert!(parse_defect_codes("nope").is_err(), "not JSON");
        assert!(
            parse_defect_codes(r#"[{"id":"D-1","title":"t"}]"#).is_err(),
            "missing description field"
        );
        assert!(
            parse_defect_codes(r#"[{"id":"","title":"t","description":"d"}]"#).is_err(),
            "empty id"
        );
        assert!(
            parse_defect_codes(&format!("{valid},{valid}")).is_err(),
            "duplicate ids"
        );
    }

    // ---- question building ---------------------------------------------------

    #[test]
    fn builds_one_noul_question_per_defect_code() {
        let codes = defect_codes();
        let questions = codes
            .iter()
            .map(|code| question_for(code).map(|question| (code.id.clone(), question)))
            .collect::<Result<Vec<_>, Error>>()
            .unwrap();
        // A duplicate id would make the request builder fail.
        let request = SystemOneRequest::new("state under test", questions).unwrap();
        assert_eq!(request.questions.len(), codes.len());

        for code in codes {
            let question = serde_json::to_value(&request.questions[code.id.as_str()]).unwrap();
            assert_eq!(question["type"], "noul", "{}", code.id);
            let instructions = question["instructions"].as_str().unwrap();
            assert!(instructions.contains(&code.id), "{instructions}");
            assert!(instructions.contains(&code.title), "{instructions}");
            assert_eq!(question["criteria"]["true"], code.description);
            assert!(question["criteria"]["false"].as_str().is_some());
        }
    }

    // ---- response parsing ----------------------------------------------------

    #[test]
    fn ranks_and_selects_suggestions() {
        // Answers for three codes; the rest of the catalog is left unanswered.
        let result: SystemOneResult = serde_json::from_str(
            r#"{
                "model": "jev-latest",
                "answers": {
                    "D-002": {"type": "noul", "noul": 0.95},
                    "D-001": {"type": "noul", "noul": 0.8},
                    "D-047": {"type": "noul", "noul": 0.79}
                },
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }"#,
        )
        .unwrap();
        let suggestion = parse_result(&result, defect_codes());

        assert_eq!(suggestion.defect_codes.len(), 50);

        // Sorted most probable first; ties keep catalog order.
        for pair in suggestion.defect_codes.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "{} sorted before {}",
                pair[0].defect_code.id,
                pair[1].defect_code.id
            );
        }

        let top = &suggestion.defect_codes[0];
        assert_eq!(top.defect_code.id, "D-002");
        assert_eq!(top.confidence, 0.95);
        assert!(top.selected);

        // 0.8 is at the threshold and is selected.
        let second = &suggestion.defect_codes[1];
        assert_eq!(second.defect_code.id, "D-001");
        assert_eq!(second.confidence, 0.8);
        assert!(second.selected);

        let third = &suggestion.defect_codes[2];
        assert_eq!(third.defect_code.id, "D-047");
        assert_eq!(third.confidence, 0.79);
        assert!(!third.selected);

        // Unanswered codes fall back to zero confidence; D-050 is last.
        let last = suggestion.defect_codes.last().unwrap();
        assert_eq!(last.defect_code.id, "D-050");
        assert_eq!(last.confidence, 0.0);
        assert!(!last.selected);
    }

    // ---- suggest_with ----------------------------------------------------------

    #[tokio::test]
    async fn blank_description_short_circuits() {
        // Point the client at a port with nothing listening: if the
        // short-circuit failed, the request would error and the unwrap panic.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let client = TypeSafeClient::builder()
            .api_key("k")
            .base_url(format!("http://{addr}"))
            .build()
            .unwrap();

        let suggestion = suggest_with(&client, "  \n\t ").await.unwrap();

        assert!(suggestion.defect_codes.is_empty());
    }

    // ---- end to end against the mock server -----------------------------------

    struct MockServer {
        url: String,
        requests: Arc<Mutex<Vec<String>>>,
    }

    /// Serve `body` for every request and record each request received,
    /// following the mock-server pattern of the offline tests in typesafe.rs.
    async fn spawn_mock(body: String) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => break,
                };
                let received = read_request(&mut socket).await;
                seen.lock().unwrap().push(received);
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
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

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    #[tokio::test]
    async fn suggests_defect_codes_end_to_end() {
        // One answer per defect code, with a few hand-picked confidences.
        let mut answers = serde_json::Map::new();
        for code in defect_codes() {
            let noul = match code.id.as_str() {
                "D-002" => 0.95, // selected
                "D-001" => 0.8,  // selected: exactly at the threshold
                "D-047" => 0.79, // just below
                _ => 0.05,
            };
            answers.insert(code.id.clone(), json!({ "type": "noul", "noul": noul }));
        }
        let response = json!({
            "model": "jev-latest",
            "answers": Value::Object(answers),
            "usage": {"input_tokens": 99, "output_tokens": 7},
        })
        .to_string();

        let server = spawn_mock(response).await;
        let client = TypeSafeClient::builder()
            .api_key("test-key")
            .base_url(server.url.clone())
            .build()
            .unwrap();
        let description =
            "During final inspection, found a scratch on the outer surface of the metal housing.";

        let suggestion = suggest_with(&client, description).await.unwrap();

        // The suggestion reflects the mocked answers.
        assert_eq!(suggestion.defect_codes.len(), 50);
        let top = &suggestion.defect_codes[0];
        assert_eq!(top.defect_code.id, "D-002");
        assert_eq!(top.confidence, 0.95);
        assert!(top.selected);
        let second = &suggestion.defect_codes[1];
        assert_eq!(second.defect_code.id, "D-001");
        assert!(second.selected);
        let third = &suggestion.defect_codes[2];
        assert_eq!(third.defect_code.id, "D-047");
        assert!(!third.selected);
        // The 47 equal-confidence codes keep catalog order, D-050 last.
        let last = suggestion.defect_codes.last().unwrap();
        assert_eq!(last.defect_code.id, "D-050");
        assert_eq!(last.confidence, 0.05);

        // One request was sent: the description as the state, one noul
        // question per defect code, keyed by the defect code ids.
        let sent = server.requests.lock().unwrap()[0].clone();
        assert!(sent.starts_with("POST /v1/systemone HTTP/1.1"), "{sent}");
        assert!(
            sent.contains("authorization: Bearer test-key\r\n"),
            "{sent}"
        );
        let body: Value = serde_json::from_str(sent.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(body["state"], description);
        assert_eq!(body["model"], "jev-latest");
        let questions = body["questions"].as_object().unwrap();
        assert_eq!(questions.len(), 50);
        assert_eq!(questions["D-002"]["type"], "noul");
        let d002 = defect_codes()
            .iter()
            .find(|code| code.id == "D-002")
            .unwrap();
        assert_eq!(questions["D-002"]["criteria"]["true"], d002.description);
        for id in questions.keys() {
            assert!(
                defect_codes().iter().any(|code| code.id == *id),
                "unexpected question id {id}"
            );
        }
    }
}
