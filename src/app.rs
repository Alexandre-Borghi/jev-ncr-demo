use std::time::Duration;

use leptos::leptos_dom::helpers::debounce;
use leptos::prelude::*;
use leptos_meta::{provide_meta_context, MetaTags, Stylesheet, Title};
use leptos_router::{
    components::{Route, Router, Routes},
    StaticSegment,
};
use serde::{Deserialize, Serialize};

pub fn shell(options: LeptosOptions) -> impl IntoView {
    view! {
        <!DOCTYPE html>
        <html lang="en">
            <head>
                <meta charset="utf-8"/>
                <meta name="viewport" content="width=device-width, initial-scale=1"/>
                <AutoReload options=options.clone() />
                <HydrationScripts options/>
                <MetaTags/>
            </head>
            <body>
                <App/>
            </body>
        </html>
    }
}

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    view! {
        // injects a stylesheet into the document <head>
        // id=leptos means cargo-leptos will hot-reload this stylesheet
        <Stylesheet id="leptos" href="/pkg/jev-ncr-demo.css"/>

        // sets the document title
        <Title text="NCR - Non-Conformance Report"/>

        // content for this welcome page
        <Router>
            <main>
                <Routes fallback=|| "Page not found.".into_view()>
                    <Route path=StaticSegment("") view=NcrPage/>
                </Routes>
            </main>
        </Router>
    }
}

/// A defect code suggested by the AI, as sent to the browser.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SuggestedDefectCode {
    pub id: String,
    pub title: String,
    pub description: String,
    /// Whether the model auto-selected this code (confidence at or above the
    /// auto-select threshold).
    pub selected: bool,
    /// Confidence that this code matches the description, from 0 to 1.
    pub confidence: f64,
}

/// Match an NCR description against the defect code catalog. On the server
/// this runs `crate::ncr::suggest_defect_codes` (the TypeSafe AI call) and
/// maps the result into [`SuggestedDefectCode`]s, best match first; on the
/// client the `#[server]` macro turns this into an HTTP call to the server.
/// A blank description returns an empty list.
#[server]
pub async fn suggest_defect_codes(
    ncr_description: String,
) -> Result<Vec<SuggestedDefectCode>, ServerFnError> {
    let suggestion = crate::ncr::suggest_defect_codes(&ncr_description)
        .await
        .map_err(ServerFnError::new)?;

    Ok(suggestion
        .defect_codes
        .into_iter()
        .map(|item| SuggestedDefectCode {
            id: item.defect_code.id,
            title: item.defect_code.title,
            description: item.defect_code.description,
            selected: item.selected,
            confidence: item.confidence,
        })
        .collect())
}

/// NCR Form state
#[derive(Clone, Debug)]
struct NcrForm {
    date: String,
    time: String,
    location: String,
    description: String,
    selected_defects: Vec<String>,
    suspected_cause: String,
    affected_part: String,
    severity: String,
    recommended_action: String,
}

/// Renders the NCR creation page
#[component]
fn NcrPage() -> impl IntoView {
    // Form state
    let form = RwSignal::new(NcrForm {
        date: "Sept 19, 2026".to_string(),
        time: "07:24 PM".to_string(),
        location: "Line 3 - Assembly".to_string(),
        description: "During final inspection, found a scratch on the outer surface of the metal housing. Scratch is about 3 cm long and visible without magnification.".to_string(),
        selected_defects: vec![],
        suspected_cause: "Handling / Transportation".to_string(),
        affected_part: "Housing (Metal)".to_string(),
        severity: "Medium".to_string(),
        recommended_action: "Rework / Refinish".to_string(),
    });

    // ---- AI analysis ---------------------------------------------------------

    /// How long to wait after typing stops before (re-)analyzing.
    const DEBOUNCE_MS: u64 = 500;
    /// How many suggested codes to show before "Show all".
    const VISIBLE_SUGGESTIONS: usize = 5;

    // The description to analyze: a debounced copy of the form's description,
    // so the API is only asked once typing pauses.
    let debounced_description = RwSignal::new(form.get_untracked().description.clone());
    let mut set_debounced_description = debounce(
        Duration::from_millis(DEBOUNCE_MS),
        move |description: String| {
            if debounced_description.get_untracked() != description {
                debounced_description.set(description);
            }
        },
    );
    Effect::new(move |_| {
        let description = form.get().description;
        set_debounced_description(description);
    });

    // Ask the server for suggestions. A LocalResource runs only in the
    // browser, so the server-rendered page paints the analyzing state and
    // the first fetch happens right after hydration. Reading
    // `debounced_description` inside the fetcher makes the resource
    // refetch whenever the debounced description changes.
    //
    // LocalResource has no loading indicator, so `is_fetching` tracks it
    // here, with a generation counter so a slow earlier fetch cannot clear
    // the flag while a newer one is still in flight.
    let is_fetching = RwSignal::new(true);
    let fetch_generation = RwSignal::new(0u32);
    let analysis = LocalResource::new(move || {
        debounced_description.get();
        let generation = fetch_generation.get_untracked() + 1;
        fetch_generation.set(generation);
        is_fetching.set(true);
        async move {
            let description = debounced_description.get_untracked();
            let result = suggest_defect_codes(description).await;
            if fetch_generation.get_untracked() == generation {
                is_fetching.set(false);
            }
            result
        }
    });

    let suggestions = Memo::new(move |_| analysis.get().and_then(Result::ok));
    let analysis_error = Memo::new(move |_| analysis.get().and_then(Result::err));

    // Auto-check the codes the model is confident about whenever a new
    // analysis arrives (manual changes are overwritten on re-analysis).
    Effect::new(move |_| {
        let Some(suggestions) = suggestions.get() else {
            return;
        };
        let selected = suggestions
            .iter()
            .filter(|code| code.selected)
            .map(|code| code.id.clone())
            .collect::<Vec<_>>();
        let mut current_form = form.get_untracked();
        current_form.selected_defects = selected;
        form.set(current_form);
    });

    // How many codes to show before the "Show all" toggle.
    let show_all = RwSignal::new(false);
    let visible_codes = Memo::new(move |_| {
        let take = if show_all.get() {
            usize::MAX
        } else {
            VISIBLE_SUGGESTIONS
        };
        suggestions
            .get()
            .map(|codes| codes.into_iter().take(take).collect::<Vec<_>>())
            .unwrap_or_default()
    });

    let auto_selected_count = move || {
        suggestions
            .get()
            .map(|codes| codes.iter().filter(|code| code.selected).count())
            .unwrap_or(0)
    };

    // Toggle defect selection
    let toggle_defect = move |defect_id: String| {
        let mut current_form = form.get_untracked();
        if current_form.selected_defects.contains(&defect_id) {
            current_form.selected_defects.retain(|id| id != &defect_id);
        } else {
            current_form.selected_defects.push(defect_id);
        }
        form.set(current_form);
    };

    // Check if defect is selected
    let is_selected =
        move |defect_id: String| -> bool { form.with(|f| f.selected_defects.contains(&defect_id)) };

    // Character count for description
    let char_count = move || form.with(|f| f.description.len());
    let max_chars = 1000;

    view! {
        <div class="ncr-container">
            // Sidebar
            <Sidebar/>

            // Main Content Area
            <div class="main-content">
                // Header
                <div class="page-header">
                    <div class="header-left">
                        <h1>"Create Non-Conformance Report (NCR)"</h1>
                        <p class="subtitle">"Describe the issue and let AI suggest relevant defect codes and other details."</p>
                    </div>
                    <div class="header-right">
                        <div class="company-selector">
                            <svg xmlns="http://www.w3.org/2000/svg" class="company-icon" viewBox="0 0 100 100" width="100" height="100">
                                <path d="M25 85V25H75V85" fill="none" stroke="#111827" stroke-width="5" stroke-linejoin="round"/>
                                <path d="M25 40H75M25 55H75M25 70H75" fill="none" stroke="#111827" stroke-width="4"/>
                                <path d="M38 85V70H62V85" fill="none" stroke="#111827" stroke-width="5"/>
                            </svg>
                            <span>"Borghi Manufacturing"</span>
                            <svg width="12" height="8" viewBox="0 0 12 8" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M1 1L6 6L11 1" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="user-avatar">"AB"</div>
                    </div>
                </div>

                // Main Form Area
                <div class="form-area">
                    // Left Column - Form
                    <div class="form-column">
                        // Date & Time and Location
                        <div class="form-row">
                            <div class="form-group">
                                <label>"Date & Time"</label>
                                <div class="datetime-input">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" stroke="#999" stroke-width="2"/>
                                        <line x1="16" y1="2" x2="16" y2="6" stroke="#999" stroke-width="2"/>
                                        <line x1="8" y1="2" x2="8" y2="6" stroke="#999" stroke-width="2"/>
                                        <line x1="3" y1="10" x2="21" y2="10" stroke="#999" stroke-width="2"/>
                                    </svg>
                                    <input
                                        type="text"
                                        value=move || form.with(|f| f.date.clone())
                                        on:input=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.date = event_target_value(&ev);
                                            form.set(f);
                                        }
                                        placeholder="Select date"
                                    />
                                    <input
                                        type="text"
                                        value=move || form.with(|f| f.time.clone())
                                        on:input=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.time = event_target_value(&ev);
                                            form.set(f);
                                        }
                                        placeholder="Select time"
                                    />
                                </div>
                            </div>
                            <div class="form-group">
                                <label>"Location / Line"</label>
                                <div class="location-input">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" stroke="#999" stroke-width="2"/>
                                        <circle cx="12" cy="10" r="3" stroke="#999" stroke-width="2"/>
                                    </svg>
                                    <select
                                        on:change=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.location = event_target_value(&ev);
                                            form.set(f);
                                        }
                                    >
                                        <option value="Line 3 - Assembly" selected=true>"Line 3 - Assembly"</option>
                                        <option value="Line 1 - Production">"Line 1 - Production"</option>
                                        <option value="Line 2 - Quality">"Line 2 - Quality"</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        // Problem Description
                        <div class="form-group full-width">
                            <label>"Problem Description (required)"</label>
                            <textarea
                                prop:value=move || form.with(|f| f.description.clone())
                                on:input=move |ev| {
                                    let mut f = form.get_untracked();
                                    f.description = event_target_value(&ev);
                                    form.set(f);
                                }
                                placeholder="Describe the issue in detail..."
                                rows="4"
                            />
                            <div class="char-count">
                                {char_count} " / " {max_chars}
                            </div>
                        </div>

                        // AI Analysis Section
                        <div class="ai-section" class:has-error=move || analysis_error.get().is_some()>
                            <div class="ai-header">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M2 17l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M2 12l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                <div class="ai-status">
                                    <span class="ai-label">
                                        {move || match (analysis_error.get(), suggestions.get()) {
                                            (Some(_), _) => "AI analysis failed",
                                            (None, Some(_)) => "AI Autofill Complete",
                                            (None, None) => "Analyzing description...",
                                        }}
                                    </span>
                                    <span class="ai-subtitle">
                                        {move || if let Some(error) = analysis_error.get() {
                                            error.to_string()
                                        } else if suggestions.get().is_some() {
                                            format!("Found {} high-confidence defect codes and additional details.", auto_selected_count())
                                        } else {
                                            "Suggesting defect codes and related information.".to_string()
                                        }}
                                    </span>
                                </div>
                            </div>
                            <Show when=move || is_fetching.get()>
                                <div class="progress-bar">
                                    <div class="progress-fill indeterminate">""</div>
                                </div>
                            </Show>
                        </div>

                        // Suggested Defect Codes
                        <div class="defect-section">
                            <div class="section-header">
                                <h3>"Suggested Defect Codes"</h3>
                                <span class="selected-count">
                                    {move || form.with(|f| f.selected_defects.len())} " selected"
                                </span>
                            </div>
                            <div class="defect-list">
                                <For
                                    each=move || visible_codes.get()
                                    // Ugly hack: add the confidence to the key
                                    // so it also gets rerendered. f64 doesn't
                                    // implement Eq so we round it to a u32.
                                    // TODO: signal re-render on confidence change
                                    key=|code| (code.id.clone(), (code.confidence * 100.0).round() as u32)
                                    children=move |code| {
                                        let code_id = code.id.clone();
                                        let code_id_click = code.id.clone();
                                        let code_id_check = code.id.clone();
                                        view! {
                                            <div
                                                class="defect-item"
                                                class:selected=move || is_selected(code_id.clone())
                                                title=move || code.description.clone()
                                                on:click=move |_| toggle_defect(code_id_click.clone())
                                            >
                                                <input
                                                    type="checkbox"
                                                    prop:checked=move || is_selected(code_id_check.clone())
                                                />
                                                <div class="defect-info">
                                                    <span class="defect-id">{move || code.id.clone()}</span>
                                                    <span class="defect-name">{move || code.title.clone()}</span>
                                                </div>
                                                <div class="confidence-badge">
                                                    {move || (code.confidence * 100.0).round()}"%"
                                                </div>
                                            </div>
                                        }
                                    }
                                />
                            </div>
                            <Show when=move || suggestions.get().is_some_and(|codes| codes.is_empty())>
                                <div class="defect-empty">"Describe the issue to get suggestions."</div>
                            </Show>
                            <Show when=move || {
                                suggestions.get().is_some_and(|codes| codes.len() > VISIBLE_SUGGESTIONS)
                            }>
                                <button class="show-more-btn" on:click=move |_| show_all.update(|show| *show = !*show)>
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <line x1="12" y1="5" x2="12" y2="19" stroke="#4F7CFF" stroke-width="2"/>
                                        <line x1="5" y1="12" x2="19" y2="12" stroke="#4F7CFF" stroke-width="2"/>
                                    </svg>
                                    {move || if show_all.get() {
                                        " Show fewer suggestions".to_string()
                                    } else {
                                        format!(
                                            " Show all {} suggestions",
                                            suggestions.get().map(|codes| codes.len()).unwrap_or(0)
                                        )
                                    }}
                                </button>
                            </Show>
                        </div>

                        // Additional Information (auto-filled)
                        <div class="additional-info-section">
                            <div class="section-header">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M2 17l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                    <path d="M2 12l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                                <h3>"Additional Information (auto-filled)"</h3>
                            </div>
                            <div class="info-grid">
                                <div class="info-group">
                                    <label>"Suspected Cause"</label>
                                    <select
                                        on:change=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.suspected_cause = event_target_value(&ev);
                                            form.set(f);
                                        }
                                    >
                                        <option value="Handling / Transportation" selected=true>"Handling / Transportation"</option>
                                        <option value="Manufacturing Error">"Manufacturing Error"</option>
                                        <option value="Material Defect">"Material Defect"</option>
                                    </select>
                                </div>
                                <div class="info-group">
                                    <label>"Affected Part"</label>
                                    <select
                                        on:change=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.affected_part = event_target_value(&ev);
                                            form.set(f);
                                        }
                                    >
                                        <option value="Housing (Metal)" selected=true>"Housing (Metal)"</option>
                                        <option value="Component A">"Component A"</option>
                                        <option value="Component B">"Component B"</option>
                                    </select>
                                </div>
                                <div class="info-group">
                                    <label>"Severity"</label>
                                    <select
                                        on:change=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.severity = event_target_value(&ev);
                                            form.set(f);
                                        }
                                    >
                                        <option value="Medium" selected=true>"Medium"</option>
                                        <option value="Low">"Low"</option>
                                        <option value="High">"High"</option>
                                        <option value="Critical">"Critical"</option>
                                    </select>
                                </div>
                                <div class="info-group">
                                    <label>"Recommended Action"</label>
                                    <select
                                        on:change=move |ev| {
                                            let mut f = form.get_untracked();
                                            f.recommended_action = event_target_value(&ev);
                                            form.set(f);
                                        }
                                    >
                                        <option value="Rework / Refinish" selected=true>"Rework / Refinish"</option>
                                        <option value="Scrap">"Scrap"</option>
                                        <option value="Return to Supplier">"Return to Supplier"</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        // Action Buttons
                        <div class="action-buttons">
                            <button class="btn btn-secondary">"Clear"</button>
                            <button class="btn btn-primary">"Save NCR"</button>
                        </div>
                    </div>

                    // Right Column - AI Panel
                    <AiPanel form=form suggestions=suggestions/>
                </div>
            </div>
        </div>
    }
}

/// Sidebar component
#[component]
fn Sidebar() -> impl IntoView {
    view! {
        <div class="sidebar">
            <div class="logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 7v10l10 5 10-5V7l-10-5z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 7l10 5 10-5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M12 22V12" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <span>"NCR"</span>
            </div>
            <nav class="nav-menu">
                <a href="#" class="nav-item active">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <polyline points="14,2 14,8 20,8" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <line x1="16" y1="13" x2="8" y2="13" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <line x1="16" y1="17" x2="8" y2="17" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span>"New NCR"</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="#ccc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span>"NCRs"</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" stroke="#ccc" stroke-width="2"/>
                        <path d="M12 6v6l4 2" stroke="#ccc" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span>"Analytics"</span>
                </a>
                <a href="#" class="nav-item">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="3" stroke="#ccc" stroke-width="2"/>
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="#ccc" stroke-width="2"/>
                    </svg>
                    <span>"Settings"</span>
                </a>
            </nav>
            <div class="sidebar-footer">
                <p>"Built by Alexandre Borghi."</p>
                <p>"Powered by TypeSafe AI's Jev."</p>
                <p>"Unofficial demo, not affiliated with TypeSafe AI."</p>
            </div>
        </div>
    }
}

/// AI Panel component
#[component]
fn AiPanel(
    form: RwSignal<NcrForm>,
    suggestions: Memo<Option<Vec<SuggestedDefectCode>>>,
) -> impl IntoView {
    // Get selected defect codes
    let selected_defects = move || {
        let codes = suggestions.get().unwrap_or_default();
        form.with(|f| {
            f.selected_defects
                .iter()
                .filter_map(|id| codes.iter().find(|code| &code.id == id).cloned())
                .collect::<Vec<_>>()
        })
    };

    let auto_selected_count = move || {
        suggestions
            .get()
            .map(|codes| codes.iter().filter(|code| code.selected).count())
            .unwrap_or(0)
    };

    view! {
        <div class="ai-panel">
            // AI Complete Header
            <div class="ai-complete-header">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 17l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12l10 5 10-5" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div class="ai-complete-content">
                    <h3>
                        {move || if suggestions.get().is_some() {
                            "AI Autofill Complete"
                        } else {
                            "Analyzing description..."
                        }}
                    </h3>
                    <p>
                        {move || if suggestions.get().is_some() {
                            format!(
                                "Found {} high-confidence defect codes and additional details.",
                                auto_selected_count()
                            )
                        } else {
                            "Suggesting defect codes and related information.".to_string()
                        }}
                    </p>
                </div>
            </div>

            // Selected Defect Codes
            <div class="selected-defects-section">
                <div class="section-header">
                    <h3>"Selected Defect Codes"</h3>
                    <span class="selected-count">
                        {move || selected_defects().len()} " selected"
                    </span>
                </div>
                <div class="selected-defects-list">
                    <For
                        each=move || selected_defects()
                        key=|code| code.id.clone()
                        children=move |code| {
                            let code_id = code.id.clone();
                            view! {
                                <div class="selected-defect-item">
                                    <span class="defect-badge">{code.id.clone()}</span>
                                    <span class="defect-text">{code.title.clone()}</span>
                                    <button
                                        class="remove-btn"
                                        title="Remove"
                                        on:click=move |_| {
                                            let mut current_form = form.get_untracked();
                                            current_form.selected_defects.retain(|id| id != &code_id);
                                            form.set(current_form);
                                        }
                                    >
                                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                            <line x1="18" y1="6" x2="6" y2="18" stroke="white" stroke-width="2"/>
                                            <line x1="6" y1="6" x2="18" y2="18" stroke="white" stroke-width="2"/>
                                        </svg>
                                    </button>
                                </div>
                            }
                        }
                    />
                </div>
            </div>

            // Additional Information
            <div class="ai-additional-info">
                <div class="section-header">
                    <h3>"Additional Information"</h3>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" stroke="#999" stroke-width="2"/>
                        <line x1="12" y1="16" x2="12" y2="12" stroke="#999" stroke-width="2"/>
                        <line x1="12" y1="8" x2="12.01" y2="8" stroke="#999" stroke-width="2"/>
                    </svg>
                </div>
                <div class="info-display">
                    <div class="info-row">
                        <div class="info-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="12" cy="12" r="10" stroke="#666" stroke-width="2"/>
                                <path d="M12 6v6l4 2" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="info-content">
                            <span class="info-label">"Suspected Cause"</span>
                            <span class="info-value">
                                {move || form.with(|f| f.suspected_cause.clone())}
                            </span>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M21.2 15c.7-1.2 1-2.5.7-3.9-.6-2.4-2.4-4.2-4.8-4.8-.9-.3-1.8-.5-2.7-.5-3.3 0-6 2.7-6 6s2.7 6 6 6c1.2 0 2.3-.4 3.3-1z" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M12 13c-1.1 0-2-.9-2-2s.9-2 2-2 2 .9 2 2-.9 2-2 2z" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="info-content">
                            <span class="info-label">"Affected Part"</span>
                            <span class="info-value">
                                {move || form.with(|f| f.affected_part.clone())}
                            </span>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <line x1="12" y1="9" x2="12" y2="13" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <line x1="12" y1="17" x2="12.01" y2="17" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="info-content">
                            <span class="info-label">"Severity"</span>
                            <span class="info-value">
                                {move || form.with(|f| f.severity.clone())}
                            </span>
                        </div>
                    </div>
                    <div class="info-row">
                        <div class="info-icon">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <circle cx="9" cy="7" r="4" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M23 21v-2a4 4 0 0 0-3-3.87" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                <path d="M16 3.13a4 4 0 0 1 0 7.75" stroke="#666" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        <div class="info-content">
                            <span class="info-label">"Recommended Action"</span>
                            <span class="info-value">
                                {move || form.with(|f| f.recommended_action.clone())}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            // AI Disclaimer
            <div class="ai-disclaimer">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" stroke="#4F7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
                <div class="disclaimer-text">
                    <p>"You can edit any of the suggested values before saving."</p>
                    <p>"The AI suggestions are based on your description and may not be 100% accurate."</p>
                </div>
            </div>
        </div>
    }
}
