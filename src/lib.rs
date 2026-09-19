pub mod app;

/// NCR defect-code suggestions via the TypeSafe AI API. Server-side only: the
/// API key comes from the environment and must not ship to the browser, and
/// the HTTP stack only compiles for the server build.
#[cfg(feature = "ssr")]
pub mod ncr;

/// Client for the TypeSafe AI API, used by [`crate::ncr`].
#[cfg(feature = "ssr")]
pub mod typesafe;

#[cfg(feature = "hydrate")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn hydrate() {
    use crate::app::*;
    console_error_panic_hook::set_once();
    leptos::mount::hydrate_body(App);
}
