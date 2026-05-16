//! Image input validation and decode-size preflight.
//!
//! Responsibilities:
//! - Reject empty `items` and missing `source` up front.
//! - For `file_path`: enforce `allowed_image_dirs` membership via canonical
//!   `Path::starts_with` (component-aware), then load bytes ourselves to
//!   eliminate the TOCTOU window before the embedding backend reopens the
//!   file.
//! - For `url`: fetch over http/https only, with a request timeout, a
//!   redirect cap, and the compressed-size limit enforced both via the
//!   Content-Length header (when present) and while streaming the body.
//!   This is not an SSRF-hardened fetcher: it does not block private /
//!   loopback / link-local / cloud-metadata addresses. Only enable url
//!   inputs from trusted callers.
//! - For all sources: enforce the compressed-size cap and run a
//!   header-only decode-size preflight via `image::ImageReader`.
//!
//! Returns the raw bytes for each input in submission order so the caller
//! can pass them as a single `&[&[u8]]` slice to fastembed's
//! `embed_image_bytes`.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use image::ImageReader;

use crate::error::{MmEmbeddingError, Result};
use crate::protobuf::mm_embedding::{ImageItem, image_item::Source as ImageSource};
use crate::settings::{DecodedImageBytesLimit, ValidatedSettings};

// Fixed limits for url fetches; not exposed as settings in v0.1. 30s is a
// generous upper bound for fetching a single image; 5 redirects covers
// typical CDN hop chains while bounding redirect-loop work. reqwest itself
// rejects redirects to non-http(s) schemes, so a redirect cannot escalate
// to file:// / data:// past the entry scheme check below.
const URL_TIMEOUT: Duration = Duration::from_secs(30);
const URL_MAX_REDIRECTS: usize = 5;

/// Image bytes plus the identifier surfaced in the result `content` field
/// (the item's label, else its file_path, else "image[#i]").
#[derive(Debug)]
pub struct PreparedImage {
    pub bytes: Vec<u8>,
    pub source_hint: String,
}

/// Validate and load each input item. Returns one `PreparedImage` per item in
/// the original order, suitable for passing as a single batched call into
/// fastembed `embed_image_bytes`.
pub fn prepare_images(
    items: &[ImageItem],
    settings: &ValidatedSettings,
) -> Result<Vec<PreparedImage>> {
    if items.is_empty() {
        return Err(MmEmbeddingError::invalid_argument(
            "items must not be empty",
        ));
    }

    // Built lazily on the first url item and reused across the rest, so a
    // request with many same-host urls reuses connections / the TLS config,
    // and a request with no url items pays no client/rustls setup cost.
    let mut http_client: Option<reqwest::blocking::Client> = None;

    let mut prepared = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let source = item
            .source
            .as_ref()
            .ok_or_else(|| MmEmbeddingError::invalid_argument("source is not set"))?;

        let (bytes, default_hint) = match source {
            ImageSource::Url(url) => {
                let client = match &http_client {
                    Some(c) => c,
                    None => http_client.insert(build_http_client()?),
                };
                let bytes = load_url(client, url, settings)?;
                (bytes, url.clone())
            }
            ImageSource::FilePath(path_str) => {
                let bytes = load_file_path(path_str, settings)?;
                (bytes, path_str.clone())
            }
            ImageSource::Data(data) => {
                if (data.len() as u64) > settings.image_size_limit_bytes {
                    return Err(MmEmbeddingError::invalid_argument(format!(
                        "data exceeds image_size_limit_bytes (limit={}, got={})",
                        settings.image_size_limit_bytes,
                        data.len()
                    )));
                }
                (data.clone(), format!("image[#{i}]"))
            }
        };

        preflight_decoded_size(&bytes, settings.max_decoded_image_bytes)?;

        let source_hint = item.label.clone().unwrap_or(default_hint);
        prepared.push(PreparedImage { bytes, source_hint });
    }

    Ok(prepared)
}

/// Read at most `limit` bytes from `reader`, erroring if the source exceeds
/// it. `take(limit + 1)` lets us detect an over-limit body without buffering
/// the whole oversized stream. `size_hint` (e.g. Content-Length / file size)
/// seeds the buffer capacity but is clamped to `limit` so a lying hint
/// cannot trigger a huge up-front allocation. `what` names the source for
/// the error message.
fn read_capped<R: Read>(
    reader: R,
    limit: u64,
    size_hint: Option<u64>,
    what: &str,
) -> Result<Vec<u8>> {
    let cap = size_hint.unwrap_or(0).min(limit);
    let mut buf = Vec::with_capacity(cap as usize);
    reader
        .take(limit + 1)
        .read_to_end(&mut buf)
        .map_err(|e| MmEmbeddingError::io(format!("{what} read failed: {e}")))?;
    if (buf.len() as u64) > limit {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "{what} exceeds image_size_limit_bytes (limit={limit})"
        )));
    }
    Ok(buf)
}

fn build_http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(URL_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(URL_MAX_REDIRECTS))
        .build()
        .map_err(|e| MmEmbeddingError::backend(format!("http client init failed: {e}")))
}

fn load_url(
    client: &reqwest::blocking::Client,
    url: &str,
    settings: &ValidatedSettings,
) -> Result<Vec<u8>> {
    // Reject anything that is not plain http/https before any network I/O,
    // so file://, data:, ftp:// etc. cannot reach the HTTP client.
    let scheme = url.split_once("://").map(|(s, _)| s.to_ascii_lowercase());
    if !matches!(scheme.as_deref(), Some("http") | Some("https")) {
        return Err(MmEmbeddingError::invalid_argument(
            "url must use the http or https scheme",
        ));
    }

    let limit = settings.image_size_limit_bytes;

    let resp = client
        .get(url)
        .send()
        .map_err(|e| MmEmbeddingError::io(format!("url fetch failed: {e}")))?;
    if !resp.status().is_success() {
        return Err(MmEmbeddingError::io(format!(
            "url fetch returned HTTP {}",
            resp.status().as_u16()
        )));
    }

    // Advisory early rejection: a lying or absent Content-Length is still
    // caught by read_capped's streaming check.
    let declared = resp.content_length();
    if let Some(len) = declared
        && len > limit
    {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "url body exceeds image_size_limit_bytes (limit={limit}, content_length={len})"
        )));
    }

    read_capped(resp, limit, declared, "url body")
}

fn load_file_path(path_str: &str, settings: &ValidatedSettings) -> Result<Vec<u8>> {
    if settings.allowed_image_dirs.is_empty() {
        return Err(MmEmbeddingError::invalid_argument(
            "file_path source not allowed (allowed_image_dirs is empty)",
        ));
    }

    let canonical = std::fs::canonicalize(path_str)
        .map_err(|_| MmEmbeddingError::io(format!("canonicalize failed: {path_str}")))?;

    if !is_within_allowed_dirs(&canonical, &settings.allowed_image_dirs) {
        return Err(MmEmbeddingError::invalid_argument(
            "file_path outside allowed_image_dirs",
        ));
    }

    let limit = settings.image_size_limit_bytes;
    let metadata = std::fs::metadata(&canonical)
        .map_err(|e| MmEmbeddingError::io(format!("metadata failed: {e}")))?;
    if metadata.len() > limit {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "file_path exceeds image_size_limit_bytes (limit={limit}, got={})",
            metadata.len()
        )));
    }

    let file =
        File::open(&canonical).map_err(|e| MmEmbeddingError::io(format!("open failed: {e}")))?;
    // read_capped re-checks the size after reading, so a file that grew
    // between the metadata check and the read is still rejected; the
    // in-memory buffer it returns is immune to later filesystem changes.
    read_capped(file, limit, Some(metadata.len()), "file_path")
}

/// Component-aware containment check. `canonical_target` must be canonical
/// already (callers obtained it via `std::fs::canonicalize`), and each entry
/// in `allowed_dirs` was canonicalized at settings load time, so we can
/// rely on `Path::starts_with` to enforce a *path component* prefix match
/// rather than a raw string prefix. This rules out the `/tmp/allow` vs
/// `/tmp/allow_evil` collision.
fn is_within_allowed_dirs(canonical_target: &Path, allowed_dirs: &[PathBuf]) -> bool {
    allowed_dirs
        .iter()
        .any(|dir| canonical_target.starts_with(dir))
}

fn preflight_decoded_size(bytes: &[u8], limit: DecodedImageBytesLimit) -> Result<()> {
    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| {
            MmEmbeddingError::invalid_argument("invalid image header (unrecognized format)")
        })?;

    let (w, h) = reader
        .into_dimensions()
        .map_err(|e| MmEmbeddingError::invalid_argument(format!("invalid image header: {e}")))?;

    let decoded_bytes = u64::from(w)
        .checked_mul(u64::from(h))
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| MmEmbeddingError::invalid_argument("decoded size overflow"))?;

    if let Some(cap) = limit.effective_bytes()
        && decoded_bytes > cap
    {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "decoded size exceeds max_decoded_image_bytes (cap={cap}, got={decoded_bytes})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_PNG: &[u8] = include_bytes!("../tests/fixtures/tiny_valid.png");
    const ZIP_BOMB_PNG: &[u8] = include_bytes!("../tests/fixtures/zip_bomb.png");

    #[test]
    fn small_image_passes_default_preflight() {
        preflight_decoded_size(TINY_PNG, DecodedImageBytesLimit::Default).unwrap();
    }

    #[test]
    fn unlimited_allows_zip_bomb() {
        preflight_decoded_size(ZIP_BOMB_PNG, DecodedImageBytesLimit::Unlimited).unwrap();
    }

    #[test]
    fn small_cap_rejects_zip_bomb() {
        let err = preflight_decoded_size(ZIP_BOMB_PNG, DecodedImageBytesLimit::Limited(1024))
            .unwrap_err();
        assert!(err.to_string().starts_with("InvalidArgument:"));
        assert!(err.to_string().contains("decoded size exceeds"));
    }

    #[test]
    fn invalid_header_rejected() {
        let err =
            preflight_decoded_size(b"NOTANIMAGE", DecodedImageBytesLimit::Default).unwrap_err();
        assert!(err.to_string().starts_with("InvalidArgument:"));
    }
}
