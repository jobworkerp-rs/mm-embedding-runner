//! Image input validation and decode-size preflight.
//!
//! Responsibilities:
//! - Reject empty `items`, missing `source`, and `url` sources up front.
//! - For `file_path`: enforce `allowed_image_dirs` membership via canonical
//!   `Path::starts_with` (component-aware), then load bytes ourselves to
//!   eliminate the TOCTOU window before the embedding backend reopens the
//!   file.
//! - For both `file_path` and `data`: enforce compressed-size cap and run a
//!   header-only decode-size preflight via `image::ImageReader`.
//!
//! Returns the raw bytes for each input in submission order so the caller
//! can pass them as a single `&[&[u8]]` slice to fastembed's
//! `embed_image_bytes`.

use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use image::ImageReader;

use crate::error::{MmEmbeddingError, Result};
use crate::protobuf::mm_embedding::{ImageItem, image_item::Source as ImageSource};
use crate::settings::{DecodedImageBytesLimit, ValidatedSettings};

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

    let mut prepared = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let source = item
            .source
            .as_ref()
            .ok_or_else(|| MmEmbeddingError::invalid_argument("source is not set"))?;

        let (bytes, default_hint) = match source {
            ImageSource::Url(_) => {
                return Err(MmEmbeddingError::invalid_argument(
                    "url source is not supported in v0.1",
                ));
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

    let metadata = std::fs::metadata(&canonical)
        .map_err(|e| MmEmbeddingError::io(format!("metadata failed: {e}")))?;
    if metadata.len() > settings.image_size_limit_bytes {
        return Err(MmEmbeddingError::invalid_argument(format!(
            "file_path exceeds image_size_limit_bytes (limit={}, got={})",
            settings.image_size_limit_bytes,
            metadata.len()
        )));
    }

    let mut file =
        File::open(&canonical).map_err(|e| MmEmbeddingError::io(format!("open failed: {e}")))?;
    let mut buf = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut buf)
        .map_err(|e| MmEmbeddingError::io(format!("read failed: {e}")))?;

    // Guard against the file growing between the metadata check and the
    // read. Subsequent embed-side reads are immune to growth because we
    // pass the in-memory buffer.
    if (buf.len() as u64) > settings.image_size_limit_bytes {
        return Err(MmEmbeddingError::invalid_argument(
            "file_path size grew during read",
        ));
    }

    Ok(buf)
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
