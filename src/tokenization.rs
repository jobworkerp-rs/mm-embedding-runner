//! Tokenization driver used for chunking accounting.
//!
//! fastembed-rs does not expose its internal tokenizer, so we maintain our
//! own `tokenizers::Tokenizer` and use it strictly for:
//! 1. Counting tokens so the chunker can decide whether splitting is needed.
//! 2. Recovering `(char_start, char_end)` spans per token so chunk
//!    boundaries can be reported as character positions in the result.
//!
//! The byte→char conversion lives in `offset::CharByteMap` so the unit tests
//! can exercise it independently without spinning up a real tokenizer.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::error::{MmEmbeddingError, Result};
use crate::offset::CharByteMap;

/// Token id paired with its `(char_start, char_end)` span in the source text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenWithSpan {
    pub id: u32,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizationOutput {
    pub tokens: Vec<TokenWithSpan>,
}

pub struct TokenizationProcessor {
    tokenizer: Tokenizer,
}

impl std::fmt::Debug for TokenizationProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizationProcessor")
            .finish_non_exhaustive()
    }
}

impl TokenizationProcessor {
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| {
            MmEmbeddingError::backend(format!(
                "failed to load tokenizer from {}: {e}",
                path.display()
            ))
        })?;
        Ok(Self { tokenizer })
    }

    /// Fetch `tokenizer.json` for the given HuggingFace repository ID and
    /// load it. fastembed downloads its own copy internally as part of model
    /// initialization, but it does not expose the resulting `Tokenizer`, so
    /// the runner maintains its own copy for chunking accounting.
    ///
    /// Cache directory is governed by the standard `HF_HOME` / `HF_HUB_CACHE`
    /// environment variables. The runner intentionally exposes no cache_dir
    /// setting because the underlying `Qwen3VLEmbedding::from_hf` API does
    /// not accept one.
    pub fn load_from_hf(repo_id: &str) -> Result<Self> {
        let api = hf_hub::api::sync::ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .map_err(|e| MmEmbeddingError::backend(format!("hf-hub API init failed: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let tokenizer_file = repo.get("tokenizer.json").map_err(|e| {
            MmEmbeddingError::backend(format!(
                "failed to download tokenizer.json for {repo_id}: {e}"
            ))
        })?;
        Self::load_from_file(&tokenizer_file)
    }

    /// Tokenize `text` and produce per-token spans in **char index** units.
    ///
    /// Implementation notes:
    /// - `tokenizers::Encoding::get_offsets()` reports byte offsets. We pair
    ///   them with a `CharByteMap` over the same text and reject any token
    ///   whose offset falls inside a multi-byte sequence.
    /// - Special tokens (BOS/EOS) typically yield `(0, 0)` from the
    ///   tokenizer. We propagate those spans unchanged so callers can detect
    ///   them by `char_start == char_end` if needed.
    pub fn tokenize_with_offsets(&self, text: &str) -> Result<TokenizationOutput> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| MmEmbeddingError::backend(format!("tokenization failed: {e}")))?;

        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();
        if ids.len() != offsets.len() {
            return Err(MmEmbeddingError::backend(format!(
                "tokenizer returned mismatched ids ({}) and offsets ({}) lengths",
                ids.len(),
                offsets.len()
            )));
        }

        let map = CharByteMap::new(text);
        let mut tokens = Vec::with_capacity(ids.len());
        for (idx, (&id, &(byte_start, byte_end))) in ids.iter().zip(offsets.iter()).enumerate() {
            // (0, 0) is the conventional signal for synthetic / special
            // tokens that have no source-text span. Pass through unchanged.
            let (char_start, char_end) = if byte_start == 0 && byte_end == 0 {
                (0, 0)
            } else {
                map.byte_range_to_char_range(byte_start, byte_end)
                    .map_err(|e| {
                        MmEmbeddingError::backend(format!("token #{idx} (id={id}): {e}"))
                    })?
            };
            tokens.push(TokenWithSpan {
                id,
                char_start,
                char_end,
            });
        }
        Ok(TokenizationOutput { tokens })
    }

    /// Count tokens only, skipping span construction for callers that just
    /// need a length estimate (e.g. the chunker's quick-path check).
    pub fn token_count(&self, text: &str) -> Result<usize> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| MmEmbeddingError::backend(format!("tokenization failed: {e}")))?;
        Ok(encoding.get_ids().len())
    }
}
