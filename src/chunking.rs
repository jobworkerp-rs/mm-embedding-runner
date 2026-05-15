//! Adapter for `command_utils::text::chunking::HierarchicalChunker`.
//!
//! We provide a `TokenProvider` impl that wraps `TokenizationProcessor`, then
//! map the resulting `HierarchicalChunk` values into a simpler
//! `EmbeddingChunk` representation that downstream code (P4-3) feeds into
//! `Qwen3VLEmbedding::embed_texts` in batched form.
//!
//! The chunker itself enforces:
//! - Paragraph- and sentence-aware splitting prioritized over forced splits.
//! - `max_chunk_tokens` and `min_chunk_tokens` from the resolved settings.
//!
//! Conventions (specification.md §3.6.1):
//! - `char_start` / `char_end` are **char indices** (Unicode scalar values).
//! - `content` always equals `text.chars().skip(char_start).take(char_end - char_start).collect()`.

use std::sync::Arc;

use command_utils::text::chunking::{
    FallbackStrategy, HierarchicalChunker, HierarchicalChunkingConfig, TokenProvider,
};

use crate::error::{MmEmbeddingError, Result};
use crate::settings::ResolvedChunkingConfig;
use crate::tokenization::TokenizationProcessor;

#[derive(Debug, Clone)]
pub struct EmbeddingChunk {
    pub content: String,
    pub char_start: usize,
    pub char_end: usize,
    pub tokens: Vec<u32>,
}

/// Bridge between `TokenizationProcessor` (our owned tokenizer) and the
/// chunker's expected `TokenProvider` interface. `Arc` so multiple chunker
/// instances over the same tokenizer (e.g. cross-test reuse) share state
/// without cloning the underlying tokenizers::Tokenizer.
pub struct TokenizationAdapter {
    tokenizer: Arc<TokenizationProcessor>,
}

impl TokenizationAdapter {
    #[must_use]
    pub fn new(tokenizer: Arc<TokenizationProcessor>) -> Self {
        Self { tokenizer }
    }
}

impl TokenProvider for TokenizationAdapter {
    type Error = MmEmbeddingError;

    fn tokenize(&self, text: &str) -> std::result::Result<Vec<u32>, Self::Error> {
        let out = self.tokenizer.tokenize_with_offsets(text)?;
        Ok(out.tokens.into_iter().map(|t| t.id).collect())
    }

    fn estimate_token_count(&self, text: &str) -> std::result::Result<usize, Self::Error> {
        self.tokenizer.token_count(text)
    }

    fn token_to_char(
        &self,
        text: &str,
        token_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        let out = self.tokenizer.tokenize_with_offsets(text)?;
        Ok(out.tokens.get(token_pos).map(|t| t.char_start))
    }

    fn char_to_token(
        &self,
        text: &str,
        char_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        let out = self.tokenizer.tokenize_with_offsets(text)?;
        for (idx, tok) in out.tokens.iter().enumerate() {
            if char_pos >= tok.char_start && char_pos < tok.char_end {
                return Ok(Some(idx));
            }
        }
        // char_pos beyond the last token: treat as "past end of input" rather
        // than an error, mirroring embedding-llm behavior.
        Ok(out
            .tokens
            .last()
            .map(|_| out.tokens.len().saturating_sub(1)))
    }

    fn get_token_spans(
        &self,
        text: &str,
    ) -> std::result::Result<Option<Vec<(usize, usize)>>, Self::Error> {
        let out = self.tokenizer.tokenize_with_offsets(text)?;
        Ok(Some(
            out.tokens
                .into_iter()
                .map(|t| (t.char_start, t.char_end))
                .collect(),
        ))
    }
}

pub fn chunk_text(
    text: &str,
    tokenizer: Arc<TokenizationProcessor>,
    config: ResolvedChunkingConfig,
) -> Result<Vec<EmbeddingChunk>> {
    // Single-chunk fast path: if the input is short enough to fit in one
    // chunk, skip the chunker entirely. This guarantees the spec invariant
    // for `embed_text` of short inputs (begin=0, end=char_count).
    let total_tokens = tokenizer.token_count(text)?;
    if total_tokens <= config.max_chunk_tokens as usize {
        return Ok(vec![EmbeddingChunk {
            content: text.to_string(),
            char_start: 0,
            char_end: text.chars().count(),
            // Defer the actual token id fetch since the embedding backend
            // will re-tokenize internally; chunk-side tokens are only used
            // by callers that want to inspect chunk lengths.
            tokens: Vec::new(),
        }]);
    }

    let hc_config = HierarchicalChunkingConfig {
        max_chunk_tokens: config.max_chunk_tokens as usize,
        min_chunk_tokens: config.min_chunk_tokens as usize,
        ..HierarchicalChunkingConfig::for_embedding(config.max_chunk_tokens as usize)
    };
    let adapter = TokenizationAdapter::new(tokenizer.clone());
    let mut chunker = HierarchicalChunker::new(
        hc_config,
        adapter,
        Some(FallbackStrategy::CharacterEstimation),
    )
    .map_err(|e| MmEmbeddingError::backend(format!("chunker init failed: {e}")))?;
    let raw_chunks = chunker
        .chunk_efficiently(text)
        .map_err(|e| MmEmbeddingError::backend(format!("chunking failed: {e}")))?;

    // HierarchicalChunker reports (char_start, char_end) ranges, but its
    // `content` field can carry internal normalization that diverges from
    // the original text slice (observed: in the embedding flow the chunker
    // may emit `content` that is shorter than `char_end - char_start`).
    //
    // The runner's contract (specification.md §3.6.1) requires
    //   content == text.chars().skip(char_start).take(char_end-char_start)
    //
    // So we drop the chunker's `content` field and rebuild it from the
    // input text via char slicing. `char_start` / `char_end` and `tokens`
    // are kept as-is because those are what the splitting decisions are
    // anchored on.
    let total_chars = text.chars().count();
    let mut out = Vec::with_capacity(raw_chunks.len());
    for c in raw_chunks {
        let char_start = c.char_start.min(total_chars);
        let char_end = c.char_end.min(total_chars).max(char_start);
        let content: String = text
            .chars()
            .skip(char_start)
            .take(char_end - char_start)
            .collect();
        out.push(EmbeddingChunk {
            content,
            char_start,
            char_end,
            tokens: c.tokens,
        });
    }
    Ok(out)
}
