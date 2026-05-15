//! Adapter for `command_utils::text::chunking::HierarchicalChunker`.
//!
//! Conventions for the produced chunks:
//! - `char_start` / `char_end` are char indices (Unicode scalar values),
//!   not byte offsets.
//! - `content` always equals
//!   `text.chars().skip(char_start).take(char_end - char_start).collect()`.

use std::sync::{Arc, Mutex};

use command_utils::text::chunking::{
    FallbackStrategy, HierarchicalChunker, HierarchicalChunkingConfig, TokenProvider,
};

use crate::error::{MmEmbeddingError, Result};
use crate::settings::ResolvedChunkingConfig;
use crate::tokenization::{TokenizationOutput, TokenizationProcessor};

#[derive(Debug, Clone)]
pub struct EmbeddingChunk {
    pub content: String,
    pub char_start: usize,
    pub char_end: usize,
    pub tokens: Vec<u32>,
}

/// Bridge between `TokenizationProcessor` and the chunker's `TokenProvider`.
///
/// The chunker calls `get_token_spans` / `tokenize` / `estimate_token_count`
/// against the *same* full text several times within one `chunk_efficiently`
/// run. Tokenizing the whole input on every call would re-run a O(N) native
/// encode pass each time, so the last `(text, output)` pair is memoized.
/// command-utils' own `TokenizationCache` only covers `tokenize` /
/// `estimate_token_count`, not the span path, which is why the memo lives
/// here. `Mutex` because `TokenProvider: Send + Sync`.
pub struct TokenizationAdapter {
    tokenizer: Arc<TokenizationProcessor>,
    last: Mutex<Option<(String, Arc<TokenizationOutput>)>>,
}

impl TokenizationAdapter {
    #[must_use]
    pub fn new(tokenizer: Arc<TokenizationProcessor>) -> Self {
        Self {
            tokenizer,
            last: Mutex::new(None),
        }
    }

    fn tokenized(&self, text: &str) -> Result<Arc<TokenizationOutput>> {
        let mut guard = self.last.lock().expect("tokenization memo poisoned");
        if let Some((cached_text, cached)) = guard.as_ref()
            && cached_text == text
        {
            return Ok(cached.clone());
        }
        let out = Arc::new(self.tokenizer.tokenize_with_offsets(text)?);
        *guard = Some((text.to_string(), out.clone()));
        Ok(out)
    }
}

impl TokenProvider for TokenizationAdapter {
    type Error = MmEmbeddingError;

    fn tokenize(&self, text: &str) -> std::result::Result<Vec<u32>, Self::Error> {
        Ok(self.tokenized(text)?.tokens.iter().map(|t| t.id).collect())
    }

    fn estimate_token_count(&self, text: &str) -> std::result::Result<usize, Self::Error> {
        Ok(self.tokenized(text)?.tokens.len())
    }

    fn token_to_char(
        &self,
        text: &str,
        token_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        Ok(self
            .tokenized(text)?
            .tokens
            .get(token_pos)
            .map(|t| t.char_start))
    }

    fn char_to_token(
        &self,
        text: &str,
        char_pos: usize,
    ) -> std::result::Result<Option<usize>, Self::Error> {
        let out = self.tokenized(text)?;
        for (idx, tok) in out.tokens.iter().enumerate() {
            if char_pos >= tok.char_start && char_pos < tok.char_end {
                return Ok(Some(idx));
            }
        }
        // char_pos past the last token clamps to the final index rather than
        // erroring, matching embedding-llm's chunker contract.
        Ok(out
            .tokens
            .last()
            .map(|_| out.tokens.len().saturating_sub(1)))
    }

    fn get_token_spans(
        &self,
        text: &str,
    ) -> std::result::Result<Option<Vec<(usize, usize)>>, Self::Error> {
        Ok(Some(
            self.tokenized(text)?
                .tokens
                .iter()
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
    // The adapter memoizes its last tokenize result. Doing the fast-path
    // length check through it (instead of TokenizationProcessor::token_count)
    // means the full-text encode is shared with the chunker when we fall
    // through to the hierarchical path, avoiding a second O(N) encode.
    let adapter = TokenizationAdapter::new(tokenizer);

    let total_tokens = adapter.estimate_token_count(text)?;
    if total_tokens <= config.max_chunk_tokens as usize {
        // Short input: one chunk spanning the whole text. tokens left empty
        // because the embedding backend re-tokenizes internally; chunk-side
        // token ids are only consumed by length-inspecting callers.
        return Ok(vec![EmbeddingChunk {
            content: text.to_string(),
            char_start: 0,
            char_end: text.chars().count(),
            tokens: Vec::new(),
        }]);
    }

    let hc_config = HierarchicalChunkingConfig {
        max_chunk_tokens: config.max_chunk_tokens as usize,
        min_chunk_tokens: config.min_chunk_tokens as usize,
        ..HierarchicalChunkingConfig::for_embedding(config.max_chunk_tokens as usize)
    };
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
    // Callers require:
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
