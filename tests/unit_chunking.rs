//! End-to-end checks for the chunking adapter. Uses the offline tokenizer
//! fixture so the suite stays in the default `cargo test` lane.

use mm_embedding_runner::chunking::chunk_text;
use mm_embedding_runner::settings::ResolvedChunkingConfig;
use mm_embedding_runner::tokenization::TokenizationProcessor;
use std::path::PathBuf;
use std::sync::Arc;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn tokenizer() -> Arc<TokenizationProcessor> {
    Arc::new(
        TokenizationProcessor::load_from_file(&PathBuf::from(FIXTURES).join("tokenizer.json"))
            .unwrap(),
    )
}

fn fixture_text(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(FIXTURES).join(name)).unwrap()
}

fn slice_chars(text: &str, char_start: usize, char_end: usize) -> String {
    // Use chars().skip/take instead of byte indexing because chunk positions
    // are recorded as char indices, not byte offsets.
    text.chars()
        .skip(char_start)
        .take(char_end.saturating_sub(char_start))
        .collect()
}

#[test]
fn short_input_returns_single_chunk_covering_full_text() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_ascii.txt");
    let cfg = ResolvedChunkingConfig {
        max_chunk_tokens: 2048,
        min_chunk_tokens: 0,
    };
    let chunks = chunk_text(&text, tk, cfg).unwrap();
    assert_eq!(chunks.len(), 1);
    let c = &chunks[0];
    assert_eq!(c.char_start, 0);
    assert_eq!(c.char_end, text.chars().count());
    assert_eq!(c.content, text);
}

#[test]
fn long_input_produces_multiple_chunks_covering_text() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_long.txt");
    // Pick a small max so chunking is forced even though the chunker is
    // free to coalesce paragraphs.
    let cfg = ResolvedChunkingConfig {
        max_chunk_tokens: 256,
        min_chunk_tokens: 0,
    };
    let chunks = chunk_text(&text, tk, cfg).unwrap();
    assert!(
        chunks.len() > 1,
        "expected multiple chunks for long text, got {}",
        chunks.len()
    );

    // Each chunk's content must equal the char-indexed slice of the
    // original text.
    for c in &chunks {
        let expected = slice_chars(&text, c.char_start, c.char_end);
        assert_eq!(
            c.content, expected,
            "chunk content does not match char slice [{}, {}]",
            c.char_start, c.char_end
        );
    }

    // Char ranges should be monotonically non-decreasing in start position.
    for pair in chunks.windows(2) {
        assert!(
            pair[0].char_start <= pair[1].char_start,
            "chunk char_start went backwards: {} -> {}",
            pair[0].char_start,
            pair[1].char_start,
        );
    }
}

#[test]
fn japanese_chunk_content_uses_char_slicing() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_ja.txt");
    let cfg = ResolvedChunkingConfig {
        max_chunk_tokens: 64,
        min_chunk_tokens: 0,
    };
    let chunks = chunk_text(&text, tk, cfg).unwrap();
    assert!(!chunks.is_empty());
    for c in &chunks {
        let expected = slice_chars(&text, c.char_start, c.char_end);
        assert_eq!(c.content, expected, "ja chunk must align on char indices");
    }
}

#[test]
fn emoji_chunk_content_uses_char_slicing() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_emoji.txt");
    let cfg = ResolvedChunkingConfig {
        max_chunk_tokens: 32,
        min_chunk_tokens: 0,
    };
    let chunks = chunk_text(&text, tk, cfg).unwrap();
    for c in &chunks {
        let expected = slice_chars(&text, c.char_start, c.char_end);
        assert_eq!(
            c.content, expected,
            "emoji chunk must align on char indices"
        );
    }
}

#[test]
fn min_chunk_tokens_below_max_is_accepted() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_long.txt");
    let cfg = ResolvedChunkingConfig {
        max_chunk_tokens: 256,
        min_chunk_tokens: 10,
    };
    let chunks = chunk_text(&text, tk, cfg).unwrap();
    assert!(!chunks.is_empty());
}
