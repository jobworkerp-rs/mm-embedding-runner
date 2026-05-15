//! Tokenizer-driven span checks against the on-disk Qwen tokenizer.json
//! fixture. These tests stay offline (load_from_file, no HF Hub fetch) so
//! they run under the default `cargo test` lane.

use mm_embedding_runner::tokenization::TokenizationProcessor;
use std::path::PathBuf;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn tokenizer() -> TokenizationProcessor {
    TokenizationProcessor::load_from_file(&PathBuf::from(FIXTURES).join("tokenizer.json"))
        .expect("tokenizer.json fixture should load")
}

fn fixture_text(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(FIXTURES).join(name)).unwrap()
}

#[test]
fn ascii_text_tokenizes_with_consistent_spans() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_ascii.txt");
    let out = tk.tokenize_with_offsets(&text).unwrap();
    assert!(!out.tokens.is_empty(), "ASCII fixture should tokenize");
    // Last token's char_end must not exceed the char count of the input.
    let char_count = text.chars().count();
    for tok in &out.tokens {
        assert!(
            tok.char_end <= char_count,
            "token span {}..{} exceeds char_count {char_count}",
            tok.char_start,
            tok.char_end
        );
        assert!(tok.char_start <= tok.char_end);
    }
}

#[test]
fn japanese_text_byte_offsets_are_converted_to_char_indices() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_ja.txt");
    let out = tk.tokenize_with_offsets(&text).unwrap();
    let char_count = text.chars().count();
    let byte_count = text.len();
    assert!(char_count < byte_count, "ja fixture should be multi-byte");
    for tok in &out.tokens {
        assert!(
            tok.char_end <= char_count,
            "token span {}..{} exceeds char_count {char_count} (byte_count={byte_count})",
            tok.char_start,
            tok.char_end
        );
        // If the conversion had treated byte offsets as char indices we
        // would see spans > char_count for tokens late in the input.
    }
}

#[test]
fn emoji_text_keeps_spans_within_char_bounds() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_emoji.txt");
    let out = tk.tokenize_with_offsets(&text).unwrap();
    let char_count = text.chars().count();
    for tok in &out.tokens {
        assert!(tok.char_end <= char_count);
    }
}

#[test]
fn token_count_matches_tokenize_with_offsets_length() {
    let tk = tokenizer();
    let text = fixture_text("sample_text_ascii.txt");
    let count = tk.token_count(&text).unwrap();
    let out = tk.tokenize_with_offsets(&text).unwrap();
    assert_eq!(count, out.tokens.len());
}
