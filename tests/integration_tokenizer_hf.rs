//! Verifies that `TokenizationProcessor::load_from_hf` can fetch a real
//! tokenizer.json from the HuggingFace Hub. Marked `#[ignore]` so it stays
//! out of the offline default test lane.

use mm_embedding_runner::tokenization::TokenizationProcessor;

#[test]
#[ignore]
fn load_qwen3_vl_embedding_tokenizer_from_hub() {
    let tk = TokenizationProcessor::load_from_hf("Qwen/Qwen3-VL-Embedding-2B")
        .expect("tokenizer.json should download from the HuggingFace Hub");
    // Smoke test: tokenize a short ASCII string and ensure at least one
    // token comes back with a non-zero char span.
    let out = tk.tokenize_with_offsets("hello world").unwrap();
    assert!(!out.tokens.is_empty());
    assert!(out.tokens.iter().any(|t| t.char_end > t.char_start));
}

#[test]
#[ignore]
fn hub_tokenizer_matches_fixture_token_count() {
    use std::path::PathBuf;
    let fix_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tokenizer.json");
    let from_file = TokenizationProcessor::load_from_file(&fix_path).unwrap();
    let from_hub = TokenizationProcessor::load_from_hf("Qwen/Qwen3-VL-Embedding-2B").unwrap();

    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sample_text_ja.txt"),
    )
    .unwrap();
    assert_eq!(
        from_file.token_count(&text).unwrap(),
        from_hub.token_count(&text).unwrap(),
        "fixture and hub-fetched tokenizers should agree",
    );
}
