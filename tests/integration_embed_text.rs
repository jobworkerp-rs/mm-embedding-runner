//! End-to-end check of the `embed_text` method path. Loads the real model
//! (via the same hf-hub cache as the backend integration tests) and verifies
//! the result contract: char-indexed positions, L2-normalized vectors, and
//! sane chunking behavior for long inputs.

mod common;

use common::{FIXTURES, assert_unit_norm, load_test_plugin, run_method};
use mm_embedding_runner::protobuf::mm_embedding::{EmbedTextArgs, MmEmbeddingResult};
use mm_embedding_runner::{METHOD_EMBED_TEXT, MmEmbeddingPlugin};
use prost::Message;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

static LOADED_PLUGIN: OnceLock<Mutex<MmEmbeddingPlugin>> = OnceLock::new();

// One loaded plugin per `cargo test` invocation so the multi-gig model
// init runs once.
fn shared_plugin() -> MutexGuard<'static, MmEmbeddingPlugin> {
    LOADED_PLUGIN
        .get_or_init(|| Mutex::new(load_test_plugin(vec![])))
        .lock()
        .unwrap()
}

fn run_embed_text(plugin: &mut MmEmbeddingPlugin, text: &str) -> MmEmbeddingResult {
    let args = EmbedTextArgs {
        text: text.to_string(),
    }
    .encode_to_vec();
    let bytes = run_method(plugin, METHOD_EMBED_TEXT, args).expect("embed_text should succeed");
    MmEmbeddingResult::decode(&bytes[..]).unwrap()
}

#[test]
#[ignore]
fn short_text_returns_single_embedding_covering_full_input() {
    let mut plugin = shared_plugin();
    let text = "a cat";
    let out = run_embed_text(&mut plugin, text);
    assert_eq!(out.embeddings.len(), 1);
    let emb = &out.embeddings[0];
    assert_eq!(emb.begin_position, 0);
    assert_eq!(emb.end_position as usize, text.chars().count());
    assert_eq!(emb.content, text);
    assert_eq!(emb.values.len(), 2048);
    assert_unit_norm(&emb.values);
    // model_info populated
    let info = out.model_info.expect("model_info should be present");
    assert_eq!(info.model_name, "Qwen/Qwen3-VL-Embedding-2B");
    assert_eq!(info.embedding_dimension, 2048);
    assert!(info.supports_vision);
    assert!(!info.supports_audio);
}

#[test]
#[ignore]
fn empty_text_is_rejected() {
    let mut plugin = shared_plugin();
    let args = EmbedTextArgs {
        text: "   ".to_string(),
    }
    .encode_to_vec();
    let err =
        run_method(&mut plugin, METHOD_EMBED_TEXT, args).expect_err("expected InvalidArgument");
    assert!(err.starts_with("InvalidArgument:"), "got: {err}");
}

#[test]
#[ignore]
fn long_text_produces_multiple_chunks() {
    let mut plugin = shared_plugin();
    let text = std::fs::read_to_string(PathBuf::from(FIXTURES).join("sample_text_long.txt"))
        .expect("long fixture should exist");
    let out = run_embed_text(&mut plugin, &text);
    assert!(out.embeddings.len() > 1, "expected multiple chunks");

    let char_count = text.chars().count();
    for emb in &out.embeddings {
        assert!((emb.end_position as usize) <= char_count);
        assert!(emb.begin_position <= emb.end_position);
        assert_eq!(emb.values.len(), 2048);
        assert_unit_norm(&emb.values);
        let expected_content: String = text
            .chars()
            .skip(emb.begin_position as usize)
            .take((emb.end_position - emb.begin_position) as usize)
            .collect();
        assert_eq!(emb.content, expected_content);
    }
}
