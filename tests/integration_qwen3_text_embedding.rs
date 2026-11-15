//! End-to-end check of the text-only architecture path (Qwen3-Embedding-*
//! family, `architectures[0] == "Qwen3ForCausalLM"`). A text-only model is
//! served through the same plugin contract as Qwen3-VL: `embed_text` works
//! with the usual chunking and position metadata, the model reports
//! `supports_vision == false`, and `embed_image` is rejected with
//! `InvalidArgument:` since there is no image path to share a vector space
//! with.

mod common;

use common::{assert_unit_norm, load_plugin_with_model, run_method};
use mm_embedding_runner::protobuf::mm_embedding::{
    EmbedImageArgs, EmbedTextArgs, ImageItem, MmEmbeddingResult, image_item::Source as ImageSource,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin};
use prost::Message;
use std::sync::{Mutex, MutexGuard, OnceLock};

const TEXT_ONLY_MODEL_ID: &str = "Qwen/Qwen3-Embedding-0.6B";
const TEXT_ONLY_EXPECTED_DIM: u32 = 1024;

static LOADED_PLUGIN: OnceLock<Mutex<MmEmbeddingPlugin>> = OnceLock::new();

fn shared_plugin() -> MutexGuard<'static, MmEmbeddingPlugin> {
    LOADED_PLUGIN
        .get_or_init(|| Mutex::new(load_plugin_with_model(TEXT_ONLY_MODEL_ID, vec![])))
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
fn short_text_returns_single_embedding_at_model_dim() {
    let mut plugin = shared_plugin();
    let text = "the quick brown fox jumps over the lazy dog";
    let out = run_embed_text(&mut plugin, text);

    assert_eq!(out.embeddings.len(), 1);
    let emb = &out.embeddings[0];
    assert_eq!(emb.begin_position, 0);
    assert_eq!(emb.end_position as usize, text.chars().count());
    assert_eq!(emb.content, text);
    // The dimension must come from the actual checkpoint's hidden_size, not
    // a hard-coded VL constant.
    assert_eq!(emb.values.len(), TEXT_ONLY_EXPECTED_DIM as usize);
    assert_unit_norm(&emb.values);
}

#[test]
#[ignore]
fn model_info_reports_text_only_capabilities() {
    let mut plugin = shared_plugin();
    let out = run_embed_text(&mut plugin, "hello world");

    let info = out.model_info.expect("model_info should be present");
    assert_eq!(info.model_name, TEXT_ONLY_MODEL_ID);
    assert_eq!(info.embedding_dimension, TEXT_ONLY_EXPECTED_DIM);
    assert!(
        !info.supports_vision,
        "text-only model must report supports_vision=false"
    );
    assert!(!info.supports_audio);
}

#[test]
#[ignore]
fn long_text_chunking_still_applies_for_text_only() {
    let mut plugin = shared_plugin();
    let chunk_seed = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ";
    let text: String = chunk_seed.repeat(200);
    let out = run_embed_text(&mut plugin, &text);

    assert!(
        out.embeddings.len() > 1,
        "expected multiple chunks for a long text input"
    );

    let char_count = text.chars().count();
    for emb in &out.embeddings {
        assert!((emb.end_position as usize) <= char_count);
        assert!(emb.begin_position <= emb.end_position);
        assert_eq!(emb.values.len(), TEXT_ONLY_EXPECTED_DIM as usize);
        assert_unit_norm(&emb.values);
        let expected_content: String = text
            .chars()
            .skip(emb.begin_position as usize)
            .take((emb.end_position - emb.begin_position) as usize)
            .collect();
        assert_eq!(emb.content, expected_content);
    }
}

#[test]
#[ignore]
fn embed_image_is_rejected_with_invalid_argument() {
    let mut plugin = shared_plugin();
    // The runtime fast-paths the rejection before any image decoding, so
    // the payload bytes are never inspected — empty data is enough.
    let args = EmbedImageArgs {
        items: vec![ImageItem {
            source: Some(ImageSource::Data(Vec::new())),
            label: None,
        }],
    }
    .encode_to_vec();
    let err = run_method(&mut plugin, METHOD_EMBED_IMAGE, args)
        .expect_err("text-only model must reject embed_image");
    assert!(
        err.starts_with("InvalidArgument:"),
        "expected InvalidArgument: prefix, got: {err}"
    );
    assert!(
        err.to_lowercase().contains("text-only"),
        "rejection message should mention text-only, got: {err}"
    );
}
