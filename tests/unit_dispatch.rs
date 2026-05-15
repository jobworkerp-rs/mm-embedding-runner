use jobworkerp_client::plugins::MultiMethodPluginRunner;
use mm_embedding_runner::protobuf::mm_embedding::{
    EmbedImageArgs, EmbedTextArgs, MultimodalEmbeddingRunnerSettings,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin};
use prost::Message;
use std::collections::HashMap;

#[test]
fn not_loaded_for_embed_text_before_load() {
    let mut plugin = MmEmbeddingPlugin::new();
    let args = EmbedTextArgs {
        text: "hello".to_string(),
    }
    .encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_TEXT));
    let err = result.expect_err("expected NotLoaded");
    assert!(err.to_string().starts_with("NotLoaded:"), "got: {err}");
}

#[test]
fn not_loaded_for_embed_image_before_load() {
    let mut plugin = MmEmbeddingPlugin::new();
    let args = EmbedImageArgs::default().encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let err = result.expect_err("expected NotLoaded");
    assert!(err.to_string().starts_with("NotLoaded:"), "got: {err}");
}

#[test]
fn malformed_args_yield_invalid_argument_even_before_load() {
    // Decode errors should be reported as InvalidArgument regardless of
    // whether the plugin has been loaded — they originate from the caller.
    let mut plugin = MmEmbeddingPlugin::new();
    let bad = vec![0xff]; // varint continuation with no follow-up byte
    let (result, _) = plugin.run(bad.clone(), HashMap::new(), Some(METHOD_EMBED_TEXT));
    let err = result.expect_err("expected InvalidArgument");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );

    let (result_img, _) = plugin.run(bad, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let err_img = result_img.expect_err("expected InvalidArgument");
    assert!(
        err_img.to_string().starts_with("InvalidArgument:"),
        "got: {err_img}"
    );
}

// load() now downloads the actual Qwen3-VL-Embedding-2B weights from the
// HuggingFace Hub, so the success path requires
// network + several GiB of disk. The integration test
// `tests/integration_backend_load.rs::full_load_succeeds` covers it.

#[test]
fn load_with_malformed_bytes_returns_invalid_argument() {
    let mut plugin = MmEmbeddingPlugin::new();
    let bad = vec![0xff];
    let err = plugin.load(bad).expect_err("expected decode error");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn load_with_empty_model_id_is_rejected() {
    let mut plugin = MmEmbeddingPlugin::new();
    let s = MultimodalEmbeddingRunnerSettings::default(); // model_id empty
    let err = plugin
        .load(s.encode_to_vec())
        .expect_err("expected validation error");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}
