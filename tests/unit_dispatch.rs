use jobworkerp_plugin_abi::v2::PluginV2;
use mm_embedding_runner::protobuf::mm_embedding::{
    EmbedImageArgs, EmbedTextArgs, MultimodalEmbeddingRunnerSettings,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin};
use prost::Message;

mod common;
use common::{block_on, run_method};

#[test]
fn not_loaded_for_embed_text_before_load() {
    let mut plugin = MmEmbeddingPlugin::new();
    let args = EmbedTextArgs {
        text: "hello".to_string(),
    }
    .encode_to_vec();
    let err = run_method(&mut plugin, METHOD_EMBED_TEXT, args).expect_err("expected NotLoaded");
    assert!(err.starts_with("NotLoaded:"), "got: {err}");
}

#[test]
fn not_loaded_for_embed_image_before_load() {
    let mut plugin = MmEmbeddingPlugin::new();
    let args = EmbedImageArgs::default().encode_to_vec();
    let err = run_method(&mut plugin, METHOD_EMBED_IMAGE, args).expect_err("expected NotLoaded");
    assert!(err.starts_with("NotLoaded:"), "got: {err}");
}

#[test]
fn malformed_args_yield_invalid_argument_even_before_load() {
    // Decode errors should be reported as InvalidArgument regardless of
    // whether the plugin has been loaded — they originate from the caller.
    // run() decodes before the load-state check to preserve this.
    let mut plugin = MmEmbeddingPlugin::new();
    let bad = vec![0xff]; // varint continuation with no follow-up byte
    let err = run_method(&mut plugin, METHOD_EMBED_TEXT, bad.clone())
        .expect_err("expected InvalidArgument");
    assert!(err.starts_with("InvalidArgument:"), "got: {err}");

    let err_img =
        run_method(&mut plugin, METHOD_EMBED_IMAGE, bad).expect_err("expected InvalidArgument");
    assert!(err_img.starts_with("InvalidArgument:"), "got: {err_img}");
}

// load() now downloads the actual Qwen3-VL-Embedding-2B weights from the
// HuggingFace Hub, so the success path requires
// network + several GiB of disk. The integration test
// `tests/integration_backend_load.rs::full_load_succeeds` covers it.

#[test]
fn load_with_malformed_bytes_returns_invalid_argument() {
    let mut plugin = MmEmbeddingPlugin::new();
    let bad = vec![0xff];
    let err = block_on(plugin.load(bad)).expect_err("expected decode error");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn load_with_empty_model_id_is_rejected() {
    let mut plugin = MmEmbeddingPlugin::new();
    let s = MultimodalEmbeddingRunnerSettings::default(); // model_id empty
    let err = block_on(plugin.load(s.encode_to_vec())).expect_err("expected validation error");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}
