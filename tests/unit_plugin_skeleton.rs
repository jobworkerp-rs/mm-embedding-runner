use jobworkerp_client::jobworkerp::data::MethodSchema;
use jobworkerp_plugin_abi::v2::PluginV2;
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin, RUNNER_NAME};
use prost::Message;
use std::collections::HashMap;

mod common;
use common::{block_on, run_method};

#[test]
fn name_matches_constant() {
    let plugin = MmEmbeddingPlugin::new();
    assert_eq!(plugin.name(), RUNNER_NAME);
}

#[test]
fn description_is_non_empty() {
    let plugin = MmEmbeddingPlugin::new();
    assert!(!plugin.description().is_empty());
}

#[test]
fn run_without_method_yields_invalid_argument() {
    let mut plugin = MmEmbeddingPlugin::new();
    let (result, _meta) = block_on(plugin.run(vec![], HashMap::new(), None));
    let err = result.expect_err("expected InvalidArgument");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn run_with_unknown_method_yields_invalid_argument() {
    let mut plugin = MmEmbeddingPlugin::new();
    let (result, _meta) =
        block_on(plugin.run(vec![], HashMap::new(), Some("frobnicate".to_string())));
    let err = result.expect_err("expected InvalidArgument");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn run_without_load_yields_not_loaded() {
    let mut plugin = MmEmbeddingPlugin::new();
    let err = run_method(&mut plugin, METHOD_EMBED_TEXT, vec![]).expect_err("expected NotLoaded");
    assert!(err.starts_with("NotLoaded:"), "got: {err}");
}

#[test]
fn method_proto_map_has_two_methods() {
    let plugin = MmEmbeddingPlugin::new();
    let map = plugin.method_proto_map();
    assert_eq!(map.len(), 2);
    assert!(map.contains_key(METHOD_EMBED_TEXT));
    assert!(map.contains_key(METHOD_EMBED_IMAGE));
}

#[test]
fn method_proto_map_marks_non_streaming() {
    let plugin = MmEmbeddingPlugin::new();
    let map = plugin.method_proto_map();
    let non_streaming =
        jobworkerp_client::jobworkerp::data::StreamingOutputType::NonStreaming as i32;
    for (name, encoded) in &map {
        // PluginV2 exchanges schemas as protobuf-encoded bytes; decode back
        // into the host type to assert on field values.
        let schema = MethodSchema::decode(&encoded[..])
            .unwrap_or_else(|e| panic!("method {name}: schema decode: {e}"));
        assert_eq!(
            schema.output_type, non_streaming,
            "method {name} should be non-streaming",
        );
        assert!(
            !schema.require_client_stream,
            "method {name} should not require client streaming",
        );
        assert!(
            !schema.args_proto.is_empty(),
            "method {name} should have a non-empty args_proto",
        );
        assert!(
            !schema.result_proto.is_empty(),
            "method {name} should have a non-empty result_proto",
        );
    }
}

#[test]
fn settings_schema_is_not_empty_object() {
    let plugin = MmEmbeddingPlugin::new();
    let schema_str = plugin.settings_schema();
    let value: serde_json::Value =
        serde_json::from_str(&schema_str).expect("settings_schema must be valid JSON");
    let properties = value
        .get("properties")
        .expect("schema should expose 'properties'");
    // model_id and max_sequence_length are mandatory settings fields, so
    // they MUST show up in the generated schema.
    assert!(properties.get("model_id").is_some(), "{schema_str}");
    assert!(
        properties.get("max_sequence_length").is_some(),
        "{schema_str}"
    );
}

#[test]
fn runner_settings_proto_returns_proto_text() {
    let plugin = MmEmbeddingPlugin::new();
    let text = plugin.runner_settings_proto();
    assert!(text.contains("MultimodalEmbeddingRunnerSettings"));
}

#[test]
fn set_cancellation_token_does_not_panic() {
    // Sanity check: set_cancellation_token must accept a CancelToken without
    // requiring a runtime; the host invokes it on every job.
    use jobworkerp_plugin_abi::cancel::FfiCancellationToken;
    use jobworkerp_plugin_abi::v2::CancelToken;
    let mut plugin = MmEmbeddingPlugin::new();
    let (ffi, _handle) = FfiCancellationToken::new_owned();
    plugin.set_cancellation_token(CancelToken::from_ffi(ffi));
}

#[test]
fn run_stream_is_unsupported() {
    use jobworkerp_plugin_abi::sink::OutputSink;
    use jobworkerp_plugin_abi::v2::HighLevelSink;
    let mut plugin = MmEmbeddingPlugin::new();
    let (tx, _rx) = tokio::sync::mpsc::channel::<Vec<u8>>(1);
    let sink = HighLevelSink::from_ffi(OutputSink::from_sender(tx));
    let err = block_on(plugin.run_stream(
        vec![],
        HashMap::new(),
        Some(METHOD_EMBED_TEXT.to_string()),
        sink,
    ))
    .expect_err("expected streaming-unsupported error");
    assert!(err.starts_with("InvalidArgument:"), "got: {err}");
}
