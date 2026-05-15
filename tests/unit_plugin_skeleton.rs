use jobworkerp_client::plugins::MultiMethodPluginRunner;
use mm_embedding_runner::{METHOD_EMBED_IMAGE, METHOD_EMBED_TEXT, MmEmbeddingPlugin, RUNNER_NAME};
use std::collections::HashMap;

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
    let (result, _meta) = plugin.run(vec![], HashMap::new(), None);
    let err = result.expect_err("expected InvalidArgument");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn run_with_unknown_method_yields_invalid_argument() {
    let mut plugin = MmEmbeddingPlugin::new();
    let (result, _meta) = plugin.run(vec![], HashMap::new(), Some("frobnicate"));
    let err = result.expect_err("expected InvalidArgument");
    assert!(
        err.to_string().starts_with("InvalidArgument:"),
        "got: {err}"
    );
}

#[test]
fn run_without_load_yields_not_loaded() {
    let mut plugin = MmEmbeddingPlugin::new();
    let (result, _meta) = plugin.run(vec![], HashMap::new(), Some(METHOD_EMBED_TEXT));
    let err = result.expect_err("expected NotLoaded");
    assert!(err.to_string().starts_with("NotLoaded:"), "got: {err}");
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
    for (name, schema) in &map {
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
    // model_id and max_sequence_length are mandatory fields per
    // specification.md §3.3; they MUST show up in the generated schema.
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
fn cancel_returns_false() {
    let mut plugin = MmEmbeddingPlugin::new();
    assert!(!plugin.cancel());
    assert!(!plugin.is_canceled());
}
