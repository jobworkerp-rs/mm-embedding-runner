//! End-to-end check of the `embed_image` method path. Mirrors
//! `integration_embed_text.rs` in shape: share a single loaded plugin
//! across tests, verify spec contract on the result protobuf.

use jobworkerp_client::plugins::MultiMethodPluginRunner;
use mm_embedding_runner::protobuf::mm_embedding::{
    Device, EmbedImageArgs, ImageItem, MmEmbeddingResult, MultimodalEmbeddingRunnerSettings,
    image_item::Source as ImageSource,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, MmEmbeddingPlugin};
use prost::Message;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

fn test_device() -> i32 {
    match std::env::var("MM_TEST_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu as i32,
        Ok("metal") => Device::Metal as i32,
        _ => Device::Cuda as i32,
    }
}

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

static LOADED_PLUGIN: OnceLock<std::sync::Mutex<MmEmbeddingPlugin>> = OnceLock::new();

fn shared_plugin() -> std::sync::MutexGuard<'static, MmEmbeddingPlugin> {
    LOADED_PLUGIN
        .get_or_init(|| {
            let mut plugin = MmEmbeddingPlugin::new();
            let fixtures_path = PathBuf::from(FIXTURES);
            // For file_path tests we need the fixtures dir explicitly listed.
            let settings = MultimodalEmbeddingRunnerSettings {
                model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
                max_sequence_length: 512,
                device: test_device(),
                allowed_image_dirs: vec![fixtures_path.to_string_lossy().into_owned()],
                ..Default::default()
            };
            plugin
                .load(settings.encode_to_vec())
                .expect("model should load for integration tests");
            std::sync::Mutex::new(plugin)
        })
        .lock()
        .unwrap()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(FIXTURES).join(name)
}

fn read_fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap()
}

fn run_embed_image(plugin: &mut MmEmbeddingPlugin, items: Vec<ImageItem>) -> MmEmbeddingResult {
    let args = EmbedImageArgs { items }.encode_to_vec();
    let (result, _meta) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let bytes = result.expect("embed_image should succeed");
    MmEmbeddingResult::decode(&bytes[..]).unwrap()
}

fn assert_unit_norm(values: &[f32]) {
    let norm_sq: f32 = values.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-3,
        "expected unit-norm embedding, got norm_sq={norm_sq}",
    );
}

#[test]
#[ignore]
fn data_source_returns_one_embedding_per_item() {
    let mut plugin = shared_plugin();
    let bytes = read_fixture_bytes("tiny_valid.png");
    let items = vec![
        ImageItem {
            source: Some(ImageSource::Data(bytes.clone())),
            label: None,
        },
        ImageItem {
            source: Some(ImageSource::Data(bytes.clone())),
            label: Some("second".to_string()),
        },
    ];
    let out = run_embed_image(&mut plugin, items);
    assert_eq!(out.embeddings.len(), 2);
    for (i, emb) in out.embeddings.iter().enumerate() {
        assert_eq!(emb.begin_position as usize, i);
        assert_eq!(emb.end_position as usize, i + 1);
        assert_eq!(emb.values.len(), 2048);
        assert_unit_norm(&emb.values);
    }
    // First item without label -> "image[#0]" hint
    assert_eq!(out.embeddings[0].content, "image[#0]");
    // Second item with label -> label wins
    assert_eq!(out.embeddings[1].content, "second");
}

#[test]
#[ignore]
fn file_path_source_succeeds_within_allowed_dirs() {
    let mut plugin = shared_plugin();
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            fixture("tiny_valid.png").to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let out = run_embed_image(&mut plugin, items);
    assert_eq!(out.embeddings.len(), 1);
    let emb = &out.embeddings[0];
    assert_eq!(emb.begin_position, 0);
    assert_eq!(emb.end_position, 1);
    assert!(
        emb.content.ends_with("tiny_valid.png"),
        "expected content to reflect file path, got {}",
        emb.content
    );
    assert_eq!(emb.values.len(), 2048);
    assert_unit_norm(&emb.values);
}

#[test]
#[ignore]
fn empty_items_rejected() {
    let mut plugin = shared_plugin();
    let args = EmbedImageArgs { items: vec![] }.encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let err = result.expect_err("expected InvalidArgument");
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
#[ignore]
fn url_source_rejected() {
    let mut plugin = shared_plugin();
    let items = vec![ImageItem {
        source: Some(ImageSource::Url("https://example.com/x.png".to_string())),
        label: None,
    }];
    let args = EmbedImageArgs { items }.encode_to_vec();
    let (result, _) = plugin.run(args, HashMap::new(), Some(METHOD_EMBED_IMAGE));
    let err = result.expect_err("expected InvalidArgument");
    assert!(err.to_string().starts_with("InvalidArgument:"));
}
