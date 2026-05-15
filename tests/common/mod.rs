//! Shared helpers for the `#[ignore]` integration / expensive tests.
//!
//! Each integration test is a separate binary, so a single loaded plugin
//! cannot be shared across files — but the device selection, fixture path,
//! and assertion helpers are identical everywhere and live here.

#![allow(dead_code)] // each test binary uses a different subset

use jobworkerp_client::plugins::MultiMethodPluginRunner;
use mm_embedding_runner::MmEmbeddingPlugin;
use mm_embedding_runner::protobuf::mm_embedding::{Device, MultimodalEmbeddingRunnerSettings};
use prost::Message;

pub const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

/// CPU inference of Qwen3-VL-Embedding-2B is impractically slow, so these
/// tests pick a device based on which acceleration feature this test binary
/// was built with, so `cargo test --features cuda` exercises CUDA while a
/// plain `cargo test` falls back to CPU instead of failing with
/// `UnsupportedDevice`. `MM_TEST_DEVICE=cpu|cuda|metal` forces a specific
/// device (and will surface `UnsupportedDevice` if that feature is absent,
/// which is the intended signal).
pub fn test_device() -> i32 {
    match std::env::var("MM_TEST_DEVICE").as_deref() {
        Ok("cpu") => return Device::Cpu as i32,
        Ok("cuda") => return Device::Cuda as i32,
        Ok("metal") => return Device::Metal as i32,
        _ => {}
    }
    if cfg!(feature = "cuda") {
        Device::Cuda as i32
    } else if cfg!(feature = "metal") {
        Device::Metal as i32
    } else {
        Device::Cpu as i32
    }
}

/// Build and load a plugin with the standard test settings, optionally
/// whitelisting directories for the `file_path` image source.
pub fn load_test_plugin(allowed_image_dirs: Vec<String>) -> MmEmbeddingPlugin {
    let mut plugin = MmEmbeddingPlugin::new();
    let settings = MultimodalEmbeddingRunnerSettings {
        model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        max_sequence_length: 512,
        device: test_device(),
        allowed_image_dirs,
        ..Default::default()
    };
    plugin
        .load(settings.encode_to_vec())
        .expect("model should load for integration tests");
    plugin
}

/// fastembed always L2-normalizes, so every produced vector must have unit
/// norm within float tolerance.
pub fn assert_unit_norm(values: &[f32]) {
    let norm_sq: f32 = values.iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-3,
        "expected unit-norm embedding, got norm_sq={norm_sq}",
    );
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}
