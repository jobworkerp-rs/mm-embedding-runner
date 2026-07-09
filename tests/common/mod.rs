//! Shared helpers for the `#[ignore]` integration / expensive tests.
//!
//! Each integration test is a separate binary, so a single loaded plugin
//! cannot be shared across files — but the device selection, fixture path,
//! and assertion helpers are identical everywhere and live here.

#![allow(dead_code)] // each test binary uses a different subset

use jobworkerp_plugin_abi::v2::PluginV2;
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

/// Build and load a plugin with the standard test settings against
/// `model_id`, optionally whitelisting directories for the `file_path`
/// image source. V2 `load` is async; tests outside the plugin's own
/// runtime drive it via a short-lived tokio runtime so the helper can
/// stay synchronous.
pub fn load_plugin_with_model(
    model_id: &str,
    allowed_image_dirs: Vec<String>,
) -> MmEmbeddingPlugin {
    let mut plugin = MmEmbeddingPlugin::new();
    let settings = MultimodalEmbeddingRunnerSettings {
        model_id: model_id.to_string(),
        max_sequence_length: 512,
        device: test_device(),
        allowed_image_dirs,
        onnx_model_file: None,
        onnx_pooling: Default::default(),
        ..Default::default()
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test driver runtime");
    rt.block_on(plugin.load(settings.encode_to_vec()))
        .expect("model should load for integration tests");
    plugin
}

/// Convenience wrapper that loads the default multimodal model
/// (`Qwen/Qwen3-VL-Embedding-2B`) — used by every legacy integration
/// test written before the multi-architecture branch.
pub fn load_test_plugin(allowed_image_dirs: Vec<String>) -> MmEmbeddingPlugin {
    load_plugin_with_model("Qwen/Qwen3-VL-Embedding-2B", allowed_image_dirs)
}

/// Drive a `PluginV2` async method from a sync test body. Each call builds
/// a private current-thread runtime so concurrent test cases (and the
/// plugin's own internal runtime, when load() builds one) never share
/// reactor state — that sharing would otherwise mask leaks between cases.
pub fn block_on<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test driver runtime")
        .block_on(fut)
}

/// `block_on(plugin.run(args, {}, Some(method)))` wrapper used by every
/// test that calls `run` — folds the `.to_string()` + empty-metadata noise
/// at the call site.
pub fn run_method(
    plugin: &mut MmEmbeddingPlugin,
    method: &str,
    args: Vec<u8>,
) -> Result<Vec<u8>, String> {
    block_on(plugin.run(
        args,
        std::collections::HashMap::new(),
        Some(method.to_string()),
    ))
    .0
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

/// Run `body` with `HF_HOME` set to `value`, restoring the previous value
/// afterwards even if `body` panics. `HF_HOME` is process-global, so callers
/// must run single-threaded (`--test-threads=1`).
pub fn with_hf_home<R>(value: &std::path::Path, body: impl FnOnce() -> R) -> R {
    let prev = std::env::var_os("HF_HOME");
    unsafe {
        std::env::set_var("HF_HOME", value);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
    unsafe {
        match prev {
            Some(v) => std::env::set_var("HF_HOME", v),
            None => std::env::remove_var("HF_HOME"),
        }
    }
    match result {
        Ok(r) => r,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}
