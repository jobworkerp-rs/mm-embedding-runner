//! Loads the real Qwen3-VL-Embedding-2B from the HuggingFace Hub and runs a
//! tiny embed_texts round-trip to confirm the backend is wired correctly.
//!
//! Tagged `#[ignore]` so default `cargo test` skips it; run explicitly with
//! `cargo test --test integration_backend_load -- --ignored` once HF_HOME is
//! warm.

mod common;

use common::with_hf_home;
use mm_embedding_runner::backend::MmBackend;
use mm_embedding_runner::settings::{
    DTypeKind, DecodedImageBytesLimit, DeviceKind, ResolvedChunkingConfig, ValidatedSettings,
};

fn default_settings() -> ValidatedSettings {
    ValidatedSettings {
        model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        device: DeviceKind::Cpu,
        gpu_device_index: 0,
        dtype: DTypeKind::F32,
        max_sequence_length: 2048,
        image_size_limit_bytes: 10 * 1024 * 1024,
        max_decoded_image_bytes: DecodedImageBytesLimit::Default,
        allowed_image_dirs: vec![],
        tokenizer_model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        chunking_config: ResolvedChunkingConfig {
            max_chunk_tokens: 2048,
            min_chunk_tokens: 0,
        },
    }
}

#[test]
#[ignore]
fn full_load_succeeds() {
    let backend = MmBackend::load(default_settings()).expect("model should load");
    let vectors = backend
        .embed_texts(&["hello"])
        .expect("embed_texts should succeed");
    assert_eq!(vectors.len(), 1);
    // Qwen3-VL-Embedding-2B has 2048-dim output.
    assert_eq!(
        vectors[0].len(),
        2048,
        "expected 2048-dim embedding from Qwen3-VL-Embedding-2B",
    );
    // L2 normalized so norm ≈ 1.
    let norm_sq: f32 = vectors[0].iter().map(|v| v * v).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-3,
        "expected unit-norm embedding, got norm_sq={norm_sq}",
    );
}

/// End-to-end proof of the HF_HOME fix: with `HF_HOME` pointed at a fresh
/// directory, loading the backend must materialize the model weights under
/// `$HF_HOME/hub/models--Qwen--Qwen3-VL-Embedding-2B/`, the same tree the
/// tokenizer uses. Before the vendored-fork patch, `Qwen3VLEmbedding::from_hf`
/// pinned weights to `$HOME/.cache/huggingface/hub` and ignored `HF_HOME`.
///
/// Requires a real model download (and practically a GPU), so it is `#[ignore]`
/// and out of the per-PR lane. `HF_HOME` is process-global, so run with
/// `--test-threads=1`.
#[test]
#[ignore]
fn weights_land_under_hf_home() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let hf_home = tmp.path().join("models");

    with_hf_home(&hf_home, || {
        MmBackend::load(default_settings()).expect("model should load under HF_HOME");
    });

    // hf-hub lays the repo out as $HF_HOME/hub/models--<org>--<name>/.
    let repo_dir = hf_home
        .join("hub")
        .join("models--Qwen--Qwen3-VL-Embedding-2B");
    assert!(
        repo_dir.is_dir(),
        "weights should materialize under HF_HOME at {}",
        repo_dir.display()
    );
}

#[test]
#[ignore]
fn cuda_request_without_cuda_feature_is_unsupported() {
    if cfg!(feature = "cuda") {
        return;
    }
    let mut s = default_settings();
    s.device = DeviceKind::Cuda;
    let err = MmBackend::load(s).expect_err("should fail without cuda feature");
    assert!(
        err.to_string().starts_with("UnsupportedDevice:"),
        "got: {err}"
    );
}
