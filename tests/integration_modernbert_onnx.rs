//! End-to-end checks for HuggingFace ModernBertModel repositories exported as
//! ONNX. These tests download a real model and are therefore ignored by
//! default.

mod common;

use common::{assert_unit_norm, test_device};
use mm_embedding_runner::backend::MmBackend;
use mm_embedding_runner::settings::{
    DTypeKind, DecodedImageBytesLimit, DeviceKind, OnnxPoolingKind, ResolvedChunkingConfig,
    ValidatedSettings,
};

const RURI_ONNX_MODEL_ID: &str = "sirasagi62/ruri-v3-310m-ONNX";

fn modernbert_settings(onnx_model_file: &str) -> ValidatedSettings {
    ValidatedSettings {
        model_id: RURI_ONNX_MODEL_ID.to_string(),
        device: match test_device() {
            x if x == mm_embedding_runner::protobuf::mm_embedding::Device::Cuda as i32 => {
                DeviceKind::Cuda
            }
            x if x == mm_embedding_runner::protobuf::mm_embedding::Device::Metal as i32 => {
                DeviceKind::Metal
            }
            _ => DeviceKind::Cpu,
        },
        gpu_device_index: 0,
        dtype: DTypeKind::F32,
        max_sequence_length: 512,
        image_size_limit_bytes: 10 * 1024 * 1024,
        max_decoded_image_bytes: DecodedImageBytesLimit::Default,
        allowed_image_dirs: vec![],
        tokenizer_model_id: RURI_ONNX_MODEL_ID.to_string(),
        onnx_model_file: onnx_model_file.to_string(),
        onnx_pooling: OnnxPoolingKind::Mean,
        chunking_config: ResolvedChunkingConfig {
            max_chunk_tokens: 512,
            min_chunk_tokens: 0,
        },
    }
}

#[test]
#[ignore]
fn ruri_modernbert_onnx_embeds_text() {
    let backend = MmBackend::load(modernbert_settings("onnx/model.onnx"))
        .expect("ModernBERT ONNX model should load");
    assert!(!backend.supports_vision());

    let vectors = backend
        .embed_texts(&["検索クエリ: 瑠璃色はどんな色ですか？"])
        .expect("embed_texts should succeed");

    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), backend.embedding_dimension());
    assert_unit_norm(&vectors[0]);
}

#[test]
#[ignore]
fn ruri_modernbert_int8_onnx_embeds_text_when_supported_by_ort() {
    let backend = MmBackend::load(modernbert_settings("onnx/model_int8.onnx"))
        .expect("ModernBERT INT8 ONNX model should load when ORT supports its operators");
    let vectors = backend
        .embed_texts(&["検索クエリ: RustでONNXを実行する"])
        .expect("embed_texts should succeed");

    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].len(), backend.embedding_dimension());
    assert_unit_norm(&vectors[0]);
}
