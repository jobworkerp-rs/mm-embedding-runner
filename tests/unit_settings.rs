use mm_embedding_runner::protobuf::mm_embedding::{
    ChunkingConfig as ProtoChunkingConfig, DType as ProtoDType, Device as ProtoDevice,
    MultimodalEmbeddingRunnerSettings,
};
use mm_embedding_runner::settings::{
    DEFAULT_IMAGE_SIZE_LIMIT_BYTES, DEFAULT_MAX_DECODED_IMAGE_BYTES, DTypeKind,
    DecodedImageBytesLimit, DeviceKind, MAX_SEQ_LEN_MAX, MAX_SEQ_LEN_MIN, validate,
};

fn base_settings() -> MultimodalEmbeddingRunnerSettings {
    MultimodalEmbeddingRunnerSettings {
        model_id: "Qwen/Qwen3-VL-Embedding-2B".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: None,
        max_decoded_image_bytes: None,
        allowed_image_dirs: vec![],
        tokenizer_model_id: None,
        chunking_config: None,
    }
}

#[test]
fn default_settings_resolve_to_defaults() {
    let v = validate(base_settings()).unwrap();
    assert_eq!(v.max_sequence_length, 2048);
    assert_eq!(v.image_size_limit_bytes, DEFAULT_IMAGE_SIZE_LIMIT_BYTES);
    assert_eq!(v.max_decoded_image_bytes, DecodedImageBytesLimit::Default);
    assert_eq!(
        v.max_decoded_image_bytes.effective_bytes(),
        Some(DEFAULT_MAX_DECODED_IMAGE_BYTES)
    );
    assert_eq!(v.device, DeviceKind::Cpu);
    assert_eq!(v.dtype, DTypeKind::F32);
    assert_eq!(v.chunking_config.max_chunk_tokens, 2048);
    assert_eq!(v.chunking_config.min_chunk_tokens, 0);
    assert_eq!(v.tokenizer_model_id, v.model_id);
}

#[test]
fn empty_model_id_is_rejected() {
    let mut s = base_settings();
    s.model_id = String::new();
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn whitespace_only_model_id_is_rejected() {
    let mut s = base_settings();
    s.model_id = "   \t\n".to_string();
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn max_sequence_length_below_minimum_is_rejected() {
    let mut s = base_settings();
    s.max_sequence_length = 15;
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn max_sequence_length_above_maximum_is_rejected() {
    let mut s = base_settings();
    s.max_sequence_length = 32_769;
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn max_sequence_length_boundary_min_is_accepted() {
    let mut s = base_settings();
    s.max_sequence_length = MAX_SEQ_LEN_MIN;
    let v = validate(s).unwrap();
    assert_eq!(v.max_sequence_length, MAX_SEQ_LEN_MIN);
    assert_eq!(v.chunking_config.max_chunk_tokens, MAX_SEQ_LEN_MIN);
}

#[test]
fn max_sequence_length_boundary_max_is_accepted() {
    let mut s = base_settings();
    s.max_sequence_length = MAX_SEQ_LEN_MAX;
    let v = validate(s).unwrap();
    assert_eq!(v.max_sequence_length, MAX_SEQ_LEN_MAX);
}

#[test]
fn unset_max_sequence_length_defaults_to_2048() {
    let mut s = base_settings();
    s.max_sequence_length = 0;
    let v = validate(s).unwrap();
    assert_eq!(v.max_sequence_length, 2048);
}

#[test]
fn chunking_config_max_zero_is_rejected() {
    let mut s = base_settings();
    s.chunking_config = Some(ProtoChunkingConfig {
        max_chunk_tokens: Some(0),
        min_chunk_tokens: Some(0),
    });
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn chunking_config_max_exceeds_seq_len_is_rejected() {
    let mut s = base_settings();
    s.max_sequence_length = 1024;
    s.chunking_config = Some(ProtoChunkingConfig {
        max_chunk_tokens: Some(2048),
        min_chunk_tokens: Some(0),
    });
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn chunking_config_min_equal_to_max_is_rejected() {
    let mut s = base_settings();
    s.chunking_config = Some(ProtoChunkingConfig {
        max_chunk_tokens: Some(512),
        min_chunk_tokens: Some(512),
    });
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn chunking_config_min_greater_than_max_is_rejected() {
    let mut s = base_settings();
    s.chunking_config = Some(ProtoChunkingConfig {
        max_chunk_tokens: Some(256),
        min_chunk_tokens: Some(512),
    });
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn allowed_image_dirs_unreachable_path_is_rejected() {
    let mut s = base_settings();
    s.allowed_image_dirs = vec!["/nonexistent/path/should/never/exist/xyz".to_string()];
    let err = validate(s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("allowed_image_dirs"));
}

#[test]
fn allowed_image_dirs_real_path_canonicalized() {
    let tmp = tempfile::tempdir().unwrap();
    let mut s = base_settings();
    s.allowed_image_dirs = vec![tmp.path().to_string_lossy().into_owned()];
    let v = validate(s).unwrap();
    assert_eq!(v.allowed_image_dirs.len(), 1);
    assert_eq!(
        v.allowed_image_dirs[0],
        tmp.path().canonicalize().unwrap(),
        "stored dir must be canonical",
    );
}

#[test]
fn unlimited_decoded_bytes_round_trips() {
    let mut s = base_settings();
    s.max_decoded_image_bytes = Some(0);
    let v = validate(s).unwrap();
    assert_eq!(v.max_decoded_image_bytes, DecodedImageBytesLimit::Unlimited);
    assert!(v.max_decoded_image_bytes.effective_bytes().is_none());
}

#[test]
fn explicit_decoded_bytes_round_trips() {
    let mut s = base_settings();
    s.max_decoded_image_bytes = Some(123_456);
    let v = validate(s).unwrap();
    assert_eq!(
        v.max_decoded_image_bytes,
        DecodedImageBytesLimit::Limited(123_456)
    );
    assert_eq!(v.max_decoded_image_bytes.effective_bytes(), Some(123_456));
}

#[test]
fn cuda_device_is_passed_through() {
    let mut s = base_settings();
    s.device = ProtoDevice::Cuda as i32;
    s.gpu_device_index = Some(2);
    let v = validate(s).unwrap();
    assert_eq!(v.device, DeviceKind::Cuda);
    assert_eq!(v.gpu_device_index, 2);
}

#[test]
fn whitespace_tokenizer_model_id_falls_back_to_model_id() {
    let mut s = base_settings();
    s.tokenizer_model_id = Some("   ".to_string());
    let v = validate(s).unwrap();
    assert_eq!(v.tokenizer_model_id, v.model_id);
}
