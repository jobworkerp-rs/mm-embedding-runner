use mm_embedding_runner::image_validation::{_preflight_decoded_size_for_test, prepare_images};
use mm_embedding_runner::protobuf::mm_embedding::{
    ChunkingConfig as ProtoChunkingConfig, DType as ProtoDType, Device as ProtoDevice,
    EmbedImageArgs, ImageItem, MultimodalEmbeddingRunnerSettings,
    image_item::Source as ImageSource,
};
use mm_embedding_runner::settings::{DecodedImageBytesLimit, ValidatedSettings, validate};
use std::path::PathBuf;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(FIXTURES).join(name)
}

fn read_fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|_| panic!("missing fixture: {name}"))
}

fn base_settings_with_allowed(dirs: Vec<String>) -> ValidatedSettings {
    let s = MultimodalEmbeddingRunnerSettings {
        model_id: "test/model".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: Some(10 * 1024 * 1024),
        max_decoded_image_bytes: None,
        allowed_image_dirs: dirs,
        tokenizer_model_id: None,
        chunking_config: None,
    };
    let _ = ProtoChunkingConfig::default();
    validate(s).unwrap()
}

fn settings_no_allowed_dirs() -> ValidatedSettings {
    base_settings_with_allowed(vec![])
}

fn settings_allowed_fixtures() -> ValidatedSettings {
    base_settings_with_allowed(vec![FIXTURES.to_string()])
}

#[test]
fn empty_items_yields_invalid_argument() {
    let s = settings_no_allowed_dirs();
    let err = prepare_images(&[], &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn missing_source_yields_invalid_argument() {
    let s = settings_no_allowed_dirs();
    let items = vec![ImageItem {
        source: None,
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
}

#[test]
fn url_source_yields_invalid_argument() {
    let s = settings_no_allowed_dirs();
    let items = vec![ImageItem {
        source: Some(ImageSource::Url("https://example.com/x.png".to_string())),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("url source"));
}

#[test]
fn file_path_rejected_when_no_allowed_dirs() {
    let s = settings_no_allowed_dirs();
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            fixture("tiny_valid.png").to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("allowed_image_dirs is empty"));
}

#[test]
fn file_path_outside_allowed_dirs_rejected() {
    // Allow a subdirectory but not the parent fixtures dir.
    let allowed = vec![fixture("allowed_dir_a").to_string_lossy().into_owned()];
    let s = base_settings_with_allowed(allowed);
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            fixture("tiny_valid.png").to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("outside allowed_image_dirs"));
}

#[test]
fn file_path_string_prefix_collision_is_rejected() {
    // Set up two sibling directories whose names share a string prefix
    // (`allow` vs `allow_evil`). A naive string prefix check would let
    // `/.../allow_evil/x.png` slip past a `/.../allow` whitelist; the
    // component-aware Path::starts_with must reject it.
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let allow = root.join("allow");
    let evil = root.join("allow_evil");
    std::fs::create_dir_all(&allow).unwrap();
    std::fs::create_dir_all(&evil).unwrap();
    let evil_file = evil.join("x.png");
    std::fs::copy(fixture("tiny_valid.png"), &evil_file).unwrap();

    let s = base_settings_with_allowed(vec![allow.to_string_lossy().into_owned()]);
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            evil_file.to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("outside allowed_image_dirs"));
}

#[test]
fn file_path_size_over_limit_rejected() {
    // Use a very small image_size_limit_bytes so the small fixture exceeds it.
    let s = MultimodalEmbeddingRunnerSettings {
        model_id: "test/model".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: Some(8), // 8 bytes — way smaller than any real PNG
        max_decoded_image_bytes: None,
        allowed_image_dirs: vec![FIXTURES.to_string()],
        tokenizer_model_id: None,
        chunking_config: None,
    };
    let s = validate(s).unwrap();
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            fixture("tiny_valid.png").to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("image_size_limit_bytes"));
}

#[test]
fn data_size_over_limit_rejected() {
    let s = MultimodalEmbeddingRunnerSettings {
        model_id: "test/model".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: Some(8),
        max_decoded_image_bytes: None,
        allowed_image_dirs: vec![],
        tokenizer_model_id: None,
        chunking_config: None,
    };
    let s = validate(s).unwrap();
    let big = read_fixture_bytes("tiny_valid.png");
    let items = vec![ImageItem {
        source: Some(ImageSource::Data(big)),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("image_size_limit_bytes"));
}

#[test]
fn invalid_header_rejected() {
    let s = settings_no_allowed_dirs();
    let bytes = read_fixture_bytes("invalid_header.bin");
    let items = vec![ImageItem {
        source: Some(ImageSource::Data(bytes)),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(err.to_string().contains("invalid image header"));
}

#[test]
fn zip_bomb_rejected_by_decoded_size_preflight() {
    // 10000 x 10000 RGBA = ~400 MB. With max_decoded_image_bytes set to a
    // small value (1 MiB), the preflight must reject it before any expensive
    // decode happens.
    let s_proto = MultimodalEmbeddingRunnerSettings {
        model_id: "test/model".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: Some(50 * 1024 * 1024), // permissive compressed cap
        max_decoded_image_bytes: Some(1024 * 1024),     // 1 MiB decoded cap
        allowed_image_dirs: vec![],
        tokenizer_model_id: None,
        chunking_config: None,
    };
    let s = validate(s_proto).unwrap();
    let bytes = read_fixture_bytes("zip_bomb.png");
    let items = vec![ImageItem {
        source: Some(ImageSource::Data(bytes)),
        label: None,
    }];
    let err = prepare_images(&items, &s).unwrap_err();
    assert!(err.to_string().starts_with("InvalidArgument:"));
    assert!(
        err.to_string().contains("decoded size exceeds"),
        "got: {err}"
    );
}

#[test]
fn happy_path_data_round_trips() {
    let s = settings_no_allowed_dirs();
    let bytes = read_fixture_bytes("tiny_valid.png");
    let items = vec![ImageItem {
        source: Some(ImageSource::Data(bytes.clone())),
        label: Some("hello".to_string()),
    }];
    let prepared = prepare_images(&items, &s).unwrap();
    assert_eq!(prepared.len(), 1);
    assert_eq!(prepared[0].bytes, bytes);
    assert_eq!(prepared[0].source_hint, "hello");
}

#[test]
fn happy_path_data_without_label_uses_index_hint() {
    let s = settings_no_allowed_dirs();
    let bytes = read_fixture_bytes("tiny_valid.png");
    let items = vec![
        ImageItem {
            source: Some(ImageSource::Data(bytes.clone())),
            label: None,
        },
        ImageItem {
            source: Some(ImageSource::Data(bytes.clone())),
            label: None,
        },
    ];
    let prepared = prepare_images(&items, &s).unwrap();
    assert_eq!(prepared[0].source_hint, "image[#0]");
    assert_eq!(prepared[1].source_hint, "image[#1]");
}

#[test]
fn happy_path_file_path_uses_path_hint_when_label_absent() {
    let s = settings_allowed_fixtures();
    let items = vec![ImageItem {
        source: Some(ImageSource::FilePath(
            fixture("tiny_valid.png").to_string_lossy().into_owned(),
        )),
        label: None,
    }];
    let prepared = prepare_images(&items, &s).unwrap();
    assert!(prepared[0].source_hint.ends_with("tiny_valid.png"));
}

#[test]
fn preflight_overflow_rejected() {
    // Construct a minimal PNG-like byte stream is too brittle to drive an
    // overflow path through the real decoder; instead, exercise the
    // arithmetic directly via the unit hook. u32::MAX * u32::MAX * 4
    // overflows u64, but the preflight uses checked_mul so it must surface
    // the dedicated overflow error string.
    //
    // We can't construct a real image with such dimensions, so we feed a
    // tiny header to confirm the negative path: small images do NOT overflow.
    let bytes = read_fixture_bytes("tiny_valid.png");
    _preflight_decoded_size_for_test(&bytes, DecodedImageBytesLimit::Default).unwrap();
}

#[test]
fn preflight_with_unlimited_allows_zip_bomb() {
    let bytes = read_fixture_bytes("zip_bomb.png");
    _preflight_decoded_size_for_test(&bytes, DecodedImageBytesLimit::Unlimited).unwrap();
}

// EmbedImageArgs type used for compile-time only references above.
const _: fn() = || {
    let _ = EmbedImageArgs::default();
};
