use mm_embedding_runner::image_validation::prepare_images;
use mm_embedding_runner::protobuf::mm_embedding::{
    DType as ProtoDType, Device as ProtoDevice, ImageItem, MultimodalEmbeddingRunnerSettings,
    image_item::Source as ImageSource,
};
use mm_embedding_runner::settings::{ValidatedSettings, validate};
use std::path::PathBuf;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(FIXTURES).join(name)
}

fn read_fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|_| panic!("missing fixture: {name}"))
}

fn settings(
    dirs: Vec<String>,
    image_size_limit_bytes: Option<u64>,
    max_decoded_image_bytes: Option<u64>,
) -> ValidatedSettings {
    let s = MultimodalEmbeddingRunnerSettings {
        model_id: "test/model".to_string(),
        device: ProtoDevice::Cpu as i32,
        gpu_device_index: None,
        dtype: ProtoDType::F32 as i32,
        max_sequence_length: 2048,
        image_size_limit_bytes: image_size_limit_bytes.or(Some(10 * 1024 * 1024)),
        max_decoded_image_bytes,
        allowed_image_dirs: dirs,
        tokenizer_model_id: None,
        chunking_config: None,
    };
    validate(s).unwrap()
}

fn base_settings_with_allowed(dirs: Vec<String>) -> ValidatedSettings {
    settings(dirs, None, None)
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
fn non_http_url_scheme_is_rejected_without_network() {
    // file://, data:, ftp:// etc. must be rejected by the scheme check
    // before any network I/O happens.
    let s = settings_no_allowed_dirs();
    for url in [
        "file:///etc/passwd",
        "ftp://example.com/x.png",
        "data:image/png;base64,AAAA",
        "javascript:alert(1)",
    ] {
        let items = vec![ImageItem {
            source: Some(ImageSource::Url(url.to_string())),
            label: None,
        }];
        let err = prepare_images(&items, &s).unwrap_err();
        assert!(
            err.to_string().starts_with("InvalidArgument:"),
            "scheme {url} should be rejected, got: {err}"
        );
        assert!(err.to_string().contains("http or https"), "got: {err}");
    }
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
    // 8-byte compressed cap is smaller than any real PNG.
    let s = settings(vec![FIXTURES.to_string()], Some(8), None);
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
    let s = settings(vec![], Some(8), None);
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
    // 10000x10000 RGBA ≈ 400 MB decoded. A 1 MiB decoded cap (with a
    // permissive compressed cap) must reject it before any real decode.
    let s = settings(vec![], Some(50 * 1024 * 1024), Some(1024 * 1024));
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
