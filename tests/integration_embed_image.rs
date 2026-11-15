//! End-to-end check of the `embed_image` method path. Mirrors
//! `integration_embed_text.rs` in shape: share a single loaded plugin
//! across tests, and verify the result protobuf contract: one embedding per
//! input image in submission order, with the input index as position.

mod common;

use common::{FIXTURES, assert_unit_norm, load_test_plugin, run_method};
use mm_embedding_runner::protobuf::mm_embedding::{
    EmbedImageArgs, ImageItem, MmEmbeddingResult, image_item::Source as ImageSource,
};
use mm_embedding_runner::{METHOD_EMBED_IMAGE, MmEmbeddingPlugin};
use prost::Message;
use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

static LOADED_PLUGIN: OnceLock<Mutex<MmEmbeddingPlugin>> = OnceLock::new();

fn shared_plugin() -> MutexGuard<'static, MmEmbeddingPlugin> {
    LOADED_PLUGIN
        .get_or_init(|| {
            // file_path tests need the fixtures dir explicitly whitelisted.
            Mutex::new(load_test_plugin(vec![FIXTURES.to_string()]))
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
    let bytes = run_method(plugin, METHOD_EMBED_IMAGE, args).expect("embed_image should succeed");
    MmEmbeddingResult::decode(&bytes[..]).unwrap()
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
    let err =
        run_method(&mut plugin, METHOD_EMBED_IMAGE, args).expect_err("expected InvalidArgument");
    assert!(err.starts_with("InvalidArgument:"));
}

#[test]
#[ignore]
fn non_http_url_scheme_rejected() {
    let mut plugin = shared_plugin();
    let items = vec![ImageItem {
        source: Some(ImageSource::Url("file:///etc/passwd".to_string())),
        label: None,
    }];
    let args = EmbedImageArgs { items }.encode_to_vec();
    let err =
        run_method(&mut plugin, METHOD_EMBED_IMAGE, args).expect_err("expected InvalidArgument");
    assert!(err.starts_with("InvalidArgument:"));
    assert!(err.contains("http or https"), "got: {err}");
}

// Serve tests/fixtures/tiny_valid.png over a throwaway loopback HTTP server
// and confirm the url source fetches + embeds it end to end.
#[test]
#[ignore]
fn http_url_source_fetches_and_embeds() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let png = read_fixture_bytes("tiny_valid.png");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let body = png.clone();
    let server = std::thread::spawn(move || {
        // Single request, then exit.
        let Ok((mut stream, _)) = listener.accept() else {
            return;
        };
        // Drain the request until the header terminator so the response is
        // not written before the client finished sending; bail on read
        // error rather than racing ahead.
        let mut req = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            match stream.read(&mut byte) {
                Ok(0) => break,
                Ok(_) => {
                    req.push(byte[0]);
                    if req.ends_with(b"\r\n\r\n") {
                        break;
                    }
                }
                Err(_) => return,
            }
        }
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        if stream.write_all(header.as_bytes()).is_err() {
            return;
        }
        let _ = stream.write_all(&body);
        let _ = stream.flush();
    });

    let mut plugin = shared_plugin();
    let items = vec![ImageItem {
        source: Some(ImageSource::Url(format!(
            "http://127.0.0.1:{port}/tiny_valid.png"
        ))),
        label: Some("from-url".to_string()),
    }];
    let out = run_embed_image(&mut plugin, items);
    server.join().unwrap();

    assert_eq!(out.embeddings.len(), 1);
    let emb = &out.embeddings[0];
    assert_eq!(emb.begin_position, 0);
    assert_eq!(emb.end_position, 1);
    assert_eq!(emb.content, "from-url");
    assert_eq!(emb.values.len(), 2048);
    assert_unit_norm(&emb.values);
}
